#!/usr/bin/env python3
"""Apply frozen v5 model weights and thresholds to a new test-only dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_context_study as base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(record: dict[str, Any], artifact: dict[str, Any]) -> list[float]:
    encoder = artifact["encoder"]
    row = [1.0]
    for field in encoder["categorical"]:
        value = str(base.get_field(record, field))
        categories = encoder["categories"][field]
        row.extend(1.0 if value == category else 0.0 for category in categories)
        row.append(0.0 if value in categories else 1.0)
    for field in encoder["numeric"]:
        value = float(base.get_field(record, field))
        row.append((value - float(encoder["means"][field])) / float(encoder["scales"][field]))
    if len(row) != len(encoder["feature_names"]):
        raise ValueError("encoded feature length mismatch")
    return row


def score(record: dict[str, Any], artifact: dict[str, Any]) -> float:
    row = encode(record, artifact)
    names = artifact["encoder"]["feature_names"]
    weights = artifact["weights"]
    return base.sigmoid(sum(value * float(weights[name]) for name, value in zip(names, row)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--locked-model-artifacts", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--baseline", default="operation_only")
    parser.add_argument("--candidate", default="measured_process_context")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.output_dir.exists():
            raise ValueError(f"refusing to overwrite {args.output_dir}")
        if sha256_file(args.manifest) != args.expected_manifest_sha256:
            raise ValueError("replication manifest hash mismatch")
        if sha256_file(args.locked_model_artifacts) != args.expected_model_sha256:
            raise ValueError("locked model artifact hash mismatch")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
        model_file = json.loads(args.locked_model_artifacts.read_text(encoding="utf-8-sig"))
        records = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        expected_records = int(manifest["design"]["records"])
        if len(records) != expected_records:
            raise ValueError(f"expected {expected_records} records, found {len(records)}")
        if any(record.get("split") != "test" for record in records):
            raise ValueError("locked replication accepts test-only records")
        if any(record.get("synthetic") or not record.get("claim_eligible") for record in records):
            raise ValueError("replication input contains non-claim-eligible records")
        expected: dict[str, dict[str, str]] = {}
        for pair in manifest["pairs"]:
            common = {
                "pair_id": str(pair["pair_id"]),
                "workload_group": str(pair["workload_group"]),
                "split": str(pair["split"]),
                "operation": str(pair["operation"]),
            }
            expected[str(pair["legitimate_run_id"])] = {**common, "label": "legitimate_admin"}
            expected[str(pair["attack_run_id"])] = {**common, "label": "attack_preparation"}
        observed_ids = {str(record["record_id"]) for record in records}
        if observed_ids != set(expected):
            raise ValueError("record IDs differ from the frozen replication manifest")
        for record in records:
            frozen = expected[str(record["record_id"])]
            for field, value in frozen.items():
                if str(record.get(field)) != value:
                    raise ValueError(f"{record['record_id']}: {field} differs from frozen manifest")
        by_pair: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            by_pair.setdefault(str(record["pair_id"]), []).append(record)
        if len(by_pair) != int(manifest["design"]["pairs"]):
            raise ValueError("pair count differs from frozen design")
        for pair_id, pair in by_pair.items():
            if len(pair) != 2 or {r["label"] for r in pair} != {"legitimate_admin", "attack_preparation"}:
                raise ValueError(f"invalid matched pair {pair_id}")
            if len({r["operation"] for r in pair}) != 1:
                raise ValueError(f"operation mismatch in {pair_id}")

        selected_models = [args.baseline, args.candidate]
        artifacts = model_file["models"]
        if any(name not in artifacts for name in selected_models):
            raise ValueError("requested model is absent from frozen artifact")
        labels = [1 if record["label"] == "attack_preparation" else 0 for record in records]
        report_models: dict[str, Any] = {}
        prediction_rows: list[dict[str, Any]] = []
        for name in selected_models:
            artifact = artifacts[name]
            scores = [score(record, artifact) for record in records]
            threshold = float(artifact["threshold"])
            metrics = base.confusion(labels, scores, threshold)
            report_models[name] = {
                "locked_threshold": threshold,
                "test": {
                    "metrics": metrics,
                    "auc": base.auc(labels, scores),
                    "pair_ranking": base.pair_ranking(records, scores),
                    "compromised_admin_audit": base.audit_subgroup(records, scores, threshold, "compromised"),
                    "other_attack_audit": base.audit_subgroup(records, scores, threshold, "not_applicable"),
                },
            }
            for record, current_score in zip(records, scores):
                prediction_rows.append(
                    {
                        "model": name,
                        "record_id": record["record_id"],
                        "pair_id": record["pair_id"],
                        "workload_group": record["workload_group"],
                        "split": "test",
                        "operation": record["operation"],
                        "label": record["label"],
                        "account_state_label": record["account_state_label"],
                        "score": f"{current_score:.17g}",
                        "threshold": f"{threshold:.17g}",
                        "alerted": str(current_score >= threshold).lower(),
                    }
                )
        baseline_metrics = report_models[args.baseline]["test"]["metrics"]
        candidate_metrics = report_models[args.candidate]["test"]["metrics"]
        report = {
            "schema_version": "rivf-locked-temporal-replication-result-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "confirmatory_temporal_replication_candidate_requires_author_review",
            "no_retraining": True,
            "claim_boundary": manifest["claim_boundary"],
            "input_sha256": sha256_file(args.input),
            "manifest_sha256": sha256_file(args.manifest),
            "locked_model_artifacts_sha256": sha256_file(args.locked_model_artifacts),
            "records": len(records),
            "pairs": len(by_pair),
            "models": report_models,
            "primary_comparison": {
                "baseline_model": args.baseline,
                "candidate_model": args.candidate,
                "delta_tpr": candidate_metrics["tpr"] - baseline_metrics["tpr"],
                "delta_fpr": candidate_metrics["fpr"] - baseline_metrics["fpr"],
                "delta_mcc": candidate_metrics["mcc"] - baseline_metrics["mcc"],
            },
        }
        args.output_dir.mkdir(parents=True)
        report_path = args.output_dir / "model-report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prediction_path = args.output_dir / "predictions.csv"
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
        candidate = report_models[args.candidate]["test"]
        text = [
            "# RIVF v6 locked temporal-replication result",
            "",
            f"Status: **{report['status']}**.",
            "",
            "The v5 model weights, encoder, and validation-selected threshold were applied without retraining.",
            "This is a same-VM temporal replication, not external validation.",
            "",
            f"- Records / pairs: {len(records)} / {len(by_pair)}",
            f"- Candidate test TPR: {candidate['metrics']['tpr']:.3f}",
            f"- Candidate test FPR: {candidate['metrics']['fpr']:.3f}",
            f"- Candidate AUC: {candidate['auc']:.3f}",
            f"- Candidate pair ranking: {candidate['pair_ranking']['pair_ranking_accuracy']:.3f}",
            f"- Compromised-admin detection: {candidate['compromised_admin_audit']['detected']}/{candidate['compromised_admin_audit']['attack_records']}",
            "",
        ]
        (args.output_dir / "results.md").write_text("\n".join(text), encoding="utf-8")
        print(json.dumps({"status": "complete", "records": len(records), "pairs": len(by_pair), "output": str(args.output_dir)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
