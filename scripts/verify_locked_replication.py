#!/usr/bin/env python3
"""Independently verify a locked, test-only temporal replication result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = {"legitimate_admin": 0, "attack_preparation": 1}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        label = LABELS[row["label"]]
        alerted = bool(row["alerted"])
        if label == 1 and alerted:
            tp += 1
        elif label == 1:
            fn += 1
        elif alerted:
            fp += 1
        else:
            tn += 1
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
    }


def auc(rows: list[dict[str, Any]]) -> float | None:
    positives = [row["score"] for row in rows if row["label"] == "attack_preparation"]
    negatives = [row["score"] for row in rows if row["label"] == "legitimate_admin"]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def pair_ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["label"]] = row["score"]
    wins = ties = losses = 0
    for pair_id, scores in pairs.items():
        if set(scores) != set(LABELS):
            raise ValueError(f"incomplete prediction pair: {pair_id}")
        attack = scores["attack_preparation"]
        legitimate = scores["legitimate_admin"]
        if attack > legitimate:
            wins += 1
        elif attack == legitimate:
            ties += 1
        else:
            losses += 1
    total = wins + ties + losses
    return {
        "pairs": total,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "pair_ranking_accuracy": (wins + 0.5 * ties) / total if total else None,
    }


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if trials == 0:
        return {"estimate": None, "lower": None, "upper": None, "successes": 0, "trials": 0}
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return {
        "estimate": estimate,
        "lower": 0.0 if successes == 0 else max(0.0, center - half),
        "upper": 1.0 if successes == trials else min(1.0, center + half),
        "successes": successes,
        "trials": trials,
    }


def exact_two_sided_binomial(first: int, second: int) -> float:
    total = first + second
    if total == 0:
        return 1.0
    lower = min(first, second)
    return min(1.0, 2.0 * sum(math.comb(total, k) for k in range(lower + 1)) / (2**total))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def check(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if not equal(actual, expected):
        errors.append(f"{name}: recomputed={actual!r}, reported={expected!r}")


def check_tree(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        check(errors, f"{name}.keys", set(actual), set(expected))
        for key in sorted(set(actual) & set(expected)):
            check_tree(errors, f"{name}.{key}", actual[key], expected[key])
        return
    check(errors, name, actual, expected)


def uncertainty_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["label"] == "attack_preparation"]
    negatives = [row for row in rows if row["label"] == "legitimate_admin"]
    ranking = pair_ranking(rows)
    by_operation: dict[str, Any] = {}
    for operation in sorted({row["operation"] for row in rows}):
        positive_rows = [row for row in positives if row["operation"] == operation]
        negative_rows = [row for row in negatives if row["operation"] == operation]
        by_operation[operation] = {
            "tpr": wilson(sum(row["alerted"] for row in positive_rows), len(positive_rows)),
            "fpr": wilson(sum(row["alerted"] for row in negative_rows), len(negative_rows)),
        }
    by_state: dict[str, Any] = {}
    for state in sorted({row["account_state_label"] for row in positives}):
        state_rows = [row for row in positives if row["account_state_label"] == state]
        by_state[state] = wilson(sum(row["alerted"] for row in state_rows), len(state_rows))
    return {
        "tpr_wilson_95": wilson(sum(row["alerted"] for row in positives), len(positives)),
        "fpr_wilson_95": wilson(sum(row["alerted"] for row in negatives), len(negatives)),
        "by_operation": by_operation,
        "attack_detection_by_account_state": by_state,
        "pair_ranking": {
            "wins": ranking["wins"],
            "ties": ranking["ties"],
            "losses": ranking["losses"],
            "accuracy_with_half_credit_for_ties": ranking["pair_ranking_accuracy"],
            "exact_sign_test_pvalue_excluding_ties": exact_two_sided_binomial(
                ranking["wins"], ranking["losses"]
            ),
        },
    }


def discordance(
    rows: list[dict[str, Any]], baseline: str, candidate: str, label: str
) -> dict[str, Any]:
    indexed: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row["label"] == label and row["model"] in {baseline, candidate}:
            indexed[row["record_id"]][row["model"]] = row["alerted"]
    candidate_only = baseline_only = agreements = 0
    for record_id, alerts in indexed.items():
        if set(alerts) != {baseline, candidate}:
            raise ValueError(f"incomplete discordance row: {record_id}")
        if alerts[baseline] == alerts[candidate]:
            agreements += 1
        elif alerts[candidate]:
            candidate_only += 1
        else:
            baseline_only += 1
    return {
        "candidate_alert_baseline_no_alert": candidate_only,
        "baseline_alert_candidate_no_alert": baseline_only,
        "agreements": agreements,
        "discordant": candidate_only + baseline_only,
        "exact_two_sided_pvalue": exact_two_sided_binomial(candidate_only, baseline_only),
    }


def alert_rate(rows: list[dict[str, Any]], model: str, label: str) -> float:
    selected = [row for row in rows if row["model"] == model and row["label"] == label]
    return sum(row["alerted"] for row in selected) / len(selected)


def bootstrap(
    rows: list[dict[str, Any]], baseline: str, candidate: str, repetitions: int, seed: int
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    pair_ids = sorted(pairs)
    rng = random.Random(seed)
    delta_tpr: list[float] = []
    delta_fpr: list[float] = []
    for _ in range(repetitions):
        sample: list[dict[str, Any]] = []
        for pair_id in rng.choices(pair_ids, k=len(pair_ids)):
            sample.extend(pairs[pair_id])
        delta_tpr.append(
            alert_rate(sample, candidate, "attack_preparation")
            - alert_rate(sample, baseline, "attack_preparation")
        )
        delta_fpr.append(
            alert_rate(sample, candidate, "legitimate_admin")
            - alert_rate(sample, baseline, "legitimate_admin")
        )
    return {
        "unit": "matched_test_pair",
        "repetitions": repetitions,
        "seed": seed,
        "delta_tpr_percentile_95": {
            "lower": percentile(delta_tpr, 0.025),
            "median": percentile(delta_tpr, 0.5),
            "upper": percentile(delta_tpr, 0.975),
        },
        "delta_fpr_percentile_95": {
            "lower": percentile(delta_fpr, 0.025),
            "median": percentile(delta_fpr, 0.5),
            "upper": percentile(delta_fpr, 0.975),
        },
    }


def verify_raw_artifacts(
    artifact_root: Path, expected_ids: set[str], errors: list[str]
) -> dict[str, Any]:
    checksum_entries = 0
    valid_procmon = 0
    invalid_json = 0
    for run_id in sorted(expected_ids):
        run_dir = artifact_root / run_id
        summary_path = run_dir / "run-summary.json"
        checksum_path = run_dir / "SHA256SUMS"
        if not summary_path.is_file() or not checksum_path.is_file():
            errors.append(f"raw.{run_id}: missing run-summary.json or SHA256SUMS")
            continue
        summary = load_json(summary_path)
        check(errors, f"raw.{run_id}.summary_run_id", summary.get("run_id"), run_id)
        check(errors, f"raw.{run_id}.status", summary.get("status"), "pass")
        check(errors, f"raw.{run_id}.containment", summary.get("final_containment", {}).get("clean"), True)
        valid_procmon += int(summary.get("drakvuf", {}).get("valid_json_records", 0))
        invalid_json += int(summary.get("drakvuf", {}).get("invalid_json_records", 0))
        for raw_line in checksum_path.read_text(encoding="utf-8-sig").splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.split(maxsplit=1)
            if len(parts) != 2:
                errors.append(f"raw.{run_id}: malformed SHA256SUMS line")
                continue
            expected_hash, relative = parts
            relative = relative.lstrip(" *").replace("/", str(Path("/")).replace("/", "\\"))
            target = run_dir / relative
            if not target.is_file():
                errors.append(f"raw.{run_id}: missing checksum target {relative}")
                continue
            checksum_entries += 1
            check(errors, f"raw.{run_id}.sha256.{relative}", sha256_file(target), expected_hash.lower())
    check(errors, "raw.invalid_json_records", invalid_json, 0)
    return {
        "runs": len(expected_ids),
        "checksum_entries_verified": checksum_entries,
        "valid_procmon_records": valid_procmon,
        "invalid_json_records": invalid_json,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--locked-model-artifacts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--finalization", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite verification output: {args.output}")
        errors: list[str] = []
        manifest = load_json(args.manifest)
        freeze = load_json(args.freeze)
        report = load_json(args.report)
        uncertainty = load_json(args.uncertainty)
        finalization = load_json(args.finalization)
        records = load_jsonl(args.dataset)

        expected_records = int(manifest["design"]["records"])
        expected_pairs = int(manifest["design"]["pairs"])
        check(errors, "dataset.records", len(records), expected_records)
        check(errors, "dataset.unique_record_ids", len({r["record_id"] for r in records}), expected_records)
        check(errors, "dataset.test_only", {r["split"] for r in records}, {"test"})
        check(errors, "dataset.claim_eligible", sum(bool(r.get("claim_eligible")) for r in records), expected_records)
        check(errors, "dataset.synthetic", sum(bool(r.get("synthetic")) for r in records), 0)

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
        check(errors, "dataset.manifest_record_ids", {r["record_id"] for r in records}, set(expected))
        pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            frozen = expected.get(str(record["record_id"]), {})
            for field, value in frozen.items():
                check(errors, f"dataset.{record['record_id']}.{field}", str(record.get(field)), value)
            pairs[str(record["pair_id"])].append(record)
        check(errors, "dataset.pairs", len(pairs), expected_pairs)
        for pair_id, pair in pairs.items():
            check(errors, f"dataset.{pair_id}.size", len(pair), 2)
            check(errors, f"dataset.{pair_id}.labels", {r["label"] for r in pair}, set(LABELS))

        check(errors, "freeze.execution_started_before_freeze", freeze.get("v6_execution_started_before_freeze"), False)
        check(errors, "freeze.no_retraining", freeze.get("no_retraining"), True)
        freeze_mismatches: list[str] = []
        for relative, expected_hash in freeze["files"].items():
            target = args.project_root / Path(relative)
            if not target.is_file() or sha256_file(target) != expected_hash:
                freeze_mismatches.append(relative)
        check(errors, "freeze.hash_mismatches", freeze_mismatches, [])

        manifest_hash = sha256_file(args.manifest)
        model_hash = sha256_file(args.locked_model_artifacts)
        dataset_hash = sha256_file(args.dataset)
        report_hash = sha256_file(args.report)
        predictions_hash = sha256_file(args.predictions)
        uncertainty_hash = sha256_file(args.uncertainty)
        check(errors, "manifest.locked_model_hash", manifest["design"]["locked_model_artifact_sha256"], model_hash)
        check(errors, "report.manifest_hash", report.get("manifest_sha256"), manifest_hash)
        check(errors, "report.locked_model_hash", report.get("locked_model_artifacts_sha256"), model_hash)
        check(errors, "report.input_hash", report.get("input_sha256"), dataset_hash)
        check(errors, "report.no_retraining", report.get("no_retraining"), True)
        check(errors, "finalization.no_retraining", finalization.get("no_retraining"), True)
        check(errors, "finalization.source_hash", finalization.get("claim_dataset_sha256"), dataset_hash)
        check(errors, "finalization.report_hash", finalization.get("model_report_sha256"), report_hash)
        check(errors, "finalization.predictions_hash", finalization.get("predictions_sha256"), predictions_hash)
        check(errors, "finalization.uncertainty_hash", finalization.get("uncertainty_sha256"), uncertainty_hash)

        with args.predictions.open("r", encoding="utf-8-sig", newline="") as handle:
            prediction_rows = []
            for raw in csv.DictReader(handle):
                prediction_rows.append(
                    {
                        **raw,
                        "score": float(raw["score"]),
                        "threshold": float(raw["threshold"]),
                        "alerted": raw["alerted"].strip().lower() == "true",
                    }
                )
        model_names = set(report["models"])
        check(errors, "predictions.models", {r["model"] for r in prediction_rows}, model_names)
        check(errors, "predictions.rows", len(prediction_rows), expected_records * len(model_names))
        check(errors, "predictions.unique_rows", len({(r["model"], r["record_id"]) for r in prediction_rows}), len(prediction_rows))
        check(errors, "predictions.record_ids", {r["record_id"] for r in prediction_rows}, set(expected))

        recomputed_metrics: dict[str, dict[str, Any]] = {}
        for model_name in sorted(model_names):
            model_rows = [row for row in prediction_rows if row["model"] == model_name]
            locked_threshold = float(report["models"][model_name]["locked_threshold"])
            for row in model_rows:
                check(errors, f"predictions.{model_name}.{row['record_id']}.threshold", row["threshold"], locked_threshold)
                check(errors, f"predictions.{model_name}.{row['record_id']}.alert", row["alerted"], row["score"] >= locked_threshold)
            metrics = confusion(model_rows)
            ranking = pair_ranking(model_rows)
            recomputed_metrics[model_name] = metrics
            check_tree(errors, f"report.{model_name}.metrics", metrics, report["models"][model_name]["test"]["metrics"])
            check(errors, f"report.{model_name}.auc", auc(model_rows), report["models"][model_name]["test"]["auc"])
            check_tree(errors, f"report.{model_name}.pair_ranking", ranking, report["models"][model_name]["test"]["pair_ranking"])
            attack_rows = [row for row in model_rows if row["label"] == "attack_preparation"]
            for state, key in (("compromised", "compromised_admin_audit"), ("not_applicable", "other_attack_audit")):
                selected = [row for row in attack_rows if row["account_state_label"] == state]
                detected = sum(row["alerted"] for row in selected)
                actual = {
                    "account_state_label": state,
                    "attack_records": len(selected),
                    "detected": detected,
                    "detection_rate": detected / len(selected) if selected else None,
                }
                check_tree(errors, f"report.{model_name}.{key}", actual, report["models"][model_name]["test"][key])

        primary = report["primary_comparison"]
        baseline = primary["baseline_model"]
        candidate = primary["candidate_model"]
        for short, metric in (("tpr", "tpr"), ("fpr", "fpr"), ("mcc", "mcc")):
            delta = recomputed_metrics[candidate][metric] - recomputed_metrics[baseline][metric]
            check(errors, f"report.primary.delta_{short}", delta, primary[f"delta_{short}"])

        for model_name in sorted(model_names):
            model_rows = [row for row in prediction_rows if row["model"] == model_name]
            check_tree(
                errors,
                f"uncertainty.{model_name}",
                uncertainty_summary(model_rows),
                uncertainty["model_summaries"][model_name],
            )
        check_tree(
            errors,
            "uncertainty.discordance.attack",
            discordance(prediction_rows, baseline, candidate, "attack_preparation"),
            uncertainty["primary_discordance"]["attack_records"],
        )
        check_tree(
            errors,
            "uncertainty.discordance.legitimate",
            discordance(prediction_rows, baseline, candidate, "legitimate_admin"),
            uncertainty["primary_discordance"]["legitimate_records"],
        )
        bootstrap_report = uncertainty["paired_cluster_bootstrap"]
        check_tree(
            errors,
            "uncertainty.bootstrap",
            bootstrap(
                prediction_rows,
                baseline,
                candidate,
                int(bootstrap_report["repetitions"]),
                int(bootstrap_report["seed"]),
            ),
            bootstrap_report,
        )

        raw = verify_raw_artifacts(args.artifact_root, set(expected), errors)
        output = {
            "schema_version": "rivf-locked-replication-independent-verification-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not errors else "fail",
            "method": "independent Python standard-library recomputation from frozen inputs, raw checksums, and exported scores",
            "identities": {
                "manifest_sha256": manifest_hash,
                "locked_model_artifacts_sha256": model_hash,
                "claim_dataset_sha256": dataset_hash,
                "model_report_sha256": report_hash,
                "predictions_sha256": predictions_hash,
                "uncertainty_sha256": uncertainty_hash,
            },
            "dataset": {
                "records": len(records),
                "pairs": len(pairs),
                "by_label": dict(sorted(Counter(str(r["label"]) for r in records).items())),
                "by_operation": dict(sorted(Counter(str(r["operation"]) for r in records).items())),
            },
            "raw_artifacts": raw,
            "models_verified": sorted(model_names),
            "checks": {
                "freeze_hashes": len(freeze["files"]),
                "no_retraining": True,
                "confusion_auc_pair_ranking_subgroups": True,
                "wilson_discordance_sign_test_bootstrap": True,
            },
            "errors": errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": output["status"], "errors": len(errors), "output": str(args.output)}, sort_keys=True))
        return 0 if not errors else 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, csv.Error) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
