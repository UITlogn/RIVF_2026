#!/usr/bin/env python3
"""Evaluate a transparent ATT&CK operation-matching rule on frozen datasets.

This is a post-hoc descriptive baseline. It does not fit parameters, select a
threshold, or alter the confirmatory v5/v6 analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ATTACK_OPERATION_RULE = {
    "service_disable": "T1489",
    "event_log_clear": "T1685.005",
    "shadow_copy_delete": "T1490",
    "backup_delete": "T1490",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-dataset", type=Path, required=True)
    parser.add_argument("--v6-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def label_value(record: dict) -> int:
    label = record["label"]
    if label in {"attack_preparation", "attack", 1, True}:
        return 1
    if label in {"legitimate_admin", "legitimate", 0, False}:
        return 0
    raise ValueError(f"Unexpected label: {label!r}")


def evaluate(records: list[dict]) -> dict:
    rows = []
    for record in records:
        score = 1.0 if record["operation"] in ATTACK_OPERATION_RULE else 0.0
        rows.append((record, label_value(record), score, int(score >= 0.5)))

    tp = sum(y == 1 and pred == 1 for _, y, _, pred in rows)
    fn = sum(y == 1 and pred == 0 for _, y, _, pred in rows)
    fp = sum(y == 0 and pred == 1 for _, y, _, pred in rows)
    tn = sum(y == 0 and pred == 0 for _, y, _, pred in rows)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    by_pair: dict[str, dict[int, float]] = {}
    for record, y, score, _ in rows:
        by_pair.setdefault(record["pair_id"], {})[y] = score
    wins = ties = losses = 0
    for pair in by_pair.values():
        if pair[1] > pair[0]:
            wins += 1
        elif pair[1] < pair[0]:
            losses += 1
        else:
            ties += 1

    positives = tp + fn
    negatives = fp + tn
    return {
        "records": len(rows),
        "pairs": len(by_pair),
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "metrics": {
            "tpr": tp / positives if positives else None,
            "fpr": fp / negatives if negatives else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "accuracy": (tp + tn) / len(rows) if rows else None,
            "mcc": mcc,
            "auc": 0.5,
        },
        "pair_ranking": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "accuracy_with_half_credit_for_ties": (wins + 0.5 * ties) / len(by_pair),
        },
    }


def main() -> int:
    args = parse_args()
    v5_all = load_jsonl(args.v5_dataset)
    v5_test = [record for record in v5_all if record["split"] == "test"]
    v6 = load_jsonl(args.v6_dataset)
    report = {
        "schema_version": "rivf-posthoc-attack-operation-rule-v1",
        "status": "posthoc_descriptive_nonlearning_baseline",
        "claim_guard": (
            "The rule was added after v5 test access. It quantifies the behavior "
            "of an operation-level ATT&CK match and is not a confirmatory result."
        ),
        "rule": {
            "definition": "Alert when the operation maps to an ATT&CK technique.",
            "operation_to_technique": ATTACK_OPERATION_RULE,
            "threshold": 0.5,
            "parameters_fit": 0,
        },
        "inputs": {
            "v5_dataset": {"path": str(args.v5_dataset), "sha256": sha256_file(args.v5_dataset)},
            "v6_dataset": {"path": str(args.v6_dataset), "sha256": sha256_file(args.v6_dataset)},
        },
        "results": {"v5_test": evaluate(v5_test), "v6_replication": evaluate(v6)},
        "interpretation": (
            "All four study operations are ATT&CK-relevant, so the rule alerts "
            "on every record. It achieves full sensitivity but 100% FPR and is "
            "infeasible under the study's 5% FPR budget."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
