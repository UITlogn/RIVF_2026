#!/usr/bin/env python3
"""Compute paired uncertainty summaries from a completed context-study run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if trials == 0:
        return {"estimate": None, "lower": None, "upper": None, "successes": 0, "trials": 0}
    estimate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (estimate + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(estimate * (1.0 - estimate) / trials + z * z / (4.0 * trials * trials)) / denominator
    lower_bound = 0.0 if successes == 0 else max(0.0, center - half)
    upper_bound = 1.0 if successes == trials else min(1.0, center + half)
    return {
        "estimate": estimate,
        "lower": lower_bound,
        "upper": upper_bound,
        "successes": successes,
        "trials": trials,
    }


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    lower = min(b, c)
    one_tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * one_tail)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def as_bool(value: str) -> bool:
    lowered = value.strip().casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            rows.append({**row, "score": float(row["score"]), "threshold": float(row["threshold"]), "alerted": as_bool(row["alerted"])})
    return rows


def rate(rows: list[dict[str, Any]], model: str, label: str) -> float:
    selected = [row for row in rows if row["model"] == model and row["label"] == label]
    if not selected:
        raise ValueError(f"no {label} test rows for {model}")
    return sum(row["alerted"] for row in selected) / len(selected)


def paired_discordance(rows: list[dict[str, Any]], baseline: str, candidate: str, label: str) -> dict[str, Any]:
    indexed: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row["label"] == label and row["model"] in {baseline, candidate}:
            indexed[row["record_id"]][row["model"]] = row["alerted"]
    b = c = agreements = 0
    for record_id, values in indexed.items():
        if set(values) != {baseline, candidate}:
            raise ValueError(f"incomplete paired model predictions for {record_id}")
        if values[baseline] == values[candidate]:
            agreements += 1
        elif values[candidate]:
            b += 1
        else:
            c += 1
    return {
        "candidate_alert_baseline_no_alert": b,
        "baseline_alert_candidate_no_alert": c,
        "agreements": agreements,
        "discordant": b + c,
        "exact_two_sided_pvalue": exact_mcnemar_pvalue(b, c),
    }


def model_summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    positives = [row for row in rows if row["model"] == model and row["label"] == "attack_preparation"]
    negatives = [row for row in rows if row["model"] == model and row["label"] == "legitimate_admin"]
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["model"] == model:
            pairs[row["pair_id"]][row["label"]] = row
    wins = ties = losses = 0
    for pair_id, pair in pairs.items():
        if set(pair) != {"legitimate_admin", "attack_preparation"}:
            raise ValueError(f"incomplete test pair {pair_id} for {model}")
        attack = pair["attack_preparation"]["score"]
        legitimate = pair["legitimate_admin"]["score"]
        if attack > legitimate:
            wins += 1
        elif attack == legitimate:
            ties += 1
        else:
            losses += 1
    by_operation: dict[str, Any] = {}
    for operation in sorted({row["operation"] for row in positives + negatives}):
        operation_positive = [row for row in positives if row["operation"] == operation]
        operation_negative = [row for row in negatives if row["operation"] == operation]
        by_operation[operation] = {
            "tpr": wilson(sum(row["alerted"] for row in operation_positive), len(operation_positive)),
            "fpr": wilson(sum(row["alerted"] for row in operation_negative), len(operation_negative)),
        }
    by_attack_account_state: dict[str, Any] = {}
    for state in sorted({row["account_state_label"] for row in positives}):
        selected = [row for row in positives if row["account_state_label"] == state]
        by_attack_account_state[state] = wilson(
            sum(row["alerted"] for row in selected), len(selected)
        )
    return {
        "tpr_wilson_95": wilson(sum(row["alerted"] for row in positives), len(positives)),
        "fpr_wilson_95": wilson(sum(row["alerted"] for row in negatives), len(negatives)),
        "by_operation": by_operation,
        "attack_detection_by_account_state": by_attack_account_state,
        "pair_ranking": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "accuracy_with_half_credit_for_ties": (wins + 0.5 * ties) / len(pairs),
            "exact_sign_test_pvalue_excluding_ties": exact_mcnemar_pvalue(wins, losses),
        },
    }


def cluster_bootstrap(rows: list[dict[str, Any]], baseline: str, candidate: str, repetitions: int, seed: int) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["model"] in {baseline, candidate}:
            pairs[row["pair_id"]].append(row)
    pair_ids = sorted(pairs)
    if not pair_ids:
        raise ValueError("no test pairs for bootstrap")
    rng = random.Random(seed)
    delta_tpr: list[float] = []
    delta_fpr: list[float] = []
    for _ in range(repetitions):
        sampled_rows: list[dict[str, Any]] = []
        for pair_id in rng.choices(pair_ids, k=len(pair_ids)):
            sampled_rows.extend(pairs[pair_id])
        delta_tpr.append(rate(sampled_rows, candidate, "attack_preparation") - rate(sampled_rows, baseline, "attack_preparation"))
        delta_fpr.append(rate(sampled_rows, candidate, "legitimate_admin") - rate(sampled_rows, baseline, "legitimate_admin"))
    return {
        "unit": "matched_test_pair",
        "repetitions": repetitions,
        "seed": seed,
        "delta_tpr_percentile_95": {"lower": percentile(delta_tpr, 0.025), "median": percentile(delta_tpr, 0.5), "upper": percentile(delta_tpr, 0.975)},
        "delta_fpr_percentile_95": {"lower": percentile(delta_fpr, 0.025), "median": percentile(delta_fpr, 0.5), "upper": percentile(delta_fpr, 0.975)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        report = json.loads(args.model_report.read_text(encoding="utf-8-sig"))
        primary = report["primary_comparison"]
        baseline = primary["baseline_model"]
        candidate = primary["candidate_model"]
        rows = read_predictions(args.predictions)
        models = sorted({row["model"] for row in rows})
        output = {
            "schema_version": "rivf-v5-paired-uncertainty-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_model": baseline,
            "candidate_model": candidate,
            "model_summaries": {model: model_summary(rows, model) for model in models},
            "primary_discordance": {
                "attack_records": paired_discordance(rows, baseline, candidate, "attack_preparation"),
                "legitimate_records": paired_discordance(rows, baseline, candidate, "legitimate_admin"),
            },
            "paired_cluster_bootstrap": cluster_bootstrap(rows, baseline, candidate, args.bootstrap_repetitions, args.seed),
            "interpretation": "Intervals quantify finite-test-set uncertainty for frozen test predictions; they do not address external-population or scenario-design uncertainty.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({"status": "complete", "models": models, "output": str(args.output)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
