#!/usr/bin/env python3
"""Independently recompute final v5 table arithmetic from dataset and predictions."""

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
SPLITS = ("development", "validation", "test")


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


def confusion(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        alerted = score >= threshold
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
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
    }


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
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


def pair_ranking(rows: list[dict[str, str]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["label"]] = float(row["score"])
    wins = ties = losses = 0
    for scores in grouped.values():
        if set(scores) != set(LABELS):
            continue
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


def exact_two_sided_binomial_pvalue(first: int, second: int) -> float:
    total = first + second
    if total == 0:
        return 1.0
    lower = min(first, second)
    return min(1.0, 2.0 * sum(math.comb(total, k) for k in range(lower + 1)) / (2**total))


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


def check_tree(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        check_value(errors, f"{name}.keys", set(actual), set(expected))
        for key in sorted(set(actual) & set(expected)):
            check_tree(errors, f"{name}.{key}", actual[key], expected[key])
        return
    check_value(errors, name, actual, expected)


def uncertainty_model_summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    positives = [row for row in selected if row["label"] == "attack_preparation"]
    negatives = [row for row in selected if row["label"] == "legitimate_admin"]
    paired: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        paired[row["pair_id"]][row["label"]] = row
    wins = ties = losses = 0
    for pair in paired.values():
        attack = pair["attack_preparation"]["score"]
        legitimate = pair["legitimate_admin"]["score"]
        if attack > legitimate:
            wins += 1
        elif attack == legitimate:
            ties += 1
        else:
            losses += 1
    by_operation: dict[str, Any] = {}
    for operation in sorted({row["operation"] for row in selected}):
        operation_positives = [row for row in positives if row["operation"] == operation]
        operation_negatives = [row for row in negatives if row["operation"] == operation]
        by_operation[operation] = {
            "tpr": wilson(sum(row["alerted"] for row in operation_positives), len(operation_positives)),
            "fpr": wilson(sum(row["alerted"] for row in operation_negatives), len(operation_negatives)),
        }
    by_account_state: dict[str, Any] = {}
    for state in sorted({row["account_state_label"] for row in positives}):
        state_rows = [row for row in positives if row["account_state_label"] == state]
        by_account_state[state] = wilson(sum(row["alerted"] for row in state_rows), len(state_rows))
    return {
        "tpr_wilson_95": wilson(sum(row["alerted"] for row in positives), len(positives)),
        "fpr_wilson_95": wilson(sum(row["alerted"] for row in negatives), len(negatives)),
        "by_operation": by_operation,
        "attack_detection_by_account_state": by_account_state,
        "pair_ranking": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "accuracy_with_half_credit_for_ties": (wins + 0.5 * ties) / len(paired),
            "exact_sign_test_pvalue_excluding_ties": exact_two_sided_binomial_pvalue(wins, losses),
        },
    }


def paired_discordance(rows: list[dict[str, Any]], baseline: str, candidate: str, label: str) -> dict[str, Any]:
    indexed: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row["label"] == label and row["model"] in {baseline, candidate}:
            indexed[row["record_id"]][row["model"]] = row["alerted"]
    candidate_only = baseline_only = agreements = 0
    for values in indexed.values():
        if values[baseline] == values[candidate]:
            agreements += 1
        elif values[candidate]:
            candidate_only += 1
        else:
            baseline_only += 1
    return {
        "candidate_alert_baseline_no_alert": candidate_only,
        "baseline_alert_candidate_no_alert": baseline_only,
        "agreements": agreements,
        "discordant": candidate_only + baseline_only,
        "exact_two_sided_pvalue": exact_two_sided_binomial_pvalue(candidate_only, baseline_only),
    }


def alert_rate(rows: list[dict[str, Any]], model: str, label: str) -> float:
    selected = [row for row in rows if row["model"] == model and row["label"] == label]
    return sum(row["alerted"] for row in selected) / len(selected)


def paired_cluster_bootstrap(
    rows: list[dict[str, Any]], baseline: str, candidate: str, repetitions: int, seed: int
) -> dict[str, Any]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["model"] in {baseline, candidate}:
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


def equal(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def check_value(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if not equal(actual, expected):
        errors.append(f"{name}: recomputed={actual!r}, reported={expected!r}")


def verify_dataset(records: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    record_ids = [str(record.get("record_id")) for record in records]
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pairs[str(record.get("pair_id"))].append(record)
    check_value(errors, "dataset.records", len(records), 144)
    check_value(errors, "dataset.unique_record_ids", len(set(record_ids)), 144)
    check_value(errors, "dataset.pairs", len(pairs), 72)
    for pair_id, pair in pairs.items():
        check_value(errors, f"dataset.pair[{pair_id}].size", len(pair), 2)
        check_value(errors, f"dataset.pair[{pair_id}].labels", {r.get("label") for r in pair}, set(LABELS))
        check_value(errors, f"dataset.pair[{pair_id}].splits", len({r.get("split") for r in pair}), 1)
    by_split_label = Counter((str(r.get("split")), str(r.get("label"))) for r in records)
    for split in SPLITS:
        for label in LABELS:
            check_value(errors, f"dataset.{split}.{label}", by_split_label[(split, label)], 24)
    check_value(errors, "dataset.claim_eligible", sum(bool(r.get("claim_eligible")) for r in records), 144)
    check_value(errors, "dataset.synthetic", sum(bool(r.get("synthetic")) for r in records), 0)
    return {
        "records": len(records),
        "unique_record_ids": len(set(record_ids)),
        "pairs": len(pairs),
        "by_split_and_label": {f"{split}|{label}": count for (split, label), count in sorted(by_split_label.items())},
        "by_operation_and_label": {
            f"{operation}|{label}": count
            for (operation, label), count in sorted(
                Counter((str(r.get("operation")), str(r.get("label"))) for r in records).items()
            )
        },
        "attack_by_account_state": dict(
            sorted(Counter(str(r.get("account_state_label")) for r in records if r.get("label") == "attack_preparation").items())
        ),
        "by_context_profile_and_label": {
            f"{profile}|{label}": count
            for (profile, label), count in sorted(
                Counter((str(r.get("context_profile")), str(r.get("label"))) for r in records).items()
            )
        },
    }


def verify_report(
    name: str,
    dataset_path: Path,
    records: list[dict[str, Any]],
    report_path: Path,
    predictions_path: Path,
    errors: list[str],
) -> dict[str, Any]:
    report = load_json(report_path)
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    record_ids = {str(record["record_id"]) for record in records}
    check_value(errors, f"{name}.input_sha256", report["input_identity"]["sha256"], sha256_file(dataset_path))
    check_value(errors, f"{name}.validation_valid", report["data_validation"]["valid"], True)
    model_names = set(report["models"])
    check_value(errors, f"{name}.prediction_models", {row["model"] for row in rows}, model_names)
    check_value(errors, f"{name}.prediction_row_count", len(rows), len(records) * len(model_names))
    check_value(errors, f"{name}.prediction_record_ids", {row["record_id"] for row in rows}, record_ids)
    check_value(errors, f"{name}.unique_model_record_rows", len({(row["model"], row["record_id"]) for row in rows}), len(rows))

    recomputed: dict[str, Any] = {}
    for model_name, model in report["models"].items():
        threshold = float(model["threshold_selected_on_validation"])
        model_rows = [row for row in rows if row["model"] == model_name]
        for row in model_rows:
            check_value(errors, f"{name}.{model_name}.{row['record_id']}.threshold", float(row["threshold"]), threshold)
            expected_alert = float(row["score"]) >= threshold
            recorded_alert = row["alerted"].strip().lower() == "true"
            check_value(errors, f"{name}.{model_name}.{row['record_id']}.alerted", recorded_alert, expected_alert)
        recomputed[model_name] = {}
        for split in SPLITS:
            split_rows = [row for row in model_rows if row["split"] == split]
            labels = [LABELS[row["label"]] for row in split_rows]
            scores = [float(row["score"]) for row in split_rows]
            metrics = confusion(labels, scores, threshold)
            ranking = pair_ranking(split_rows)
            for metric, actual in metrics.items():
                check_value(errors, f"{name}.{model_name}.{split}.metrics.{metric}", actual, model[split]["metrics"][metric])
            check_value(errors, f"{name}.{model_name}.{split}.auc", auc(labels, scores), model[split]["auc"])
            for metric, actual in ranking.items():
                check_value(errors, f"{name}.{model_name}.{split}.pair_ranking.{metric}", actual, model[split]["pair_ranking"][metric])
            recomputed[model_name][split] = {"metrics": metrics, "auc": auc(labels, scores), "pair_ranking": ranking}

        test_rows = [row for row in model_rows if row["split"] == "test" and row["label"] == "attack_preparation"]
        for state, report_key in (("compromised", "compromised_admin_audit"), ("not_applicable", "other_attack_audit")):
            selected = [row for row in test_rows if row["account_state_label"] == state]
            detected = sum(float(row["score"]) >= threshold for row in selected)
            expected = model["test"][report_key]
            check_value(errors, f"{name}.{model_name}.{report_key}.attack_records", len(selected), expected["attack_records"])
            check_value(errors, f"{name}.{model_name}.{report_key}.detected", detected, expected["detected"])
            rate = detected / len(selected) if selected else None
            check_value(errors, f"{name}.{model_name}.{report_key}.detection_rate", rate, expected["detection_rate"])

    primary = report["primary_comparison"]
    baseline = recomputed[primary["baseline_model"]]["test"]["metrics"]
    candidate = recomputed[primary["candidate_model"]]["test"]["metrics"]
    for metric in ("tpr", "fpr", "mcc"):
        check_value(
            errors,
            f"{name}.primary_comparison.test_{metric}_difference",
            candidate[metric] - baseline[metric],
            primary[f"test_{metric}_difference"],
        )
    return {
        "report_sha256": sha256_file(report_path),
        "predictions_sha256": sha256_file(predictions_path),
        "prediction_rows": len(rows),
        "models_verified": sorted(model_names),
        "splits_verified": list(SPLITS),
    }


def verify_uncertainty(
    name: str,
    report_path: Path,
    predictions_path: Path,
    uncertainty_path: Path,
    errors: list[str],
) -> dict[str, Any]:
    report = load_json(report_path)
    uncertainty = load_json(uncertainty_path)
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            rows.append(
                {
                    **row,
                    "score": float(row["score"]),
                    "threshold": float(row["threshold"]),
                    "alerted": row["alerted"].strip().lower() == "true",
                }
            )
    baseline = report["primary_comparison"]["baseline_model"]
    candidate = report["primary_comparison"]["candidate_model"]
    check_value(errors, f"{name}.uncertainty.baseline_model", baseline, uncertainty["baseline_model"])
    check_value(errors, f"{name}.uncertainty.candidate_model", candidate, uncertainty["candidate_model"])
    models = sorted({row["model"] for row in rows})
    check_value(errors, f"{name}.uncertainty.models", set(models), set(uncertainty["model_summaries"]))
    for model in models:
        check_tree(
            errors,
            f"{name}.uncertainty.model_summaries.{model}",
            uncertainty_model_summary(rows, model),
            uncertainty["model_summaries"][model],
        )
    recomputed_discordance = {
        "attack_records": paired_discordance(rows, baseline, candidate, "attack_preparation"),
        "legitimate_records": paired_discordance(rows, baseline, candidate, "legitimate_admin"),
    }
    check_tree(
        errors,
        f"{name}.uncertainty.primary_discordance",
        recomputed_discordance,
        uncertainty["primary_discordance"],
    )
    bootstrap_report = uncertainty["paired_cluster_bootstrap"]
    recomputed_bootstrap = paired_cluster_bootstrap(
        rows,
        baseline,
        candidate,
        int(bootstrap_report["repetitions"]),
        int(bootstrap_report["seed"]),
    )
    check_tree(
        errors,
        f"{name}.uncertainty.paired_cluster_bootstrap",
        recomputed_bootstrap,
        bootstrap_report,
    )
    return {
        "uncertainty_sha256": sha256_file(uncertainty_path),
        "test_prediction_rows": len(rows),
        "wilson_summaries_verified": len(models),
        "bootstrap_repetitions_verified": int(bootstrap_report["repetitions"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registered-dataset", type=Path, required=True)
    parser.add_argument("--measured-dataset", type=Path, required=True)
    parser.add_argument("--registered-report", type=Path, required=True)
    parser.add_argument("--registered-predictions", type=Path, required=True)
    parser.add_argument("--registered-uncertainty", type=Path, required=True)
    parser.add_argument("--measured-report", type=Path, required=True)
    parser.add_argument("--measured-predictions", type=Path, required=True)
    parser.add_argument("--measured-uncertainty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite verification output: {args.output}")
        registered_records = load_jsonl(args.registered_dataset)
        measured_records = load_jsonl(args.measured_dataset)
        errors: list[str] = []
        registered_dataset_summary = verify_dataset(registered_records, errors)
        measured_dataset_summary = verify_dataset(measured_records, errors)
        registered_projection = {
            (r.get("record_id"), r.get("pair_id"), r.get("split"), r.get("operation"), r.get("label"))
            for r in registered_records
        }
        measured_projection = {
            (r.get("record_id"), r.get("pair_id"), r.get("split"), r.get("operation"), r.get("label"))
            for r in measured_records
        }
        check_value(errors, "dataset.v1_v2_identity_projection", registered_projection, measured_projection)
        analyses = {
            "registered": verify_report(
                "registered",
                args.registered_dataset,
                registered_records,
                args.registered_report,
                args.registered_predictions,
                errors,
            ),
            "measured": verify_report(
                "measured",
                args.measured_dataset,
                measured_records,
                args.measured_report,
                args.measured_predictions,
                errors,
            ),
        }
        analyses["registered"].update(
            verify_uncertainty(
                "registered",
                args.registered_report,
                args.registered_predictions,
                args.registered_uncertainty,
                errors,
            )
        )
        analyses["measured"].update(
            verify_uncertainty(
                "measured",
                args.measured_report,
                args.measured_predictions,
                args.measured_uncertainty,
                errors,
            )
        )
        output = {
            "schema_version": "rivf-v5-independent-result-verification-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass" if not errors else "fail",
            "method": "standard-library recomputation from claim-eligible JSONL and exported per-record scores",
            "datasets": {
                "registered_v1": {
                    "path": str(args.registered_dataset),
                    "sha256": sha256_file(args.registered_dataset),
                    **registered_dataset_summary,
                },
                "measured_v2": {
                    "path": str(args.measured_dataset),
                    "sha256": sha256_file(args.measured_dataset),
                    **measured_dataset_summary,
                },
            },
            "analyses": analyses,
            "checks": {
                "confusion_matrices": "all models and all splits",
                "derived_metrics": ["TPR", "FPR", "precision", "accuracy", "MCC", "AUC"],
                "paired_metrics": "pair wins/ties/losses and pair-ranking accuracy",
                "subgroups": "test compromised-admin and other attack records",
                "primary_deltas": ["TPR", "FPR", "MCC"],
                "uncertainty": [
                    "Wilson 95% intervals",
                    "exact paired discordance p-values",
                    "exact pair sign-test p-values",
                    "deterministic matched-pair bootstrap percentile intervals",
                ],
            },
            "errors": errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": output["status"], "errors": len(errors), "output": str(args.output)}, sort_keys=True))
        return 0 if not errors else 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError, csv.Error) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
