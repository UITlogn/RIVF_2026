#!/usr/bin/env python3
"""Validate, train, calibrate, and evaluate the RIVF context study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StudyError(RuntimeError):
    pass


LABEL_TO_INT = {"legitimate_admin": 0, "attack_preparation": 1}
REQUIRED_CONTEXT = {
    "source_signature_status": str,
    "launcher_type": str,
    "parent_chain_trust": str,
    "session_type": str,
    "account_role": str,
    "target_scope": str,
    "host_scope_count": int,
    "prior_auth_anomaly_count_10m": int,
    "prior_discovery_count_5m": int,
    "prior_remote_execution_count_5m": int,
    "prior_security_impairment_count_5m": int,
    "operation_rate_60s": (int, float),
    "seconds_since_session_start": (int, float),
    "out_of_hours": bool,
}
ALLOWED_OPERATIONS = {
    "shadow_copy_delete",
    "event_log_clear",
    "service_disable",
    "backup_delete",
    "security_tool_stop",
    "remote_admin_execution",
}
ALLOWED_ACCOUNT_STATES = {"expected_use", "compromised", "not_applicable", "unknown"}
ALLOWED_CONTEXT_VALUES = {
    "source_signature_status": {"trusted", "untrusted", "unknown"},
    "launcher_type": {
        "interactive_shell",
        "remote_management",
        "service",
        "scheduler",
        "delayed_child",
        "unknown",
    },
    "parent_chain_trust": {"high", "mixed", "low", "unknown"},
    "session_type": {"console", "rdp", "network", "service", "batch", "unknown"},
    "account_role": {"administrator", "system", "service", "standard", "unknown"},
    "target_scope": {"local_host", "remote_single", "multi_host", "unknown"},
}
PROVENANCE_FIELDS = {"capture_profile_id", "source_artifact", "source_artifact_sha256"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"expected an object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StudyError(f"cannot read JSONL {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudyError(f"invalid JSONL line {number}: {exc}") from exc
        if not isinstance(value, dict):
            raise StudyError(f"JSONL line {number} is not an object")
        records.append(value)
    if not records:
        raise StudyError("input JSONL contains no records")
    return records


def get_field(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise StudyError(f"record {record.get('record_id')} lacks feature {dotted}")
        value = value[part]
    return value


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "context-experiment-config-v1":
        raise StudyError("unsupported experiment config version")
    feature_sets = config.get("feature_sets")
    if not isinstance(feature_sets, dict) or not feature_sets:
        raise StudyError("feature_sets must be a non-empty object")
    forbidden = set(config.get("forbidden_feature_fields", []))
    for model_name, specification in feature_sets.items():
        if not isinstance(specification, dict):
            raise StudyError(f"feature set {model_name} must be an object")
        selected = list(specification.get("categorical", [])) + list(
            specification.get("numeric", [])
        )
        leaked = forbidden.intersection(selected)
        if leaked:
            raise StudyError(
                f"feature set {model_name} contains forbidden fields: {sorted(leaked)}"
            )
    primary = config.get("primary_comparison", {})
    for key in ("baseline_model", "candidate_model"):
        if primary.get(key) not in feature_sets:
            raise StudyError(f"primary_comparison.{key} is not a defined feature set")
    maximum_fpr = primary.get("maximum_validation_false_positive_rate")
    if not isinstance(maximum_fpr, (int, float)) or not 0 <= maximum_fpr < 1:
        raise StudyError("maximum validation FPR must lie in [0, 1)")


def validate_records(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    required_splits = set(config.get("required_splits", []))
    required_labels = set(config.get("required_labels", []))
    record_ids: set[str] = set()
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[str] = []

    for index, record in enumerate(records, 1):
        prefix = f"record #{index}"
        required = {
            "schema_version",
            "record_id",
            "pair_id",
            "split",
            "workload_group",
            "operation",
            "label",
            "account_state_label",
            "synthetic",
            "telemetry_equivalent",
            "quality_eligible",
            "claim_eligible",
            "exclusion_reason",
            "context",
            "provenance",
        }
        missing = sorted(required.difference(record))
        if missing:
            errors.append(f"{prefix} missing fields {missing}")
            continue
        extra = sorted(set(record).difference(required))
        if extra:
            errors.append(f"{prefix} has unexpected fields {extra}")
        if record["schema_version"] != "context-record-v1":
            errors.append(f"{prefix} has unsupported schema_version")
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{prefix} has invalid record_id")
        elif record_id in record_ids:
            errors.append(f"duplicate record_id {record_id}")
        else:
            record_ids.add(record_id)
        if record["split"] not in required_splits:
            errors.append(f"{record_id} has invalid split {record['split']!r}")
        if record["label"] not in required_labels:
            errors.append(f"{record_id} has invalid label {record['label']!r}")
        if record["operation"] not in ALLOWED_OPERATIONS:
            errors.append(f"{record_id} has invalid operation {record['operation']!r}")
        if record["account_state_label"] not in ALLOWED_ACCOUNT_STATES:
            errors.append(
                f"{record_id} has invalid account_state_label {record['account_state_label']!r}"
            )
        if (
            record["account_state_label"] == "compromised"
            and record["label"] != "attack_preparation"
        ):
            errors.append(f"{record_id} compromised account state conflicts with label")
        if not isinstance(record["synthetic"], bool):
            errors.append(f"{record_id} synthetic must be boolean")
        if not isinstance(record["telemetry_equivalent"], bool):
            errors.append(f"{record_id} telemetry_equivalent must be boolean")
        if not isinstance(record["quality_eligible"], bool):
            errors.append(f"{record_id} quality_eligible must be boolean")
        if not isinstance(record["claim_eligible"], bool):
            errors.append(f"{record_id} claim_eligible must be boolean")
        if record["synthetic"] and record["claim_eligible"]:
            errors.append(f"{record_id} synthetic record cannot be claim-eligible")
        if record["claim_eligible"] and not (
            record["telemetry_equivalent"] and record["quality_eligible"]
        ):
            errors.append(
                f"{record_id} claim-eligible record lacks telemetry/quality eligibility"
            )
        if record["claim_eligible"] and record["exclusion_reason"] is not None:
            errors.append(f"{record_id} claim-eligible record has exclusion_reason")
        if not record["claim_eligible"] and not record["exclusion_reason"]:
            errors.append(f"{record_id} excluded record lacks exclusion_reason")
        context = record["context"]
        if not isinstance(context, dict):
            errors.append(f"{record_id} context must be an object")
        else:
            extra_context = sorted(set(context).difference(REQUIRED_CONTEXT))
            if extra_context:
                errors.append(f"{record_id} context has unexpected fields {extra_context}")
            for field, expected_type in REQUIRED_CONTEXT.items():
                if field not in context:
                    errors.append(f"{record_id} context lacks {field}")
                elif field != "out_of_hours" and isinstance(context[field], bool):
                    errors.append(f"{record_id} context.{field} has the wrong type")
                elif not isinstance(context[field], expected_type):
                    errors.append(f"{record_id} context.{field} has the wrong type")
                elif isinstance(context[field], (int, float)) and not isinstance(
                    context[field], bool
                ):
                    if not math.isfinite(float(context[field])):
                        errors.append(f"{record_id} context.{field} must be finite")
                    elif context[field] < 0:
                        errors.append(f"{record_id} context.{field} must be non-negative")
                if field in ALLOWED_CONTEXT_VALUES and context.get(field) not in ALLOWED_CONTEXT_VALUES[field]:
                    errors.append(f"{record_id} context.{field} has an unsupported value")
        provenance = record["provenance"]
        if not isinstance(provenance, dict) or not provenance.get("capture_profile_id"):
            errors.append(f"{record_id} lacks provenance.capture_profile_id")
        elif set(provenance) != PROVENANCE_FIELDS:
            errors.append(f"{record_id} provenance fields do not match the contract")
        else:
            source_path = provenance["source_artifact"]
            source_hash = provenance["source_artifact_sha256"]
            if source_path is not None and not isinstance(source_path, str):
                errors.append(f"{record_id} provenance.source_artifact is invalid")
            if source_hash is not None and (
                not isinstance(source_hash, str)
                or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)
            ):
                errors.append(f"{record_id} provenance.source_artifact_sha256 is invalid")
            if record["claim_eligible"] and (not source_path or not source_hash):
                errors.append(f"{record_id} claim-eligible record lacks artifact provenance")

        pairs[str(record["pair_id"])].append(record)
        group_splits[str(record["workload_group"])].add(str(record["split"]))
        counts_by_split[str(record["split"])][str(record["label"])] += 1

    for pair_id, pair_records in pairs.items():
        if len(pair_records) != 2:
            errors.append(f"pair {pair_id} has {len(pair_records)} records; expected 2")
            continue
        if {record["label"] for record in pair_records} != required_labels:
            errors.append(f"pair {pair_id} does not contain exactly one record per label")
        for field in ("operation", "split", "workload_group"):
            if len({record[field] for record in pair_records}) != 1:
                errors.append(f"pair {pair_id} differs on {field}")
        profiles = {
            record["provenance"].get("capture_profile_id")
            for record in pair_records
            if isinstance(record.get("provenance"), dict)
        }
        if len(profiles) != 1:
            errors.append(f"pair {pair_id} uses different capture profiles")

    for workload_group, splits in group_splits.items():
        if len(splits) != 1:
            errors.append(
                f"workload group {workload_group} leaks across splits {sorted(splits)}"
            )

    observed_splits = set(counts_by_split)
    missing_splits = sorted(required_splits.difference(observed_splits))
    if missing_splits:
        errors.append(f"missing required splits {missing_splits}")
    for split in required_splits.intersection(observed_splits):
        missing_labels = sorted(required_labels.difference(counts_by_split[split]))
        if missing_labels:
            errors.append(f"split {split} lacks labels {missing_labels}")
        compromised = sum(
            1
            for record in records
            if record.get("split") == split
            and record.get("label") == "attack_preparation"
            and record.get("account_state_label") == "compromised"
        )
        if compromised == 0:
            errors.append(f"split {split} lacks a compromised-admin attack record")

    report = {
        "schema_version": "context-validation-report-v1",
        "created_at": utc_now(),
        "valid": not errors,
        "errors": errors,
        "counts": {
            "records": len(records),
            "pairs": len(pairs),
            "workload_groups": len(group_splits),
            "synthetic_records": sum(bool(record.get("synthetic")) for record in records),
            "claim_eligible_records": sum(
                bool(record.get("claim_eligible")) for record in records
            ),
            "by_split": {
                split: dict(sorted(label_counts.items()))
                for split, label_counts in sorted(counts_by_split.items())
            },
            "compromised_admin_attacks_by_split": {
                split: sum(
                    1
                    for record in records
                    if record.get("split") == split
                    and record.get("label") == "attack_preparation"
                    and record.get("account_state_label") == "compromised"
                )
                for split in sorted(observed_splits)
            },
        },
        "guards": [
            "pair operation/split/workload/capture profile equality",
            "workload-group split disjointness",
            "synthetic claim exclusion",
            "compromised-admin coverage in every split",
            "forbidden-feature check from experiment config",
        ],
    }
    return report


def select_analysis_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Exclude an entire real-data pair when either side fails telemetry/quality gates."""
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)
    selected: list[dict[str, Any]] = []
    excluded_pairs: list[str] = []
    for pair_id, pair_records in pairs.items():
        synthetic_smoke_pair = all(record["synthetic"] for record in pair_records)
        real_eligible_pair = all(
            record["telemetry_equivalent"] and record["quality_eligible"]
            for record in pair_records
        )
        if synthetic_smoke_pair or real_eligible_pair:
            selected.extend(pair_records)
        else:
            excluded_pairs.append(pair_id)
    if not selected:
        raise StudyError("no complete analysis-eligible pairs remain after quality gates")
    return selected, sorted(excluded_pairs)


class Encoder:
    def __init__(self, categorical: list[str], numeric: list[str]) -> None:
        self.categorical = categorical
        self.numeric = numeric
        self.categories: dict[str, list[str]] = {}
        self.means: dict[str, float] = {}
        self.scales: dict[str, float] = {}
        self.feature_names: list[str] = ["intercept"]

    def fit(self, records: list[dict[str, Any]]) -> None:
        for field in self.categorical:
            values = sorted({str(get_field(record, field)) for record in records})
            self.categories[field] = values
            self.feature_names.extend(f"{field}={value}" for value in values)
            self.feature_names.append(f"{field}=__UNKNOWN__")
        for field in self.numeric:
            values = [float(get_field(record, field)) for record in records]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            scale = math.sqrt(variance)
            if scale < 1e-12:
                scale = 1.0
            self.means[field] = mean
            self.scales[field] = scale
            self.feature_names.append(field)

    def transform_one(self, record: dict[str, Any]) -> list[float]:
        row = [1.0]
        for field in self.categorical:
            value = str(get_field(record, field))
            categories = self.categories[field]
            row.extend(1.0 if value == category else 0.0 for category in categories)
            row.append(0.0 if value in categories else 1.0)
        for field in self.numeric:
            value = float(get_field(record, field))
            row.append((value - self.means[field]) / self.scales[field])
        return row

    def transform(self, records: list[dict[str, Any]]) -> list[list[float]]:
        return [self.transform_one(record) for record in records]

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical": self.categorical,
            "numeric": self.numeric,
            "categories": self.categories,
            "means": self.means,
            "scales": self.scales,
            "feature_names": self.feature_names,
        }


def sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def train_logistic(
    matrix: list[list[float]],
    labels: list[int],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> list[float]:
    if not matrix or len(matrix) != len(labels):
        raise StudyError("empty or misaligned training matrix")
    weights = [0.0] * len(matrix[0])
    count = len(matrix)
    for epoch in range(epochs):
        gradient = [0.0] * len(weights)
        for row, label in zip(matrix, labels):
            prediction = sigmoid(sum(weight * value for weight, value in zip(weights, row)))
            error = prediction - label
            for index, value in enumerate(row):
                gradient[index] += error * value
        rate = learning_rate / math.sqrt(1.0 + epoch / 200.0)
        for index in range(len(weights)):
            regularization = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= rate * (gradient[index] / count + regularization)
    return weights


def predict(matrix: list[list[float]], weights: list[float]) -> list[float]:
    return [sigmoid(sum(weight * value for weight, value in zip(weights, row))) for row in matrix]


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
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    accuracy = (tp + tn) / len(labels) if labels else None
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "accuracy": accuracy,
        "mcc": mcc,
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


def select_threshold(labels: list[int], scores: list[float], maximum_fpr: float) -> dict[str, Any]:
    unique = sorted(set(scores), reverse=True)
    candidates = [max(unique) + 1e-12, *unique, min(unique) - 1e-12]
    feasible: list[tuple[tuple[float, float, float], float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = confusion(labels, scores, threshold)
        fpr = metrics["fpr"]
        if fpr is not None and fpr <= maximum_fpr + 1e-12:
            key = (float(metrics["tpr"] or 0.0), -float(fpr), threshold)
            feasible.append((key, threshold, metrics))
    if not feasible:
        raise StudyError("no threshold satisfies the validation FPR cap")
    _, threshold, metrics = max(feasible, key=lambda item: item[0])
    return {"threshold": threshold, "validation_metrics": metrics}


def pair_ranking(records: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for record, score in zip(records, scores):
        grouped[record["pair_id"]].append((record["label"], score))
    wins = ties = losses = 0
    for pair in grouped.values():
        by_label = {label: score for label, score in pair}
        if set(by_label) != set(LABEL_TO_INT):
            continue
        attack = by_label["attack_preparation"]
        legitimate = by_label["legitimate_admin"]
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


def split_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [record for record in records if record["split"] == split]


def audit_subgroup(
    records: list[dict[str, Any]], scores: list[float], threshold: float, state: str
) -> dict[str, Any]:
    selected = [
        (record, score)
        for record, score in zip(records, scores)
        if record["label"] == "attack_preparation"
        and record["account_state_label"] == state
    ]
    detected = sum(score >= threshold for _, score in selected)
    return {
        "account_state_label": state,
        "attack_records": len(selected),
        "detected": detected,
        "detection_rate": detected / len(selected) if selected else None,
    }


def run_models(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    development = split_records(records, "development")
    validation = split_records(records, "validation")
    test = split_records(records, "test")
    training = config["training"]
    maximum_fpr = float(
        config["primary_comparison"]["maximum_validation_false_positive_rate"]
    )
    report_models: dict[str, Any] = {}
    model_artifacts: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []

    for model_name, feature_specification in config["feature_sets"].items():
        encoder = Encoder(
            list(feature_specification.get("categorical", [])),
            list(feature_specification.get("numeric", [])),
        )
        encoder.fit(development)
        development_matrix = encoder.transform(development)
        validation_matrix = encoder.transform(validation)
        test_matrix = encoder.transform(test)
        development_labels = [LABEL_TO_INT[record["label"]] for record in development]
        validation_labels = [LABEL_TO_INT[record["label"]] for record in validation]
        test_labels = [LABEL_TO_INT[record["label"]] for record in test]
        weights = train_logistic(
            development_matrix,
            development_labels,
            epochs=int(training["epochs"]),
            learning_rate=float(training["learning_rate"]),
            l2=float(training["l2"]),
        )
        development_scores = predict(development_matrix, weights)
        validation_scores = predict(validation_matrix, weights)
        test_scores = predict(test_matrix, weights)
        selection = select_threshold(validation_labels, validation_scores, maximum_fpr)
        threshold = float(selection["threshold"])
        test_metrics = confusion(test_labels, test_scores, threshold)
        report_models[model_name] = {
            "feature_count_including_intercept": len(weights),
            "threshold_selected_on_validation": threshold,
            "maximum_validation_fpr": maximum_fpr,
            "development": {
                "metrics": confusion(development_labels, development_scores, threshold),
                "auc": auc(development_labels, development_scores),
                "pair_ranking": pair_ranking(development, development_scores),
            },
            "validation": {
                "metrics": selection["validation_metrics"],
                "auc": auc(validation_labels, validation_scores),
                "pair_ranking": pair_ranking(validation, validation_scores),
            },
            "test": {
                "metrics": test_metrics,
                "auc": auc(test_labels, test_scores),
                "pair_ranking": pair_ranking(test, test_scores),
                "compromised_admin_audit": audit_subgroup(
                    test, test_scores, threshold, "compromised"
                ),
                "other_attack_audit": audit_subgroup(
                    test, test_scores, threshold, "not_applicable"
                ),
            },
        }
        model_artifacts[model_name] = {
            "encoder": encoder.to_dict(),
            "weights": dict(zip(encoder.feature_names, weights)),
            "threshold": threshold,
        }
        for split_name, split_rows, split_scores in (
            ("development", development, development_scores),
            ("validation", validation, validation_scores),
            ("test", test, test_scores),
        ):
            for record, score in zip(split_rows, split_scores):
                prediction_rows.append(
                    {
                        "record_id": record["record_id"],
                        "pair_id": record["pair_id"],
                        "workload_group": record["workload_group"],
                        "split": split_name,
                        "operation": record["operation"],
                        "label": record["label"],
                        "account_state_label": record["account_state_label"],
                        "model": model_name,
                        "score": score,
                        "threshold": threshold,
                        "alerted": score >= threshold,
                    }
                )

    baseline_name = config["primary_comparison"]["baseline_model"]
    candidate_name = config["primary_comparison"]["candidate_model"]
    baseline = report_models[baseline_name]["test"]["metrics"]
    candidate = report_models[candidate_name]["test"]["metrics"]
    primary = {
        "baseline_model": baseline_name,
        "candidate_model": candidate_name,
        "test_tpr_difference": candidate["tpr"] - baseline["tpr"],
        "test_fpr_difference": candidate["fpr"] - baseline["fpr"],
        "test_mcc_difference": candidate["mcc"] - baseline["mcc"],
        "interpretation_guard": "Bootstrap differences validate pipeline mechanics only when any input record is synthetic or claim-ineligible.",
    }
    return (
        report_models,
        {"primary_comparison": primary},
        model_artifacts,
        prediction_rows,
    )


def render_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    validation = report["data_validation"]
    primary = report["primary_comparison"]
    lines = [
        "# RIVF context-study automated result",
        "",
        f"Status: **{report['status']}**. Generated: {report['created_at']}.",
        "",
        "> This run validates the automated analysis path. Synthetic or claim-ineligible input cannot support a paper efficacy claim.",
        "",
        "## Dataset",
        "",
        f"- Records: {validation['counts']['records']}",
        f"- Matched pairs: {validation['counts']['pairs']}",
        f"- Workload groups: {validation['counts']['workload_groups']}",
        f"- Synthetic records: {validation['counts']['synthetic_records']}",
        f"- Claim-eligible records: {validation['counts']['claim_eligible_records']}",
        f"- Analysis records after pairwise quality gates: {validation['counts']['analysis_records']}",
        f"- Excluded quality pairs: {validation['counts']['excluded_quality_pairs']}",
        "",
        "## Same-FPR comparison",
        "",
        "Thresholds were selected independently on validation under the configured FPR cap and then applied unchanged to workload-disjoint test records.",
        "",
        "| Model | Val FPR | Val TPR | Test FPR | Test TPR | Test precision | Test MCC | Test AUC | Pair ranking | Compromised-admin detection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, model in report["models"].items():
        validation_metrics = model["validation"]["metrics"]
        test_metrics = model["test"]["metrics"]
        compromised = model["test"]["compromised_admin_audit"]
        lines.append(
            f"| `{model_name}` | {render_number(validation_metrics['fpr'])} | "
            f"{render_number(validation_metrics['tpr'])} | {render_number(test_metrics['fpr'])} | "
            f"{render_number(test_metrics['tpr'])} | {render_number(test_metrics['precision'])} | "
            f"{render_number(test_metrics['mcc'])} | {render_number(model['test']['auc'])} | "
            f"{render_number(model['test']['pair_ranking']['pair_ranking_accuracy'])} | "
            f"{compromised['detected']}/{compromised['attack_records']} |"
        )
    lines.extend(
        [
            "",
            "## Primary comparison",
            "",
            f"- Baseline: `{primary['baseline_model']}`",
            f"- Candidate: `{primary['candidate_model']}`",
            f"- Test TPR difference: {render_number(primary['test_tpr_difference'])}",
            f"- Test FPR difference: {render_number(primary['test_fpr_difference'])}",
            f"- Test MCC difference: {render_number(primary['test_mcc_difference'])}",
            "",
            "## Scientific guards",
            "",
        ]
    )
    lines.extend(f"- {guard}" for guard in report["scientific_guards"])
    return "\n".join(lines) + "\n"


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "record_id",
        "pair_id",
        "workload_group",
        "split",
        "operation",
        "label",
        "account_state_label",
        "model",
        "score",
        "threshold",
        "alerted",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_json(args.config)
        validate_config(config)
        records = load_jsonl(args.input)
        validation = validate_records(records, config)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not validation["valid"]:
            write_json(args.output_dir / "validation-report.json", validation)
            raise StudyError("record validation failed: " + "; ".join(validation["errors"]))
        analysis_records, excluded_pairs = select_analysis_records(records)
        validation["counts"]["analysis_records"] = len(analysis_records)
        validation["counts"]["excluded_quality_pairs"] = len(excluded_pairs)
        validation["excluded_quality_pair_ids"] = excluded_pairs
        write_json(args.output_dir / "validation-report.json", validation)
        models, primary, model_artifacts, predictions = run_models(analysis_records, config)
        status = (
            "engineering_smoke_test_only"
            if any(record["synthetic"] for record in analysis_records)
            else (
                "held_out_result_candidate_requires_manual_audit"
                if all(
                    record["claim_eligible"]
                    for record in analysis_records
                    if record["split"] == "test"
                )
                else "exploratory_real_data_not_claim_eligible"
            )
        )
        report = {
            "schema_version": "context-model-report-v1",
            "created_at": utc_now(),
            "status": status,
            "study_id": config["study_id"],
            "research_question": config["research_question"],
            "data_validation": validation,
            "models": models,
            "primary_comparison": primary["primary_comparison"],
            "scientific_guards": config["scientific_guards"],
            "input_identity": {
                "path": str(args.input),
                "sha256": sha256_file(args.input),
            },
            "config_identity": {
                "path": str(args.config),
                "sha256": sha256_file(args.config),
            },
        }
        write_json(args.output_dir / "model-report.json", report)
        write_json(
            args.output_dir / "model-artifacts.json",
            {
                "schema_version": "context-model-artifacts-v1",
                "created_at": utc_now(),
                "status": status,
                "models": model_artifacts,
            },
        )
        write_predictions(args.output_dir / "predictions.csv", predictions)
        write_text(args.output_dir / "results.md", render_markdown(report))
        output_names = [
            "validation-report.json",
            "model-report.json",
            "model-artifacts.json",
            "predictions.csv",
            "results.md",
        ]
        manifest = {
            "schema_version": "context-run-manifest-v1",
            "created_at": utc_now(),
            "status": status,
            "inputs": {
                str(args.config): sha256_file(args.config),
                str(args.input): sha256_file(args.input),
                str(Path(__file__)): sha256_file(Path(__file__)),
            },
            "outputs": {
                name: sha256_file(args.output_dir / name) for name in output_names
            },
        }
        write_json(args.output_dir / "run-manifest.json", manifest)
        print(
            json.dumps(
                {
                    "status": status,
                    "records": len(records),
                    "analysis_records": len(analysis_records),
                    "pairs": validation["counts"]["pairs"],
                    "output_dir": str(args.output_dir),
                },
                sort_keys=True,
            )
        )
        return 0
    except StudyError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
