#!/usr/bin/env python3
"""Build auditable paper evidence from frozen per-record scores.

This script does not fit or tune a model. It reconstructs held-out ROC points
from the exported v5 scores, groups equal scores before advancing the curve,
cross-checks trapezoidal ROC AUC against the Mann--Whitney definition with
half credit for ties, exports the three-level ablation, and enumerates exact
within-pair feature collisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score


LABEL = {"legitimate_admin": 0, "attack_preparation": 1}
LOGISTIC_MODELS = [
    "operation_only",
    "measured_actor_context",
    "measured_process_context",
    "timing_proxy_sensitivity",
]
DISCOVERY = [
    "observed_whoami",
    "observed_hostname",
    "observed_ipconfig",
    "observed_tasklist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-dataset", type=Path, required=True)
    parser.add_argument("--logistic-predictions", type=Path, required=True)
    parser.add_argument("--rf-predictions", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("label") not in LABEL:
                raise ValueError(f"{path}:{line_number}: unsupported label")
            records.append(record)
    return records


def load_logistic(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and row["model"] in LOGISTIC_MODELS:
                grouped[row["model"]].append(row)
    if set(grouped) != set(LOGISTIC_MODELS):
        raise ValueError(f"missing logistic test predictions: {sorted(grouped)}")
    for model, rows in grouped.items():
        if len(rows) != 48:
            raise ValueError(f"{model}: expected 48 test rows, got {len(rows)}")
    return dict(grouped)


def load_rf(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["collection"] == "v5" and row["split"] == "test":
                rows.append(row)
    if len(rows) != 48:
        raise ValueError(f"RF: expected 48 v5 test rows, got {len(rows)}")
    return rows


def score_rows(rows: list[dict]) -> tuple[list[int], list[float], float]:
    labels = [LABEL[row["label"]] for row in rows]
    scores = [float(row["score"]) for row in rows]
    thresholds = {float(row["threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise ValueError("prediction file contains multiple thresholds")
    return labels, scores, thresholds.pop()


def confusion(labels: list[int], scores: list[float], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores, strict=True):
        alerted = score >= threshold
        if label and alerted:
            tp += 1
        elif label:
            fn += 1
        elif alerted:
            fp += 1
        else:
            tn += 1
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / (tp + fn),
        "fpr": fp / (fp + tn),
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "accuracy": (tp + tn) / len(labels),
        "mcc": 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator,
    }


def mann_whitney_auc(labels: list[int], scores: list[float]) -> tuple[float, int, int, int]:
    positive = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negative = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    wins = ties = losses = 0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1
            elif pos == neg:
                ties += 1
            else:
                losses += 1
    auc = (wins + 0.5 * ties) / (len(positive) * len(negative))
    return auc, wins, ties, losses


def grouped_roc(labels: list[int], scores: list[float]) -> list[dict]:
    positives = sum(labels)
    negatives = len(labels) - positives
    by_score: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(labels, scores, strict=True):
        by_score[score].append(label)
    points = [{
        "threshold": "+inf",
        "group_positive": 0,
        "group_negative": 0,
        "cumulative_tp": 0,
        "cumulative_fp": 0,
        "tpr": 0.0,
        "fpr": 0.0,
    }]
    tp = fp = 0
    for score in sorted(by_score, reverse=True):
        group = by_score[score]
        group_positive = sum(group)
        group_negative = len(group) - group_positive
        tp += group_positive
        fp += group_negative
        points.append({
            "threshold": score,
            "group_positive": group_positive,
            "group_negative": group_negative,
            "cumulative_tp": tp,
            "cumulative_fp": fp,
            "tpr": tp / positives,
            "fpr": fp / negatives,
        })
    return points


def trapezoid_auc(points: list[dict]) -> float:
    total = 0.0
    for left, right in zip(points, points[1:], strict=False):
        width = right["fpr"] - left["fpr"]
        total += width * (left["tpr"] + right["tpr"]) / 2.0
    return total


def pair_ranking(rows: list[dict]) -> dict:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[row["pair_id"]][row["label"]] = float(row["score"])
    wins = ties = losses = 0
    for pair_id, values in grouped.items():
        if set(values) != set(LABEL):
            raise ValueError(f"incomplete pair: {pair_id}")
        delta = values["attack_preparation"] - values["legitimate_admin"]
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "accuracy_half_credit": (wins + 0.5 * ties) / len(grouped),
    }


def feature_vector(record: dict) -> tuple:
    context = record["context"]
    return (
        record["operation"],
        context["launcher_type"],
        context["account_role"],
        *(int(bool(context[field])) for field in DISCOVERY),
    )


def collision_rows(records: list[dict], full_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    test = [record for record in records if record["split"] == "test"]
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for record in test:
        by_pair[record["pair_id"]].append(record)
    scores = {row["record_id"]: row for row in full_rows}
    all_compromised = []
    collisions = []
    for pair_id, pair in sorted(by_pair.items()):
        if len(pair) != 2:
            raise ValueError(f"{pair_id}: expected two records")
        by_label = {record["label"]: record for record in pair}
        attack = by_label["attack_preparation"]
        if attack["account_state_label"] != "compromised":
            continue
        legitimate = by_label["legitimate_admin"]
        attack_prediction = scores[attack["record_id"]]
        legit_vector = feature_vector(legitimate)
        attack_vector = feature_vector(attack)
        row = {
            "pair_id": pair_id,
            "operation": attack["operation"],
            "legitimate_launcher": legit_vector[1],
            "attack_launcher": attack_vector[1],
            "legitimate_discovery_WHIT": "".join(str(value) for value in legit_vector[3:]),
            "attack_discovery_WHIT": "".join(str(value) for value in attack_vector[3:]),
            "legitimate_records": 1,
            "attack_records": 1,
            "exact_feature_collision": legit_vector == attack_vector,
            "attack_score": float(attack_prediction["score"]),
            "attack_threshold": float(attack_prediction["threshold"]),
            "attack_alerted": attack_prediction["alerted"].casefold() == "true",
        }
        all_compromised.append(row)
        if row["exact_feature_collision"]:
            collisions.append(row)
    if len(all_compromised) != 12 or len(collisions) != 8:
        raise ValueError(
            f"expected 12 compromised pairs and 8 collisions, got "
            f"{len(all_compromised)} and {len(collisions)}"
        )
    return all_compromised, collisions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_roc(curves: list[dict], output: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
    })
    colors = ["#7f7f7f", "#1769aa", "#c43c39"]
    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)
    for curve, color in zip(curves, colors, strict=True):
        x = [point["fpr"] for point in curve["points"]]
        y = [point["tpr"] for point in curve["points"]]
        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.55,
            marker="o",
            markersize=2.3,
            label=f'{curve["display_name"]} ({curve["auc_rank"]:.3f})',
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.55)
    ax.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        title="Held-out ROC from grouped raw scores",
    )
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")

    dataset = load_jsonl(args.v5_dataset)
    logistic = load_logistic(args.logistic_predictions)
    rf_rows = load_rf(args.rf_predictions)
    model_report = json.loads(args.model_report.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)

    display_names = {
        "operation_only": "Operation only",
        "measured_process_context": "Logistic context",
        "random_forest": "Random Forest",
    }
    roc_sources = {
        "operation_only": logistic["operation_only"],
        "measured_process_context": logistic["measured_process_context"],
        "random_forest": rf_rows,
    }
    curves = []
    roc_csv = []
    for name, rows in roc_sources.items():
        labels, scores, _ = score_rows(rows)
        points = grouped_roc(labels, scores)
        rank_auc, wins, ties, losses = mann_whitney_auc(labels, scores)
        trap_auc = trapezoid_auc(points)
        library_auc = float(roc_auc_score(np.asarray(labels), np.asarray(scores)))
        if not (
            math.isclose(rank_auc, trap_auc, abs_tol=1e-12)
            and math.isclose(rank_auc, library_auc, abs_tol=1e-12)
        ):
            raise ValueError(f"AUC cross-check failed for {name}")
        curve = {
            "model": name,
            "display_name": display_names[name],
            "unique_score_groups": len(points) - 1,
            "auc_rank": rank_auc,
            "auc_trapezoid_grouped_points": trap_auc,
            "auc_sklearn": library_auc,
            "positive_negative_score_wins": wins,
            "positive_negative_score_ties": ties,
            "positive_negative_score_losses": losses,
            "points": points,
        }
        curves.append(curve)
        for sequence, point in enumerate(points):
            roc_csv.append({"model": name, "sequence": sequence, **point})
    write_csv(args.output_dir / "roc-points.csv", roc_csv)
    plot_roc(curves, args.output_dir / "model-diagnostics.png")

    ablation = []
    for level, name in enumerate(LOGISTIC_MODELS):
        rows = logistic[name]
        labels, scores, threshold = score_rows(rows)
        auc, wins, ties, losses = mann_whitney_auc(labels, scores)
        metrics = confusion(labels, scores, threshold)
        paired = pair_ranking(rows)
        compromised = [
            row for row in rows
            if row["label"] == "attack_preparation"
            and row["account_state_label"] == "compromised"
        ]
        ablation.append({
            "level": level,
            "model": name,
            "threshold": threshold,
            **metrics,
            "auc": auc,
            "pair_wins": paired["wins"],
            "pair_ties": paired["ties"],
            "pair_losses": paired["losses"],
            "pair_ranking_accuracy": paired["accuracy_half_credit"],
            "compromised_admin_detected": sum(
                float(row["score"]) >= threshold for row in compromised
            ),
            "compromised_admin_total": len(compromised),
            "analysis_status": (
                "audit_motivated_prefrozen_sensitivity"
                if name == "timing_proxy_sensitivity"
                else "frozen_primary_ablation"
            ),
        })
    write_csv(args.output_dir / "ablation-table.csv", ablation)

    full_rows = logistic["measured_process_context"]
    compromised, collisions = collision_rows(dataset, full_rows)
    write_csv(args.output_dir / "compromised-admin-pair-audit.csv", compromised)
    write_csv(args.output_dir / "exact-collision-pairs.csv", collisions)

    merged_scores: dict[str, dict] = {}
    for name, rows in logistic.items():
        for row in rows:
            merged = merged_scores.setdefault(row["record_id"], {
                "record_id": row["record_id"],
                "pair_id": row["pair_id"],
                "operation": row["operation"],
                "label": row["label"],
                "account_state_label": row["account_state_label"],
            })
            merged[f"{name}_score"] = row["score"]
            merged[f"{name}_threshold"] = row["threshold"]
    for row in rf_rows:
        merged_scores[row["record_id"]]["random_forest_score"] = row["score"]
        merged_scores[row["record_id"]]["random_forest_threshold"] = row["threshold"]
    write_csv(
        args.output_dir / "heldout-scores.csv",
        [merged_scores[key] for key in sorted(merged_scores)],
    )

    report = {
        "schema_version": "rivf-paper-evidence-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "audited_reconstruction_from_frozen_scores",
        "inputs": {
            "v5_dataset": {"path": str(args.v5_dataset), "sha256": sha256_file(args.v5_dataset)},
            "logistic_predictions": {"path": str(args.logistic_predictions), "sha256": sha256_file(args.logistic_predictions)},
            "rf_predictions": {"path": str(args.rf_predictions), "sha256": sha256_file(args.rf_predictions)},
            "model_report": {"path": str(args.model_report), "sha256": sha256_file(args.model_report)},
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "roc_method": (
            "Unique score values are processed in descending order. All records "
            "with an equal score enter the ROC point together. Adjacent grouped "
            "points are connected linearly; trapezoidal AUC is cross-checked "
            "against Mann--Whitney AUC with 0.5 credit for positive-negative ties."
        ),
        "roc": [{key: value for key, value in curve.items() if key != "points"} for curve in curves],
        "ablation": ablation,
        "collision_audit": {
            "compromised_admin_pairs": len(compromised),
            "exact_feature_collision_pairs": len(collisions),
            "noncollision_compromised_pairs": len(compromised) - len(collisions),
            "detected_collision_attacks": sum(row["attack_alerted"] for row in collisions),
            "detected_noncollision_attacks": sum(
                row["attack_alerted"] for row in compromised if not row["exact_feature_collision"]
            ),
            "claim_boundary": (
                "The eight collisions bound deterministic pointwise models using "
                "this exact feature projection on this frozen test set. They do "
                "not show that the underlying telemetry or sensor contains no "
                "additional usable information."
            ),
        },
        "threshold_rule_from_frozen_engine": {
            "maximum_validation_fpr": model_report["models"]["measured_process_context"]["maximum_validation_fpr"],
            "rule": (
                "Enumerate thresholds above the maximum score, at every unique "
                "validation score, and below the minimum. Among thresholds with "
                "FPR <= 0.05, maximize TPR; break ties by lower FPR, then higher threshold."
            ),
        },
        "outputs": [
            "model-diagnostics.png",
            "roc-points.csv",
            "ablation-table.csv",
            "compromised-admin-pair-audit.csv",
            "exact-collision-pairs.csv",
            "heldout-scores.csv",
        ],
    }
    report_path = args.output_dir / "paper-evidence-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "output": str(args.output_dir),
        "roc": report["roc"],
        "collision_audit": report["collision_audit"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
