#!/usr/bin/env python3
"""Run a transparent post-hoc Random Forest robustness check and paper figures.

This analysis was specified after the v5 held-out result was opened.  It must be
reported as exploratory/post-hoc and must not replace the frozen logistic
primary analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


LABEL_TO_INT = {"legitimate_admin": 0, "attack_preparation": 1}
CATEGORICAL = ["operation", "launcher_type", "account_role"]
NUMERIC = [
    "observed_whoami",
    "observed_hostname",
    "observed_ipconfig",
    "observed_tasklist",
]
RF_PARAMETERS = {
    "n_estimators": 1000,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 20260914,
    "n_jobs": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-dataset", type=Path, required=True)
    parser.add_argument("--v6-dataset", type=Path, required=True)
    parser.add_argument("--logistic-predictions", type=Path, required=True)
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
            if line.strip():
                value = json.loads(line)
                if value.get("label") not in LABEL_TO_INT:
                    raise ValueError(f"{path}:{line_number}: unsupported label")
                records.append(value)
    return records


def rows_for_model(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        context = record["context"]
        row = {
            "operation": record["operation"],
            "launcher_type": context["launcher_type"],
            "account_role": context["account_role"],
        }
        row.update({field: int(bool(context[field])) for field in NUMERIC})
        rows.append(row)
    return rows


def matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row[field] for field in CATEGORICAL + NUMERIC] for row in rows],
        dtype=object,
    )


def labels(records: list[dict]) -> np.ndarray:
    return np.asarray([LABEL_TO_INT[record["label"]] for record in records])


def split(records: list[dict], name: str) -> list[dict]:
    return [record for record in records if record["split"] == name]


def metric_summary(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    return {
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "tpr": float(tp / (tp + fn)),
        "fpr": float(fp / (fp + tn)),
        "precision": precision,
        "accuracy": float((tp + tn) / len(y_true)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, scores)),
    }


def select_threshold(y_true: np.ndarray, scores: np.ndarray, maximum_fpr: float) -> tuple[float, dict]:
    candidates = sorted(set(float(value) for value in scores))
    candidates.append(max(candidates) + 1e-12)
    feasible = []
    for threshold in candidates:
        summary = metric_summary(y_true, scores, threshold)
        if summary["fpr"] <= maximum_fpr + 1e-12:
            key = (summary["tpr"], -summary["fpr"], threshold)
            feasible.append((key, threshold, summary))
    if not feasible:
        raise RuntimeError("no threshold satisfies the validation FPR cap")
    _, threshold, summary = max(feasible, key=lambda item: item[0])
    return float(threshold), summary


def pair_ranking(records: list[dict], scores: np.ndarray) -> dict:
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for record, score in zip(records, scores, strict=True):
        grouped[record["pair_id"]][record["label"]] = float(score)
    wins = ties = losses = 0
    for pair_id, values in grouped.items():
        if set(values) != set(LABEL_TO_INT):
            raise ValueError(f"incomplete pair: {pair_id}")
        delta = values["attack_preparation"] - values["legitimate_admin"]
        if math.isclose(delta, 0.0, abs_tol=1e-12):
            ties += 1
        elif delta > 0:
            wins += 1
        else:
            losses += 1
    total = len(grouped)
    return {
        "pairs": total,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "accuracy_with_half_credit_for_ties": (wins + 0.5 * ties) / total,
    }


def subgroup(records: list[dict], scores: np.ndarray, threshold: float, state: str) -> dict:
    selected = [
        float(score)
        for record, score in zip(records, scores, strict=True)
        if record["label"] == "attack_preparation"
        and record["account_state_label"] == state
    ]
    detected = sum(score >= threshold for score in selected)
    return {
        "account_state_label": state,
        "attack_records": len(selected),
        "detected": detected,
        "detection_rate": detected / len(selected),
    }


def load_logistic_test_predictions(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and row["model"] in {
                "operation_only",
                "measured_process_context",
            }:
                grouped[row["model"]].append(row)
    result = {}
    for model, rows in grouped.items():
        result[model] = (
            np.asarray([LABEL_TO_INT[row["label"]] for row in rows]),
            np.asarray([float(row["score"]) for row in rows]),
            np.asarray([row["alerted"].casefold() == "true" for row in rows], dtype=int),
        )
    if set(result) != {"operation_only", "measured_process_context"}:
        raise ValueError("missing frozen logistic test predictions")
    return result


def plot_diagnostics(
    frozen: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    rf_y: np.ndarray,
    rf_scores: np.ndarray,
    rf_threshold: float,
    output: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
    })
    fig = plt.figure(figsize=(7.15, 2.45), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.75])
    roc_ax = fig.add_subplot(grid[0, 0])
    colors = {
        "Operation only": "#7f7f7f",
        "Logistic context": "#1769aa",
        "Random Forest": "#c43c39",
    }
    roc_inputs = [
        ("Operation only", frozen["operation_only"][0], frozen["operation_only"][1]),
        (
            "Logistic context",
            frozen["measured_process_context"][0],
            frozen["measured_process_context"][1],
        ),
        ("Random Forest", rf_y, rf_scores),
    ]
    for label, y_true, scores in roc_inputs:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        roc_ax.step(fpr, tpr, where="post", linewidth=1.6, color=colors[label], label=f"{label} ({auc:.3f})")
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.6)
    roc_ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    roc_ax.set_title("(a) Held-out ROC")
    roc_ax.grid(alpha=0.2, linewidth=0.5)
    roc_ax.legend(loc="lower right", frameon=False)

    cm_grid = grid[0, 1].subgridspec(1, 3, wspace=0.28)
    cm_inputs = [
        ("Op-only", frozen["operation_only"][0], frozen["operation_only"][2]),
        ("Logistic", frozen["measured_process_context"][0], frozen["measured_process_context"][2]),
        ("RF", rf_y, (rf_scores >= rf_threshold).astype(int)),
    ]
    images = []
    for index, (label, y_true, y_pred) in enumerate(cm_inputs):
        ax = fig.add_subplot(cm_grid[0, index])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=24)
        images.append(image)
        for row in range(2):
            for column in range(2):
                value = int(cm[row, column])
                ax.text(column, row, str(value), ha="center", va="center", color="white" if value > 12 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Legit.", "Attack"], rotation=35, ha="right")
        ax.set_yticks([0, 1], ["Legit.", "Attack"] if index == 0 else ["", ""])
        ax.set_xlabel("Predicted")
        if index == 0:
            ax.set_ylabel("True")
        ax.set_title(label)
    fig.colorbar(images[-1], ax=[fig.axes[-3], fig.axes[-2], fig.axes[-1]], fraction=0.035, pad=0.02)
    fig.text(0.73, 1.01, "(b) Held-out confusion matrices", ha="center", va="bottom", fontsize=8.5)
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rounded_box(ax, x: float, y: float, text: str, color: str, width: float = 0.22) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": "#333333", "linewidth": 0.7},
    )


def arrow(ax, start: tuple[float, float], end: tuple[float, float], dashed: bool = False) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#333333", "linestyle": "--" if dashed else "-"},
    )


def plot_process_chains(output: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 7.5})
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.35), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Batch preparation: context is observable", loc="left", fontsize=8)
    rounded_box(ax, 0.12, 0.46, "PowerShell", "#dceeff")
    rounded_box(ax, 0.42, 0.73, "Discovery\n0/2/4 tools", "#fff0cc")
    rounded_box(ax, 0.42, 0.28, "cmd.exe", "#f8d7da")
    rounded_box(ax, 0.78, 0.28, "Admin target\n(same operation)", "#e2f0d9")
    arrow(ax, (0.23, 0.53), (0.33, 0.68), dashed=True)
    arrow(ax, (0.23, 0.41), (0.33, 0.31))
    arrow(ax, (0.53, 0.28), (0.65, 0.28))

    ax = axes[1]
    ax.set_title("(b) Direct compromised admin: exact observable overlap", loc="left", fontsize=8)
    rounded_box(ax, 0.12, 0.68, "Legitimate\nPowerShell", "#dceeff")
    rounded_box(ax, 0.78, 0.68, "Admin target", "#e2f0d9")
    rounded_box(ax, 0.12, 0.25, "Compromised\nPowerShell", "#f8d7da")
    rounded_box(ax, 0.78, 0.25, "Same target", "#e2f0d9")
    arrow(ax, (0.24, 0.68), (0.66, 0.68))
    arrow(ax, (0.24, 0.25), (0.66, 0.25))
    ax.text(0.5, 0.47, "same measured launcher + process indicators", ha="center", va="center", fontsize=6.8, color="#555555")
    fig.savefig(output, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")

    v5 = load_jsonl(args.v5_dataset)
    v6 = load_jsonl(args.v6_dataset)
    development = split(v5, "development")
    validation = split(v5, "validation")
    test = split(v5, "test")
    if [len(development), len(validation), len(test), len(v6)] != [48, 48, 48, 48]:
        raise ValueError("unexpected split sizes")

    transformer = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(len(CATEGORICAL)))),
            ("numeric", "passthrough", list(range(len(CATEGORICAL), len(CATEGORICAL) + len(NUMERIC)))),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(**RF_PARAMETERS)
    model = Pipeline([("encode", transformer), ("classifier", classifier)])
    model.fit(matrix(rows_for_model(development)), labels(development))

    split_scores = {
        "development": model.predict_proba(matrix(rows_for_model(development)))[:, 1],
        "validation": model.predict_proba(matrix(rows_for_model(validation)))[:, 1],
        "test": model.predict_proba(matrix(rows_for_model(test)))[:, 1],
        "v6_locked": model.predict_proba(matrix(rows_for_model(v6)))[:, 1],
    }
    threshold, validation_metrics = select_threshold(labels(validation), split_scores["validation"], 0.05)

    report = {
        "schema_version": "rivf-posthoc-random-forest-robustness-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_exploratory_after_v5_test_opened",
        "claim_guard": (
            "Random Forest was added after inspection of the frozen v5 result. "
            "It is a robustness check, not a preregistered primary or confirmatory result."
        ),
        "inputs": {
            "v5_dataset": {"path": str(args.v5_dataset), "sha256": sha256_file(args.v5_dataset)},
            "v6_dataset": {"path": str(args.v6_dataset), "sha256": sha256_file(args.v6_dataset)},
            "logistic_predictions": {
                "path": str(args.logistic_predictions),
                "sha256": sha256_file(args.logistic_predictions),
            },
        },
        "software": {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "features": {"categorical": CATEGORICAL, "binary": NUMERIC},
        "model": {"algorithm": "sklearn.ensemble.RandomForestClassifier", "parameters": RF_PARAMETERS},
        "selection": {
            "fit_split": "v5 development only",
            "threshold_split": "v5 validation only",
            "maximum_validation_fpr": 0.05,
            "selected_threshold": threshold,
            "validation_metrics": validation_metrics,
        },
        "results": {},
        "no_v6_retraining": True,
    }

    split_records = {
        "development": development,
        "validation": validation,
        "test": test,
        "v6_locked": v6,
    }
    for name, records in split_records.items():
        current_labels = labels(records)
        current_scores = split_scores[name]
        result = {
            "metrics": metric_summary(current_labels, current_scores, threshold),
            "pair_ranking": pair_ranking(records, current_scores),
        }
        if name in {"test", "v6_locked"}:
            result["compromised_admin_audit"] = subgroup(records, current_scores, threshold, "compromised")
            result["other_attack_audit"] = subgroup(records, current_scores, threshold, "not_applicable")
        report["results"][name] = result

    frozen = load_logistic_test_predictions(args.logistic_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "random-forest-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (args.output_dir / "random-forest-predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["collection", "record_id", "pair_id", "split", "operation", "label", "account_state_label", "score", "threshold", "alerted"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, records in split_records.items():
            collection = "v6" if name == "v6_locked" else "v5"
            for record, score in zip(records, split_scores[name], strict=True):
                writer.writerow({
                    "collection": collection,
                    "record_id": record["record_id"],
                    "pair_id": record["pair_id"],
                    "split": name,
                    "operation": record["operation"],
                    "label": record["label"],
                    "account_state_label": record["account_state_label"],
                    "score": f"{float(score):.17g}",
                    "threshold": f"{threshold:.17g}",
                    "alerted": bool(score >= threshold),
                })

    plot_diagnostics(
        frozen,
        labels(test),
        split_scores["test"],
        threshold,
        args.output_dir / "model-diagnostics.png",
    )
    plot_process_chains(args.output_dir / "process-chain-comparison.png")
    print(json.dumps({
        "output": str(args.output_dir),
        "threshold": threshold,
        "test": report["results"]["test"],
        "v6_locked": report["results"]["v6_locked"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
