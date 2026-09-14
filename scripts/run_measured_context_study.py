#!/usr/bin/env python3
"""Run the v5 measured-context analysis while reusing the frozen model engine."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import run_context_study as base


MEASURED_FIELDS = {
    "observed_whoami",
    "observed_hostname",
    "observed_ipconfig",
    "observed_tasklist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_v2_and_make_base_view(records: list[dict]) -> list[dict]:
    base_view: list[dict] = []
    errors: list[str] = []
    for record in records:
        record_id = str(record.get("record_id", "<unknown>"))
        if record.get("schema_version") != "context-record-v2":
            errors.append(f"{record_id}: expected context-record-v2")
            continue
        context = record.get("context")
        if not isinstance(context, dict):
            errors.append(f"{record_id}: context is not an object")
            continue
        missing = sorted(MEASURED_FIELDS.difference(context))
        if missing:
            errors.append(f"{record_id}: missing measured fields {missing}")
        for field in MEASURED_FIELDS.intersection(context):
            if type(context[field]) is not bool:
                errors.append(f"{record_id}: context.{field} is not boolean")
        stripped = copy.deepcopy(record)
        stripped["schema_version"] = "context-record-v1"
        for field in MEASURED_FIELDS:
            stripped["context"].pop(field, None)
        base_view.append(stripped)
    if errors:
        raise base.StudyError("measured-context validation failed: " + "; ".join(errors))
    return base_view


def main() -> int:
    args = parse_args()
    try:
        if args.output_dir.exists():
            raise base.StudyError(f"refusing to overwrite output directory: {args.output_dir}")
        config = base.load_json(args.config)
        base.validate_config(config)
        records = base.load_jsonl(args.input)
        base_view = validate_v2_and_make_base_view(records)
        validation = base.validate_records(base_view, config)
        validation["input_schema_version"] = "context-record-v2"
        validation["measured_feature_validation"] = {
            "valid": True,
            "fields": sorted(MEASURED_FIELDS),
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if not validation["valid"]:
            base.write_json(args.output_dir / "validation-report.json", validation)
            raise base.StudyError("record validation failed: " + "; ".join(validation["errors"]))
        analysis_records, excluded_pairs = base.select_analysis_records(records)
        validation["counts"]["analysis_records"] = len(analysis_records)
        validation["counts"]["excluded_quality_pairs"] = len(excluded_pairs)
        validation["excluded_quality_pair_ids"] = excluded_pairs
        base.write_json(args.output_dir / "validation-report.json", validation)
        models, primary, model_artifacts, predictions = base.run_models(
            analysis_records, config
        )
        status = (
            "held_out_result_candidate_requires_manual_audit"
            if all(
                record["claim_eligible"]
                for record in analysis_records
                if record["split"] == "test"
            )
            else "exploratory_real_data_not_claim_eligible"
        )
        report = {
            "schema_version": "context-measured-model-report-v1",
            "created_at": base.utc_now(),
            "status": status,
            "study_id": config["study_id"],
            "research_question": config["research_question"],
            "data_validation": validation,
            "models": models,
            "primary_comparison": primary["primary_comparison"],
            "scientific_guards": config["scientific_guards"],
            "input_identity": {"path": str(args.input), "sha256": base.sha256_file(args.input)},
            "config_identity": {"path": str(args.config), "sha256": base.sha256_file(args.config)},
            "analysis_amendment": "context_research/protocol/dual-context-v5-analysis-amendment-1.md",
        }
        base.write_json(args.output_dir / "model-report.json", report)
        base.write_json(
            args.output_dir / "model-artifacts.json",
            {
                "schema_version": "context-measured-model-artifacts-v1",
                "created_at": base.utc_now(),
                "status": status,
                "models": model_artifacts,
            },
        )
        base.write_predictions(args.output_dir / "predictions.csv", predictions)
        base.write_text(args.output_dir / "results.md", base.render_markdown(report))
        output_names = [
            "validation-report.json",
            "model-report.json",
            "model-artifacts.json",
            "predictions.csv",
            "results.md",
        ]
        run_manifest = {
            "schema_version": "context-measured-run-manifest-v1",
            "created_at": base.utc_now(),
            "status": status,
            "inputs": {
                str(args.config): base.sha256_file(args.config),
                str(args.input): base.sha256_file(args.input),
                str(Path(__file__)): base.sha256_file(Path(__file__)),
                str(Path(base.__file__)): base.sha256_file(Path(base.__file__)),
            },
            "outputs": {
                name: base.sha256_file(args.output_dir / name) for name in output_names
            },
        }
        base.write_json(args.output_dir / "run-manifest.json", run_manifest)
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
    except (base.StudyError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

