#!/usr/bin/env python3
"""Audit exact opposite-label feature collisions in the frozen v5 dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not an object: {path}")
        values.append(value)
    return values


def get_field(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"{record.get('record_id')} lacks feature {dotted}")
        value = value[part]
    return value


def feature_vector(record: dict[str, Any], specification: dict[str, Any]) -> tuple[Any, ...]:
    fields = [*specification.get("categorical", []), *specification.get("numeric", [])]
    return tuple(get_field(record, field) for field in fields)


def audit_model(records: list[dict[str, Any]], specification: dict[str, Any]) -> dict[str, Any]:
    by_split_pair: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_split_pair[str(record["split"])][str(record["pair_id"])].append(record)
    split_reports: dict[str, Any] = {}
    for split, pairs in sorted(by_split_pair.items()):
        identical = distinguishable = 0
        identical_by_attack_state: dict[str, int] = defaultdict(int)
        for pair_id, pair in pairs.items():
            if len(pair) != 2 or {row["label"] for row in pair} != {
                "legitimate_admin",
                "attack_preparation",
            }:
                raise ValueError(f"{split}/{pair_id} is not a complete matched pair")
            legitimate = next(row for row in pair if row["label"] == "legitimate_admin")
            attack = next(row for row in pair if row["label"] == "attack_preparation")
            if feature_vector(legitimate, specification) == feature_vector(attack, specification):
                identical += 1
                identical_by_attack_state[str(attack["account_state_label"])] += 1
            else:
                distinguishable += 1

        collision_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record["split"] == split:
                collision_groups[feature_vector(record, specification)].append(record)
        mixed_groups = [
            group for group in collision_groups.values() if len({row["label"] for row in group}) > 1
        ]
        mixed_record_ids = {row["record_id"] for group in mixed_groups for row in group}
        total_pairs = len(pairs)
        split_reports[split] = {
            "pairs": total_pairs,
            "within_pair_identical_feature_vectors": identical,
            "within_pair_distinguishable_feature_vectors": distinguishable,
            "identical_by_attack_account_state": dict(sorted(identical_by_attack_state.items())),
            "maximum_pair_ranking_if_all_distinguishable_pairs_are_oriented_correctly": (
                (distinguishable + 0.5 * identical) / total_pairs
            ),
            "cross_label_collision_groups": len(mixed_groups),
            "records_in_cross_label_collision_groups": len(mixed_record_ids),
        }
    return {"features": specification, "by_split": split_reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite output: {args.output}")
        records = load_jsonl(args.input)
        config = load_json(args.config)
        reports = {
            name: audit_model(records, specification)
            for name, specification in config["feature_sets"].items()
        }
        output = {
            "schema_version": "rivf-v5-observability-audit-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "records": len(records),
            "models": reports,
            "interpretation": (
                "Within-pair identical vectors cannot receive different scores from a deterministic "
                "pointwise model using only the listed features. This is an observability bound, not "
                "an estimate of external-population error."
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps({"status": "complete", "records": len(records), "output": str(args.output)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

