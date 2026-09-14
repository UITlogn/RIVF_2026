# RIVF temporal replication v6: frozen locked-model plan

Frozen after the exploratory v5 held-out analysis on 2026-09-13 and before any
v6 VM execution.

## Purpose

V6 is a test-only temporal replication of the v5 controlled scenario study.
It asks whether the already-frozen v5 `measured_process_context` model produces
the same directional result on newly executed VM runs. Model weights, encoder,
numeric scaling, and the validation-selected threshold are copied byte-for-byte
from the finalized v5 artifact and cannot be retrained or reselected on v6.

This is a same-VM, same-snapshot, same-template replication. It is not external
validation and does not address organization, host, operator, command-family,
or natural-enterprise distribution shift.

## Frozen inputs and design

- Campaign manifest: `context_research/config/temporal-replication-v6.json`
- Campaign manifest SHA-256:
  `9192e5729f837b60a02033144d881c69c27fd139994b8e36905772e803e11250`
- V5 locked model artifact:
  `output/context-study/runs/rivf-context-dual-context-v5-measured-final/model-artifacts.json`
- Locked model artifact SHA-256:
  `b7c198bd01113e3226b3efbb43535199beff654928c5b8f38153fa152fd58cd7`
- Candidate: `measured_process_context`, threshold `0.8722271137626169`.
- Baseline: `operation_only`, threshold `0.500000000001`.
- Capture profile:
  `sha256:9af9799d9b35ef0cc6b47008761638b47f8db91a91847fc3c8a15aefa11b7e2a`.
- Randomization seed: `20260913`.
- Size: 24 matched pairs / 48 new VM runs, all designated replication test.
- Four operations, six pairs each.
- Twelve batch attack-preparation and twelve direct compromised-admin pairs.
- Twelve legitimate-first and twelve attack-first pairs.
- Templates are the 24 frozen v5 test scenarios, re-executed under new pair,
  workload, and run identifiers in a newly randomized order.

## Access and analysis policy

- Do not inspect v6 aggregate feature/label associations, scores, thresholds, or
  results until all 48 records are collected and the final schema, lineage,
  archive, and per-file checksum audits pass.
- Do not train, fit, calibrate, or select any parameter on v6.
- Apply the locked candidate and baseline to every accepted record exactly once.
- Report TPR, FPR, MCC, AUC, pair ranking, operation strata, compromised-admin
  detection, Wilson intervals, exact paired discordance, and matched-pair
  bootstrap intervals.
- Preserve failures separately. A quality or infrastructure failure is an
  exclusion/engineering record, never a classifier false negative.

## Interpretation gates

The primary replication check is directional: locked candidate delta-TPR must
remain positive without increasing the point-estimate FPR above 0.05. Report the
exact result even if this check fails. No v6-driven model change may be evaluated
on v6; a changed model requires a newly frozen dataset.

Because scenario templates deterministically control several measured process
features, v6 mainly tests capture, derivation, and locked-scoring repeatability.
It does not resolve the v5 observability ceiling or establish population-level
performance.

## Safety and containment

- No malware is executed.
- The Windows VM remains offline and disposable.
- Commands target guard-confirmed nonexistent resources and measure attempts,
  not successful destructive effects.
- The VM is reverted after every run; network isolation and final containment
  evidence are mandatory.
- The campaign remains stop-on-first-genuine-failure. A completed pass archive
  may be recovered after a host/SSH interruption only if independent ingestion,
  checksum, outcome, sequence, revert, and containment gates all pass.
