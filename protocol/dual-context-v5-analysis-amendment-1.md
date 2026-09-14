# RIVF dual-context v5: analysis amendment 1

Recorded during collection on 2026-09-13 (Asia/Saigon), before any v5 model,
threshold selection, or aggregate test analysis.

## Reason

A provenance audit found two construct-validity problems in the original
`full_context` specification:

1. `seconds_since_session_start` is actually elapsed time from scripted guest
   scenario start to target start, not OS logon-session age.
2. `out_of_hours` is derived from the restored guest snapshot clock, not the
   real collection time.

Most other full-context fields are constants or `unknown` under the v5
single-host protocol. Using all of them under the broad name `full_context`
would obscure which evidence is actually measured and variable.

## Frozen interpretation policy

- Preserve and report the original registered analysis; do not replace it after
  seeing results.
- Do not interpret scenario-elapsed time as session age or `out_of_hours` as an
  operational temporal signal.
- Add a measured-only sensitivity model using only operation plus features
  extracted from observed process creation: target launcher and presence of the
  four frozen discovery images (`whoami.exe`, `hostname.exe`, `ipconfig.exe`,
  and `tasklist.exe`).
- Exclude record/pair IDs, split, workload group, label, scenario profile,
  account-state label, PIDs, event UIDs, absolute timestamps, and provenance
  paths/hashes from every model.
- Treat a model including scenario-elapsed time as a separate timing-proxy
  sensitivity, never as the primary context claim.
- Compare every model at its independently selected validation threshold under
  the same FPR <= 0.05 cap, then apply that threshold unchanged to test.
- Report pair ranking, test TPR/FPR/MCC/AUC, exact confusion counts, and separate
  results for direct compromised-administrator and batch attack-preparation
  cases.
- Report constant/unmeasured fields as a limitation instead of interpreting
  their learned coefficients.

## Test-access deviation log

The archive for test run `ctxv5-p047-r01` was inspected solely to recover from a
host-side interruption after remote completion. The operator observed its
protocol-known label/operation, pass status, event count, validated sequence,
containment state, and scenario timing. No v5 model, score, threshold, aggregate
feature-label statistic, or test metric was computed. Because exact timing was
visible during recovery, the measured-only model explicitly excludes timing.

This deviation must be disclosed in the experiment log and, if space permits,
the reproducibility material.

