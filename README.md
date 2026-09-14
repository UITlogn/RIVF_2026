# RIVF 2026 reviewer artifact

## Status and scope

This is a prepared, non-public reviewer package for the paper *Measuring
Context Beyond the Command*. At submission it has no public DOI, repository
URL, or release license. The package verifies analytical claims from cleaned
feature rows and exported per-record scores. It does not reproduce VM capture
from scratch because unsanitized raw traces, access bundles, and credentials
are excluded pending privacy review.

The root `SHA256SUMS.txt` covers every packaged file except itself.

## Primary analytical inputs

- `data/complete-pairs-measured-claim-v2.jsonl`: 144 cleaned v5 feature rows
  in 72 matched pairs; this is not a raw DRAKVUF event stream.
- `scores/v5-logistic-predictions.csv`: per-record logistic scores, thresholds,
  and alerts for every frozen feature level and the timing sensitivity.
- `scores/v6-locked-predictions.csv`: locked v6 scores.
- `scores/random-forest-predictions.csv`: post-hoc Random Forest scores.
- `evidence/heldout-scores.csv`: compact joined held-out score table.

## Directly reported evidence

- `evidence/ablation-table.csv`: three frozen feature levels plus the
  audit-motivated timing sensitivity.
- `evidence/exact-collision-pairs.csv`: all eight exact Level-2 collision
  pairs with launcher, W/H/I/T indicators, counts, score, and alert.
- `evidence/compromised-admin-pair-audit.csv`: all 12 direct
  compromised-administrator pairs, including four noncollisions.
- `evidence/roc-points.csv`: tie-grouped ROC points reconstructed from scores.
- `evidence/paper-evidence-report.json`: AUC cross-checks, threshold rule,
  collision boundary, environment versions, and input hashes.
- `evidence/model-diagnostics.png`: paper ROC generated from those points.

## Reproduce the paper evidence

Use Python 3.13.3, NumPy 2.2.6, scikit-learn 1.9.1, and Matplotlib 3.11.2:

```powershell
python -B scripts/build_paper_evidence.py `
  --v5-dataset data/complete-pairs-measured-claim-v2.jsonl `
  --logistic-predictions scores/v5-logistic-predictions.csv `
  --rf-predictions scores/random-forest-predictions.csv `
  --model-report reports/v5-model-report.json `
  --output-dir reproduced-evidence
```

The script refuses to overwrite an output directory. Compare generated files
with `evidence/`; the JSON creation timestamp is expected to differ. The
primary logistic settings are in `config/experiment-v2-measured.json`, and
`scripts/run_context_study.py` implements training and threshold selection.

## ROC handling

Unique scores are processed in descending order. Records sharing a score enter
one ROC point, so ties are never arbitrarily ordered. Adjacent grouped points
are connected linearly. Trapezoidal AUC must equal the Mann--Whitney definition
(0.5 credit for positive-negative ties) and scikit-learn's cross-check.

## Experiment chronology

1. The v5 analysis plan froze the 72-pair manifest, group-disjoint split, and
   seed before analysis.
2. Recovery of one interrupted host-side archive exposed that run's validated
   process sequence and timing, but no model score, threshold, feature-label
   aggregate, or test metric.
3. A hash-frozen amendment excluded timing from the primary representation
   before model fitting and test scoring; timing remained a sensitivity.
4. Development fit the model, validation selected the threshold, and the full
   held-out test was scored without adjustment.
5. After v5 results, v6 was frozen before execution and used byte-identical
   model artifacts. It repeats the same VM/template family.
6. Random Forest and operation-mapping controls were added after test access
   and are post-hoc.

## Capture environment

- Guest: Windows 10 22H2 x64, build 10.0.19045.3803, 2 vCPU, 4096 MiB.
- Hypervisor: Xen 4.19.2 on Debian 12 / Linux 6.12.95+deb12-amd64.
- VMI: DRAKVUF 1.1-da982ab, LibVMI 0:15:0; Python 3.11.2.
- Capture: one offline reverted VM and one profile for v5/v6.

Not every hardware field was snapshotted into every run archive. No GPU was
used.

## Interpretation boundaries

- V5/v6 are self-collected controlled attempts, not a public malware or
  enterprise dataset.
- Labels are scripted; targets are nonexistent and no malware/effect is used.
- Observed FPR 0/24 has Wilson 95% upper bound 13.8%.
- V6 is same-template temporal repeatability, not external validation.
- Eight collisions bound pointwise models using the exact Level-2 projection
  on this test set; they do not exhaust information in raw telemetry.
- Included lineage reports record internal raw-checksum audits. Without the raw
  traces, they cannot independently prove the raw bytes to a reviewer.
