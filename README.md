# RIVF 2026 research artifact

Code, cleaned data, results and figures supporting *Measuring Context Beyond the Command: A Matched-Pair Study of Dual-Use Windows Administrative Actions*. Updated to match the final revised study on 17 November 2026.

This repository contains the analytical research artifact. It excludes manuscript LaTeX/PDF, editorial reports, reviewer responses, credentials, malware binaries and VM images. No new experiment captures were added in the revision: the newly packaged v6 data are the existing 48 historical repeat captures.

## What is included

| Path | Purpose |
|---|---|
| `data/complete-pairs-measured-claim-v2.jsonl` | 144 cleaned v5 feature records, 72 matched pairs |
| `data/complete-pairs-temporal-replication-v6.jsonl` | 48 cleaned v6 records, 24 repeated template pairs |
| `schema/context-record-v2.schema.json` | Feature-record contract |
| `config/`, `protocol/` | Frozen experiment settings and historical analysis plans |
| `scores/`, `reports/` | Original predictions, frozen fitted model, uncertainty and lineage evidence |
| `evidence/revision/` | Current overlap, design, collision, ablation and verification outputs |
| `evidence/process-chain-comparison.png` | Final schematic: 12 batch and 12 direct pairs; 8 direct collisions and 4 discovery differences |
| `evidence/model-diagnostics.png` | Final ROC, labeled as finite-design results with RF post-hoc |
| `scripts/` | Fitting, independent numerical checks and figure reconstruction |

Existing v5 data, scores and frozen models are reused rather than duplicated. The current figures supersede the earlier presentation. Root-level evidence files with names such as `heldout-scores.csv` and operation-mapping outputs are retained historical analyses; their filenames are not claims of generalization or a representative ATT&CK detector comparison.

## Reproduce the current numerical claims

Use Python 3.11 or later. The following audit/refit path uses only the standard library. Run from the repository root:

```text
python -B scripts/audit_revision.py
python -B scripts/verify_revision_figures.py
```

The first script recomputes joint feature-label overlap, six authored design patterns, all eight exact collision pairs, L0/L1/L2 counts, AUC, within-pair ranking, Wilson intervals and the explicitly post-hoc vector-lookup diagnostic. The second reconstructs every ROC coordinate from original scores and checks the schematic counts. Outputs are written to `evidence/revision/`.

To refit the historical logistic models:

```text
python -B scripts/run_measured_context_study.py --config config/experiment-v2-measured.json --input data/complete-pairs-measured-claim-v2.jsonl --output-dir .reproduced/refit
```

The output directory must not exist. Compare `.reproduced/refit/predictions.csv` byte for byte with `scores/v5-logistic-predictions.csv`, and the `models` member of `.reproduced/refit/model-artifacts.json` with `reports/v5-model-artifacts.json`. Timestamps and source paths may differ. Use the measured-v2 wrapper, not the v1 base-engine entry point. The engine's historical status wording does not establish held-out generalization.

The historical v6 score export contains L0 and L2. The current audit also evaluates the frozen L1 model on cleaned v6 rows; it does not claim an additional historical score export or a new collection.

## Rebuild the two final figures

```text
python -m pip install -r requirements-figures.txt
python -B scripts/audit_revision.py
python -B scripts/verify_revision_figures.py
python -B scripts/render_revision_roc.py
python -B scripts/render_revision_schematic.py
```

The figure scripts use Matplotlib 3.11.2. Matplotlib is not needed for the numerical audit or deterministic logistic refit. Historical `build_paper_evidence.py` and RF-fitting utilities additionally use NumPy/scikit-learn and reproduce the earlier exploratory presentation; the commands above are the route to the final figures.

## Interpretation

- Each historical partition has 48 records, 24 pairs and 28 distinct L2 vectors. Development, validation, test and v6 have identical joint feature-label multisets.
- L2 gives 16/24 TP and 0/24 FP in this finite scripted design; these are not deployment estimates. Wilson 95% intervals are about 46.7–82.0% and 0.0–13.8%, respectively.
- Eight opposite-label pairs have identical L2 vectors, bounding deterministic pointwise pair ranking at 83.3% on this sample/projection only.
- Launcher gain reflects scenario authoring. Direct quiet/full pairs collide; four direct partial pairs differ in discovery indicators.
- V6 repeats the same VM/template family and supports repeatability, not external validity. More events or repeat captures do not increase independent scenario diversity.
- Target absence was intended. The audited context runner checks service/log absence; an equivalent explicit drive-absence guard was not found. Successful destructive effects were not the study endpoint.

Capture used Windows 10 22H2 build 19045.3803 (2 vCPU, 4 GiB), Xen 4.19.2 on Debian 12/Linux 6.12.95, DRAKVUF 1.1-da982ab and LibVMI 0:15:0. The package reproduces feature/score arithmetic, not VM capture. Without unsanitized raw traces, historical raw-lineage reports cannot independently authenticate their underlying bytes. Historical freeze manifests retain original workspace paths and are provenance records, not a self-contained capture environment.

## Integrity

`MANIFEST.json` inventories the research artifact; `SHA256SUMS.txt` hashes all distributed files except itself. `.gitattributes` preserves exact bytes across checkouts, including the original CSV line endings. This avoids platform-dependent checksum failures. `scripts/verify_artifact.py` checks file coverage and hashes:

```text
python -B scripts/verify_artifact.py
```

After regenerating evidence, timestamp/interpreter metadata may change; rerun verification against a clean checkout when checking the published bytes. No new release license or DOI is asserted by this update.
