# RIVF dual-context v5: frozen exploratory plan

Frozen on: 2026-09-12 (Asia/Saigon)

## Frozen inventory

- Manifest: `context_research/config/dual-context-campaign-v5.json`
- Manifest SHA-256:
  `3fd9481b6fa5bb2226456fe5b0935712709eecde84755303f8c2f5600b409bc8`
- Capture profile:
  `sha256:9af9799d9b35ef0cc6b47008761638b47f8db91a91847fc3c8a15aefa11b7e2a`
- 72 matched pairs / 144 records; 24 pairs per split and six pairs for
  every operation within each split.
- Pair and side order remain deterministic under seed `20260911`.

## Capture scope and rationale

- DRAKVUF procmon uses context views for both `powershell.exe` and `cmd.exe`.
- These are the two frozen parent contexts that invoke all discovery and target
  processes in the scenario inventory. Direct operations are observed from
  PowerShell; batch targets are observed from cmd.
- This captures the complete relevant two-level scenario chain without hooking
  unrelated system processes. It must not be described as arbitrary-depth,
  whole-system descendant capture.
- The v5 smoke pair observed all four discovery processes, the cmd launcher and
  `wbadmin.exe` target with matching parent PID. It emitted 884 procmon records
  on attack and 684 on legitimate, versus 8,550--16,556 under unstable
  whole-system capture.

## Balance and quality gates

- Labels have identical distributions of 0, 2 and 4 observed discovery images.
- Every split contains 12 batch attack-preparation and 12 direct
  compromised-admin cases.
- `account_state_label` is audit-only and forbidden as a model feature.
- Launcher type is inferred from the observed target parent.
- Every run must contain each guest-recorded discovery image before the target,
  plus the operation-specific target image and command token.
- Batch attack-preparation additionally requires `cmd.exe -> target.exe` with
  correct event order.
- The VM is offline, reverted after each run and the campaign stops on first
  failure. Failed archives are preserved and never enter complete pairs.

## Analysis policy

- Do not run or inspect test analysis before all 144 records pass ingestion and
  the dataset passes schema audit.
- Train on development, select the threshold on validation at FPR <= 0.05, then
  apply it once to test without adjustment.
- Report operation-only, actor-context and full-context TPR, FPR, MCC, ROC AUC,
  pair ranking and the compromised-admin audit.
- Twenty-four legitimate records in validation and test give an empirical FPR
  step of 1/24, approximately 0.0417.

## Claim boundary

V5 is exploratory command-attempt evidence against guarded nonexistent targets
from a measured non-elevated account. It does not support claims about
successful destructive operations or population ransomware detection and is
not pooled with earlier capture profiles.

## Why v5 replaces v4 whole-system capture

Whole-system procmon v4 intermittently ended after only 21--54 seconds and left
the guest non-injectable, consistent with DRAKVUF's `-b` early-BSoD path. V5
reduces hooks to the two scenario-relevant parent contexts while preserving the
previously missing `cmd -> target` edge. No model or test metric was inspected
or used to make this engineering correction.
