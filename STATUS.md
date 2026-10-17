# Status

## Current phase

**T2 — Locked streaming reference verified; production numerical policy open**

No fused kernel or performance claim exists yet.

| Stage | Requirement | Status |
|---|---|---|
| T0 Upstream lock | DirectK/DirectV semantic contracts identified | PASS (exact SHA + file locks) |
| T1 Composition proof | DirectK -> online softmax -> DirectV exact composition written | PASS (real algebra) |
| T2 Reference | Dense decoded-quantized and streaming direct references agree under declared semantics | PASS (CPU reference fixtures) |
| T3 Numerical | Floating-point contract fixed | IN PROGRESS (production policy open) |
| T4 Stage lowering | DirectK/softmax/DirectV native paths verified independently | NOT STARTED |
| T5 Fusion | Representation-native fused/streaming executor implemented | NOT STARTED |
| T6 Boundary | Context/batch/fusion crossover mapped | NOT STARTED |
| T7 Model | Full-model decode integration | NOT STARTED |
| T8 Freeze | Claims/results/manifests/upstream SHAs frozen | NOT STARTED |

## Original bootstrap blockers (see checkpoint below for resolved items)

1. Lock exact upstream commits rather than tracking moving `main` branches.
2. Specify mask/scale/softmax order and numerical semantics explicitly.
3. Prove local composition before any fused implementation.
4. Define the minimal online-softmax state that can consume DirectV without materializing full P.

## Verified CPU reference checkpoint — 2026-10-17

Source `46aabef1b6f93dfa968423a91d50bae158928cf6`; [25-test formal run and raw evidence](results/20261017T045329Z_b9d2ff6b/FINDINGS.md). Exact dependencies are pinned where used. These are CPU reference results only.
