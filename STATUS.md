# Status

## Current phase

**T0 — Composition contract design**

No fused kernel or performance claim exists yet.

| Stage | Requirement | Status |
|---|---|---|
| T0 Upstream lock | DirectK/DirectV semantic contracts identified | OPEN |
| T1 Composition proof | DirectK -> online softmax -> DirectV exact composition written | NOT STARTED |
| T2 Reference | Dense decoded-quantized and streaming direct references agree under declared semantics | NOT STARTED |
| T3 Numerical | Floating-point contract fixed | NOT STARTED |
| T4 Stage lowering | DirectK/softmax/DirectV native paths verified independently | NOT STARTED |
| T5 Fusion | Representation-native fused/streaming executor implemented | NOT STARTED |
| T6 Boundary | Context/batch/fusion crossover mapped | NOT STARTED |
| T7 Model | Full-model decode integration | NOT STARTED |
| T8 Freeze | Claims/results/manifests/upstream SHAs frozen | NOT STARTED |

## Immediate blockers

1. Lock exact upstream commits rather than tracking moving `main` branches.
2. Specify mask/scale/softmax order and numerical semantics explicitly.
3. Prove local composition before any fused implementation.
4. Define the minimal online-softmax state that can consume DirectV without materializing full P.
