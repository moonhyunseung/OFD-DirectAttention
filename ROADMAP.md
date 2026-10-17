# Roadmap

## Phase 1 — Dependency freeze

- Select exact DirectK and DirectV commits.
- Select SignedInt4 commit only if signed representation is used.
- Record imported semantic contracts and incompatibilities.

Exit criterion: no integration result depends on a moving upstream branch.

## Phase 2 — Composition proof

Write the complete stage order:

```text
packed K + Q
-> score
-> scale/mask
-> online softmax state
-> packed V accumulation
-> Y
```

Prove each local rewrite and the composition. No fused kernel is allowed before this closes.

## Phase 3 — Reference executor

- decoded-quantized dense baseline;
- staged direct reference;
- streaming direct reference;
- deterministic tests covering masks, lengths, blocks, and edge cases.

## Phase 4 — Numerical contract

Separate real-valued algebraic exactness from floating-point reassociation. Freeze accumulation dtypes, ordering assumptions, tolerances, and baseline semantics.

## Phase 5 — Stage lowering

Verify DirectK, softmax-state merge, and DirectV machine paths independently. Inspect generated code and reject hidden reconstruction.

## Phase 6 — Fusion

Fuse only after individual stage contracts pass. Record register/shared-memory pressure, tile geometry, synchronization, and state layout.

## Phase 7 — Boundary experiments

Map where the exact direct executor wins or loses across context, batch, GQA ratio, tile geometry, and representation choice. Preserve negative regions.

## Phase 8 — Model integration and freeze

Run matched full-model decode, then freeze commits, upstream locks, model revision, hardware/toolchain, raw results, analysis, and final claim statuses.
