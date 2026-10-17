# Composition proof

This document is the gate between sibling operator proofs and the integrated executor.

## Stage 1 — DirectK

For locked DirectK semantics, packed K is consumed to produce the same decoded-quantized logits `Z` as `Q @ Khat^T` without materializing dense `Khat`.

This stage must be imported by exact upstream commit SHA and revalidated locally.

## Stage 2 — scale and mask

Attention scaling and mask/active-token restriction are applied in the same semantic order as the frozen reference. No upstream representation change may silently move across this edge without an explicit proof.

## Stage 3 — online softmax

For a token block with logits `z_i`, maintain

```text
m = max_i z_i
l = sum_i exp(z_i - m)
o = sum_i exp(z_i - m) * v_i
```

For two block states `(m1,l1,o1)` and `(m2,l2,o2)`, define

```text
m = max(m1,m2)
l = exp(m1-m)*l1 + exp(m2-m)*l2
o = exp(m1-m)*o1 + exp(m2-m)*o2
```

Then the merged state represents the same real-valued softmax numerator/denominator over the union of both token blocks. Final output is `o/l`.

This identity is exact over the real-valued algebraic model; floating-point merge order is covered separately.

## Stage 4 — DirectV

The block output contribution `o` must consume packed V through the locked DirectV exact identity rather than requiring dense `Vhat`.

The central open proof obligation is to write the DirectV coefficient form directly inside the online-softmax numerator state and show equality to the decoded-quantized block numerator.

## Composition criterion

The integrated path is accepted only when every local stage either:

- returns the same state expected by the next original operator; or
- supplies a transformed state with an exact transformed next map such that the local diagram commutes.

No stage may rely on an approximation hidden inside a neighboring stage.

## Current status

- DirectK theorem: upstream candidate, not yet SHA-locked here.
- Online-softmax merge identity: written at formula level.
- DirectV theorem: upstream candidate, not yet SHA-locked here.
- Full packed-K -> streaming softmax -> packed-V composition: **OPEN** until upstream locks and local reference tests exist.
