# OFD-DirectAttention

**Exact representation-native quantized attention from packed K/V through online softmax, without requiring dense dequantized K/V materialization.**

This is the integration/system repository in the focused OFD research line. It composes frozen upstream contracts from `OFD-DirectK`, `OFD-DirectV`, and optionally `OFD-SignedInt4` into one exact attention executor.

## Target chain

```text
Q + packed K
  -> exact partial logits
  -> mask/scale
  -> exact online-softmax state
  -> packed V accumulation
  -> output Y
```

The repository asks which intermediates are mathematically required. Dense decoded `Khat` and `Vhat` are not assumed to be required. Full logits/probability materialization is also treated as an implementation question subject to exact online-softmax composition.

## Semantic target

The reference semantics are the frozen decoded-quantized attention operator, not the original pre-quantized FP attention.

A DirectAttention implementation must compose locally exact stages so that the final output equals the declared decoded-quantized reference operator over the real-valued algebraic model, with a separately declared floating-point numerical contract.

## Upstream rule

No moving dependency is allowed. Every integration result must lock exact upstream commit SHAs for DirectK, DirectV, and SignedInt4 when used.

## Research order

1. lock upstream semantic contracts;
2. write composition proof;
3. define exact online-softmax streaming state;
4. build reference implementation;
5. fix numerical contract;
6. lower stages independently;
7. fuse only after stage correctness is frozen;
8. map failure/crossover boundaries;
9. integrate into real model decode;
10. freeze claims/results/manifests.

## Non-goals

Approximate softmax, token dropping, block dropping, truncated low-rank replacement, or changed quantization semantics do not belong in the exact path.

## Reproducible reference checkpoint

See [reference findings](analysis/REFERENCE_FINDINGS.md) for the pinned source SHA, raw run, numerical observations and remaining gates. Install the recorded environment and run `python experiments/verify_reference.py` from a clean worktree. No GPU performance claim is made.
