# Architecture

## Exact semantic chain

The repository treats the following order as explicit rather than implicit:

```text
Q, packed K
  -> DirectK score contribution
  -> attention scale
  -> mask / active-token restriction
  -> online softmax state
  -> DirectV accumulation from packed V
  -> final normalized output Y
```

Any model-specific operator outside this attention core must be documented separately before integration.

## Streaming state

For each query row and token block, the intended exact softmax summary is conceptually

```text
m = block max logit
l = sum exp(logit - m)
o = sum exp(logit - m) * decoded-quantized V
```

Two block states may be merged exactly in real arithmetic by rescaling both states to the new maximum. The DirectV research task is to express `o` directly from packed V without first reconstructing dense V.

## Materialization questions

The architecture does not assume that any of these must exist as full tensors:

- dense decoded K;
- dense decoded V;
- full logits;
- full probabilities.

Whether each intermediate can be removed is decided by exact stage composition first, then by hardware lowering.

## Fusion rule

Fusion is not evidence of correctness. A fused kernel is accepted only if it implements the already-frozen stage semantics and numerical contract.
