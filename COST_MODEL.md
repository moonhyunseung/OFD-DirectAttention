# DirectAttention symbolic cost model

This file tracks logical intermediates and work categories only. It does not claim latency.

## Dense decoded baseline

A conventional decoded-quantized reference may logically materialize:

```text
Khat: L*d
Vhat: L*d
Z: r*L
P: r*L
```

and then perform score and value contractions.

## Direct streaming target

The representation-native path aims to replace full materialization with block-local state:

```text
packed K block
-> partial logits
-> online-softmax state (m,l,o)
-> packed V block contribution
```

Potentially eliminated full intermediates:

- dense `Khat`;
- dense `Vhat`;
- full `Z`;
- full `P`.

Whether all four disappear in a concrete implementation is a claim to verify, not an assumption.

## Work that remains

DirectAttention does not algebraically eliminate the fundamental K score contraction or V token-coordinate contraction. It reorganizes representation work, intermediate state, and reduction order.

## Cost categories to separate

1. packed load/unpack/recode;
2. K contraction;
3. scale/mask;
4. max/exp/sum softmax state;
5. V coefficient/code contraction;
6. state rescaling/merge;
7. final normalization;
8. synchronization/register/shared-memory traffic.

## Fusion accounting

A fused kernel is useful only if removed intermediate traffic/work exceeds added state-management, recoding, register, or occupancy costs. The crossover is empirical after exact composition and machine lowering are verified.