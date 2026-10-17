# Semantic contract

## Reference operator

The exact target is decoded-quantized attention, not the original pre-quantized FP operator.

For frozen decoded states `Khat=D_K(C_K,S_K)` and `Vhat=D_V(C_V,S_V)`:

```text
Z = alpha * Q @ Khat^T
Z_masked = mask(Z)
P = softmax(Z_masked)
Y = P @ Vhat
```

The precise mask/active-token convention must be stated by each integration target. The operator order is part of the semantics.

## DirectAttention requirement

A direct executor must return the same real-valued `Y` as the reference operator for every admissible packed state under the locked upstream representation contracts.

It may change intermediate representation or evaluation order only through locally exact identities, exact representation bijections, or exact online-reduction identities.

## Materialization is not semantics

The reference definition does not require these to be materialized as full tensors:

- `Khat`;
- `Vhat`;
- `Z`;
- `P`.

Materialization choices are implementation details provided the final operator semantics are preserved.

## Numerical layer

Algebraic equality over the reals and bitwise IEEE equality are distinct. The repository defines a separate numerical contract for floating-point reassociation, exponentials, and reduction order.