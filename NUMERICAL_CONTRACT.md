# Numerical contract

## Exactness levels

`ALGEBRAIC-EXACT` means the integrated operator is equal to the frozen decoded-quantized attention operator over real arithmetic.

`BITWISE-EXACT` would require identical IEEE evaluation order, reduction order, exponential implementation, and casts. It is not assumed by default.

## Numerical sources of divergence

Even when the algebra is exact, floating-point outputs may differ because of:

- K contraction reassociation;
- scale/correction order;
- blockwise max/exp rescaling;
- online-softmax merge order;
- V reduction order;
- accumulation dtype and cast points.

## Reference hierarchy

1. high-precision decoded-quantized reference for diagnosis;
2. frozen production-like decoded baseline;
3. staged direct reference;
4. optimized direct/fused implementation.

The acceptance contract must be declared before performance tuning.

## Reporting

Numerical reports should include max/mean absolute error, a scale-aware relative metric that is not dominated by near-zero outputs, and any downstream model-level check required by the integration target.

Numerical closeness is never used as proof of the algebraic composition.