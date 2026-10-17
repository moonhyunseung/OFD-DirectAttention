# Kernel gate

Optimized or fused kernels enter this directory only after stage semantics, upstream locks, and numerical contracts are frozen.

A candidate must declare:

- locked DirectK/DirectV/SignedInt4 SHAs;
- exact streaming state layout;
- accumulation dtype/order;
- whether any dense K/V/logit/P tensor is reconstructed internally;
- intended machine primitive and tile geometry;
- expected symbolic memory/work reduction;
- failure/crossover hypothesis.

A fused kernel that is faster but cannot be mapped back to the frozen semantic chain is not accepted as evidence.