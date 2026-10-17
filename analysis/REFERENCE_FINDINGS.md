# Reference evidence — OFD-DirectAttention

- Source commit: `46aabef1b6f93dfa968423a91d50bae158928cf6`.
- Recorded run: `results/20261017T045329Z_b9d2ff6b`.
- Formal tests: 25 passed; see run.log. The source was committed and clean before execution.
- No GPU or performance measurement was performed. No original OFD E00–E14 result was modified.

## Direct observations

| Compute | Rows | Max absolute difference | Max defined relative L2 | Zero-reference/nonzero-error rows |
|---|---:|---:|---:|---:|
| torch.float64 | 60 | 5.59323637753e-15 | 2.27351843615e-14 | 0 |
| torch.float32 | 60 | 3.09944152832e-06 | 4.75004790013e-06 | 0 |

Largest absolute difference: case `large_logits`, L=17, `torch.float32`, `streaming`. Raw effects are preserved; no thresholds were tuned to them.

All outputs in this profile were finite. Relative L2 is explicitly undefined when the baseline is exactly zero but reassociation produces a nonzero residual; these rows are neither removed nor reported as zero error. Maxima are across different fixtures, not simultaneous guarantees.

## Interpretation

The fixed K/V sources compose into a block-streaming reference under the explicit empty/all-masked output0 extension. Local distributivity and the online merge identity prove the real-valued composition; CPU tests check its implementation. Block-local logical codes/weights are materialized, but no full-length logits/probability tensor or dense scaled K/V is required by the streaming reference.

## Not established

The finite fixture set does not provide a global forward-error guarantee, production softmax/accumulation contract, GPU packed lowering, speedup, or serving result. In particular stable softmax avoids overflow but cannot remove score reassociation sensitivity.

## Next justified work

Close the production numerical policy, verify K, softmax-state merge and V native machine paths separately, then consider fusion. Do not claim whole-model benefit from this CPU reference.
