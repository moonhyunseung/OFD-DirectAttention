# Tests

Required test layers:

1. upstream semantic compatibility against locked DirectK/DirectV SHAs;
2. mask/scale ordering tests;
3. online-softmax block merge tests;
4. DirectV-in-softmax-numerator composition tests;
5. staged direct vs dense decoded reference tests;
6. floating-point numerical-contract tests;
7. optimized stage regression tests;
8. fused-kernel regression tests;
9. model integration regression tests.

Fusion tests are never allowed to substitute for stage-level correctness tests.