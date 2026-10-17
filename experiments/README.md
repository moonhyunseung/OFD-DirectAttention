# Experiments

Experiments are boundary tests for already-proved composition and verified implementations.

Each experiment declares:

1. frozen upstream SHAs;
2. exact theorem/composition already established;
3. one unresolved implementation or hardware assumption;
4. failure condition;
5. boundary variables;
6. minimal measurement;
7. stopping rule;
8. raw result and analysis paths.

Planned series:

- E00 upstream/reference compatibility;
- E01 streaming composition correctness;
- E02 stagewise lowering costs;
- E03 fusion without hidden reconstruction;
- E04 context/batch/GQA boundary grid;
- E05 layer/model integration;
- E06 full-model decode;
- E07 negative/failure boundary freeze.

Forbidden: using latency sweeps to choose algebra, silently changing quantization semantics, or labeling approximate softmax/token dropping as exact.