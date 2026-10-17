# Reproducibility

Every integration result must be reconstructible from a frozen dependency set.

Required metadata:

- DirectAttention commit SHA;
- DirectK upstream SHA;
- DirectV upstream SHA;
- SignedInt4 upstream SHA when used;
- model/checkpoint revision;
- exact config and command;
- hardware model/count;
- driver/CUDA/PyTorch/Triton/compiler versions;
- dtype and accumulation policy;
- context/batch/GQA geometry;
- warmup and repetition protocol;
- raw result path;
- analysis script/path and SHA.

## Result layout

```text
results/<experiment_id>/<timestamp>/
  manifest.json
  upstream_lock.json
  raw/
  derived/
  summary.md
```

No performance number is considered durable if it cannot be tied to an upstream lock and raw measurement artifact.