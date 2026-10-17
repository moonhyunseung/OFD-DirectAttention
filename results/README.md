# Results

Only manifest-backed, upstream-locked results belong here.

Recommended structure:

```text
results/<experiment_id>/<timestamp>/
  manifest.json
  upstream_lock.json
  raw/
  derived/
  summary.md
```

Every summary separates:

- directly measured values;
- derived quantities;
- stage attribution;
- interpretation/hypotheses;
- claim statuses affected;
- negative/crossover regions.

A result without an upstream lock cannot support an integration claim.