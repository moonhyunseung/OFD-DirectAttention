# Analysis

System-level analysis must preserve stage attribution and upstream provenance.

Required practice:

- separate DirectK, softmax, DirectV, fusion, and integration costs;
- derive results from versioned raw outputs and upstream locks;
- report end-to-end and stagewise crossover regions, including regressions;
- distinguish semantic/materialization claims from machine-performance claims;
- never infer universal behavior from one context/batch point;
- retain failed and negative runs that constrain the system claim.

Any figure/table used for a durable claim should be reproducible from a recorded manifest and analysis script.