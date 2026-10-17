# Theory workspace

This directory contains exact composition mathematics before implementation.

Planned topics:

- DirectK interface contract;
- mask/scale ordering;
- online-softmax merge algebra;
- DirectV streaming interface;
- composition proof;
- materialization-elimination arguments;
- lower bounds and non-goals.

No hardware shortcut may alter the operator order silently. Every local rewrite must expose its source/target state and compatibility with the next stage.