# Reference implementations

This directory is for transparent, compositional correctness references.

Planned references:

- `dense_decoded_attention.py` — frozen decoded-quantized baseline;
- `directk_stage.py` — locked DirectK reference adapter;
- `online_softmax.py` — exact block-state reference;
- `directv_stage.py` — locked DirectV reference adapter;
- `streaming_attention.py` — staged representation-native reference.

Reference code may be intentionally slow. Its purpose is to make every semantic edge inspectable before fusion or hardware optimization.