# Integration gate

Model/framework integration begins only after the standalone composition reference and stage lowerings are frozen.

Every integration target must record:

- framework/runtime and version;
- model/checkpoint revision;
- exact attention geometry;
- locked DirectK/DirectV/SignedInt4 SHAs;
- replaced call path and untouched baseline path;
- numerical comparison protocol;
- end-to-end and stage-level measurement protocol.

Integration must not silently change RoPE, masks, gating, output projection, cache semantics, or quantization policy. Any model-specific operator outside the frozen attention core is documented before replacement.