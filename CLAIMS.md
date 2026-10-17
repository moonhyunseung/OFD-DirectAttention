# Claims

| ID | Claim | Type | Status | Evidence required |
|---|---|---|---|---|
| A-C0 | DirectK semantics are frozen and imported by commit SHA. | Dependency | OPEN | Locked upstream SHA + local compatibility check |
| A-C1 | DirectV semantics are frozen and imported by commit SHA. | Dependency | OPEN | Locked upstream SHA + local compatibility check |
| A-C2 | Token-block online softmax admits an exact associative merge state for the declared masked/scaled logits. | Algebraic | OPEN | Explicit derivation and reference test |
| A-C3 | DirectK, online softmax, and DirectV compose to the same decoded-quantized attention operator. | Composition | OPEN | Commuting/local composition proof |
| A-C4 | Dense decoded K/V materialization can be removed from the exact attention path. | System algebra | OPEN | Composition proof + reference implementation |
| A-C5 | Full logits and/or full probability materialization can be avoided with exact streaming state. | System algebra | OPEN | Streaming derivation + reference implementation |
| A-C6 | A packed streaming attention kernel can realize the exact composition without hidden dense reconstruction. | Implementation | OPEN | Generated-code/path verification |
| A-C7 | The representation-native executor outperforms the frozen decoded baseline in a measurable workload regime. | Hardware/empirical | OPEN | Boundary experiment |
| A-C8 | The gain transfers to full-model decode under matched semantics. | System/empirical | OPEN | Full-model paired measurement |
| A-C9 | The resulting evidence supports a broader OFD principle: downstream operators need not reconstruct representation details they do not mathematically require. | Research synthesis | OPEN | Closed lower claims + scoped argument |

## Claim discipline

A-C9 is not allowed until A-C3 through A-C8 are resolved. A failed fusion or speed claim does not invalidate the upstream algebraic identities; it narrows the systems claim.
