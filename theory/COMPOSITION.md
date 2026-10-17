# Composition of exact local rewrites

Let Khat=S_K(C_K-8) and Vhat=S_V(C_V-8) denote the unchanged decoded-quantized tensors. DirectK is the distributive rewrite of Q Khat^T. Apply alpha exactly once to those scores, then a boolean mask. Each unmasked score is unchanged in real arithmetic. For a block b define m_b=max z, w_b=exp(z-m_b), l_b=sum w_b and o_b=sum w_b Vhat. DirectV's coefficient-prescaled form computes exactly that o_b by distributivity. Thus no rotation, token selection, approximation or requantization enters any local stage.

For disjoint blocks a,b choose m=max(m_a,m_b). Then l=exp(m_a-m)l_a+exp(m_b-m)l_b and o=exp(m_a-m)o_a+exp(m_b-m)o_b. Substituting their definitions gives l=sum exp(z-m) and o=sum exp(z-m)Vhat over the union. Induction gives final o/l equal to dense decoded-quantized attention for a nonempty unmasked row. Merge is associative/commutative over real arithmetic because it summarizes the same union; floating-point order can differ.

We explicitly extend the operator to empty/all-masked rows with output0. Its neutral state is (-infinity,0,0). Implementations avoid -infinity-(-infinity) by using a safe zero origin when both states are empty; their weights are0. This is a declared extension, not an assertion that IEEE softmax(all -infinity) is defined.

The block reference materializes block logits, block weights and logical code tensors, not full-length logits/probabilities or dense scaled K/V. It is not a fused GPU kernel and has no traffic/instruction/speed claim. Exact upstream files, import-only adaptations and source hashes are locked in manifests/upstream_lock.json.
