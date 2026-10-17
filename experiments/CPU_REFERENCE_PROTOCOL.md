# CPU attention reference profile v1

Run `python experiments/verify_reference.py` from a clean committed tree. CPU only, one thread, seed1729+L. r6,d256,m32; L0,1,17,129,512; FP32-generated Q cast to explicit FP64/FP32 compute; FP16 stored scales. All casts and operator ordering are visible in reference/executor.py and the locked upstream modules. PyTorch selects CPU reduction/FMA order; no bitwise equivalence or FP16 accumulation claim. Alpha1/16 is applied once before mask/softmax. No rotation or model approximation.

Dense baseline decodes the unchanged packed state. Staged direct materializes full logits/weights. Streaming direct uses blocks1,16,31, with no full-length logits/weights or dense scaled K/V. Cases include no mask, partial masks with an all-masked row, empty context, boundary fragments and Q scaled10000 to challenge softmax stability. Empty/all-masked output0 is a declared extension, not IEEE softmax equality.

Before execution, FP64 reference regression tests freeze rtol=atol=1e-12 on listed fixtures, including alternate merge order and neutral state. Numerical profile reports max absolute and relative L2 errors against dense, and requires finite outputs. FP32 characterization has no posthoc accuracy gate. These are reference test criteria; production T2 remains open pending an intended production accumulation/exp policy and its accuracy budget. Algebraic composition is proved separately, not inferred from numerical closeness. No GPU speed, traffic, register or serving claims follow.

Exact upstream source commits and any import-only adaptation hashes are recorded in manifests/upstream_lock.json. The clean source SHA, full environment, raw CSV and test log accompany every formal run. The signed codec is not used in this offset-only integration.
