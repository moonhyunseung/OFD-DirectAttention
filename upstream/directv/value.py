"""Exact real-algebra references over frozen stored INT4 state.
Logical code tensors are materialized. These are not packed hardware kernels.
"""
import torch
from upstream.directk.quant import dequantize_int4_groupwise, unpack_codes, offset_bytes_to_twos_complement

def _check(p, v):
    if p.ndim != 2 or len(v.orig_shape) != 2 or p.shape[1] != v.orig_shape[0]:
        raise ValueError('expected coefficients [r,L] and packed V originating from [L,d]')

def decode_then_value(p,v,*,compute_dtype=torch.float64):
    _check(p,v)
    return p.to(compute_dtype) @ dequantize_int4_groupwise(v,compute_dtype)

def direct_value(p,v,*,compute_dtype=torch.float64,signed=False):
    _check(p,v)
    if signed:
        raw=offset_bytes_to_twos_complement(v.packed)
        n=torch.stack((raw&15,(raw>>4)&15),-1).reshape(*raw.shape[:-1],v.group_size).to(torch.int16)
        c=torch.where(n>=8,n-16,n).to(compute_dtype)
    else:
        c=unpack_codes(v).to(compute_dtype)
    w=p.to(compute_dtype)[:,:,None]*v.scale.to(compute_dtype)[None,:,:]
    y=torch.einsum('htg,tgj->hgj',w,c)
    if not signed:
        y=y-8*w.sum(dim=1)[:,:,None]
    return y.reshape(p.shape[0],v.orig_shape[1])
