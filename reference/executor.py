"""Dense, staged and block-streaming CPU references, not fused GPU kernels."""
from dataclasses import dataclass
import torch
from upstream.directk.quant import PackedInt4,dequantize_int4_groupwise
from upstream.directk.score import direct_packed_score
from upstream.directv.value import direct_value

@dataclass(frozen=True)
class State:
    maximum: torch.Tensor
    denominator: torch.Tensor
    numerator: torch.Tensor

def empty_state(rows,dim,*,dtype,device):
    return State(torch.full((rows,),-torch.inf,dtype=dtype,device=device),torch.zeros(rows,dtype=dtype,device=device),torch.zeros(rows,dim,dtype=dtype,device=device))

def merge(a,b):
    m=torch.maximum(a.maximum,b.maximum)
    safe=torch.where(torch.isfinite(m),m,0.)
    wa=torch.exp(a.maximum-safe);wb=torch.exp(b.maximum-safe)
    return State(m,wa*a.denominator+wb*b.denominator,wa[:,None]*a.numerator+wb[:,None]*b.numerator)

def finalize(s):
    denom=torch.where(s.denominator>0,s.denominator,1.)
    return s.numerator/denom[:,None]

def _check(q,k,v,mask,dtype):
    if dtype not in (torch.float32,torch.float64):raise ValueError('CPU profile declares FP32 or FP64 compute only')
    if q.ndim!=2 or len(k.orig_shape)!=2 or len(v.orig_shape)!=2:raise ValueError('expected q=[r,d], K/V=[L,d]')
    if q.shape[1]!=k.orig_shape[1] or k.orig_shape[0]!=v.orig_shape[0]:raise ValueError('incompatible Q/K/V geometry')
    if mask is not None and (mask.dtype!=torch.bool or mask.shape!=(q.shape[0],k.orig_shape[0])):raise ValueError('mask must be boolean [r,L]')

def _weights(z,mask):
    if mask is not None:z=z.masked_fill(~mask,-torch.inf)
    m=z.max(dim=1).values
    safe=torch.where(torch.isfinite(m),m,0.)
    w=torch.exp(z-safe[:,None])
    return m,w

def dense(q,k,v,alpha,*,mask=None,compute_dtype=torch.float64):
    _check(q,k,v,mask,compute_dtype)
    if k.orig_shape[0]==0:return torch.zeros(q.shape[0],v.orig_shape[1],dtype=compute_dtype,device=q.device)
    z=(q.to(compute_dtype)@dequantize_int4_groupwise(k,compute_dtype).T)*alpha
    _,w=_weights(z,mask);l=w.sum(dim=1)
    p=w/torch.where(l>0,l,1.)[:,None]
    return p@dequantize_int4_groupwise(v,compute_dtype)

def staged(q,k,v,alpha,*,mask=None,compute_dtype=torch.float64):
    _check(q,k,v,mask,compute_dtype)
    if k.orig_shape[0]==0:return finalize(empty_state(q.shape[0],v.orig_shape[1],dtype=compute_dtype,device=q.device))
    z=direct_packed_score(q,k,alpha,compute_dtype=compute_dtype)
    m,w=_weights(z,mask)
    return finalize(State(m,w.sum(dim=1),direct_value(w,v,compute_dtype=compute_dtype)))

def slice_cache(p,start,stop):
    return PackedInt4(p.packed[start:stop],p.scale[start:stop],(stop-start,p.orig_shape[1]),p.group_size)

def block_state(q,k,v,alpha,*,mask=None,compute_dtype=torch.float64):
    _check(q,k,v,mask,compute_dtype)
    if k.orig_shape[0]==0:return empty_state(q.shape[0],v.orig_shape[1],dtype=compute_dtype,device=q.device)
    z=direct_packed_score(q,k,alpha,compute_dtype=compute_dtype)
    m,w=_weights(z,mask)
    return State(m,w.sum(dim=1),direct_value(w,v,compute_dtype=compute_dtype))

def streaming(q,k,v,alpha,*,block_size=16,mask=None,compute_dtype=torch.float64):
    _check(q,k,v,mask,compute_dtype)
    if type(block_size) is not int or block_size<=0:raise ValueError('positive integer block size required')
    state=empty_state(q.shape[0],v.orig_shape[1],dtype=compute_dtype,device=q.device)
    for start in range(0,k.orig_shape[0],block_size):
        stop=min(start+block_size,k.orig_shape[0])
        b=block_state(q,slice_cache(k,start,stop),slice_cache(v,start,stop),alpha,mask=None if mask is None else mask[:,start:stop],compute_dtype=compute_dtype)
        state=merge(state,b)
    return finalize(state)
