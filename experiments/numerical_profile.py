"""CPU direct-attention characterization, no performance claims."""
import csv,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference.executor import dense,staged,streaming,slice_cache
from upstream.directk.quant import quantize_int4_groupwise

def run(out):
    torch.set_num_threads(1);rows=[]
    for dtype in [torch.float64,torch.float32]:
      for length in [0,1,17,129,512]:
       gen=torch.Generator().manual_seed(1729+length)
       q=torch.randn(6,256,generator=gen,dtype=torch.float32).to(dtype)
       k=quantize_int4_groupwise(torch.randn(max(1,length),256,generator=gen));k=slice_cache(k,0,length)
       v=quantize_int4_groupwise(torch.randn(max(1,length),256,generator=gen));v=slice_cache(v,0,length)
       for case in ['unmasked','partial_and_empty_row','large_logits']:
        qq=q*10000 if case=='large_logits' else q
        mask=None
        if case=='partial_and_empty_row':
            mask=torch.arange(length)[None,:]<=torch.arange(6)[:,None]*max(1,length//5);mask[0]=False
        ref=dense(qq,k,v,1/16,mask=mask,compute_dtype=dtype)
        for path,block in [('staged',0),('streaming',1),('streaming',16),('streaming',31)]:
            y=staged(qq,k,v,1/16,mask=mask,compute_dtype=dtype) if path=='staged' else streaming(qq,k,v,1/16,mask=mask,compute_dtype=dtype,block_size=block)
            e=(y-ref).to(torch.float64);rn=torch.linalg.vector_norm(ref.to(torch.float64)).item();en=torch.linalg.vector_norm(e).item()
            rows.append(dict(case=case,length=length,r=6,d=256,group_size=32,scale_dtype='float16',input_dtype='float32_cast_to_compute',compute_dtype=str(dtype),path=path,block_size=block,max_abs=e.abs().max().item(),relative_l2=en/rn if rn else (0. if en==0 else 'undefined_zero_reference'),finite=bool(torch.isfinite(y).all()),seed=1729+length))
    with (out/'numerical.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return dict(rows=len(rows),all_finite=all(r['finite'] for r in rows),max_abs=max(r['max_abs'] for r in rows),max_relative_l2=max(r['relative_l2'] for r in rows if isinstance(r['relative_l2'],float)))
