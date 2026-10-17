import pytest
import torch
from reference.executor import dense,staged,streaming,block_state,slice_cache,merge,finalize,empty_state
from upstream.directk.quant import quantize_int4_groupwise

def fixture(length=33):
    g=torch.Generator().manual_seed(1729+length)
    q=torch.randn(6,256,generator=g,dtype=torch.float64)
    k=quantize_int4_groupwise(torch.randn(max(1,length),256,generator=g))
    v=quantize_int4_groupwise(torch.randn(max(1,length),256,generator=g))
    return q,slice_cache(k,0,length),slice_cache(v,0,length)

@pytest.mark.parametrize('length',[0,1,15,16,17,33,129])
@pytest.mark.parametrize('block',[1,16,31])
def test_partition_masks_and_empty(length,block):
    q,k,v=fixture(length)
    mask=torch.arange(length)[None,:] <= torch.arange(6)[:,None]*25
    mask[0]=False
    ref=dense(q,k,v,1/16,mask=mask)
    for y in [staged(q,k,v,1/16,mask=mask),streaming(q,k,v,1/16,mask=mask,block_size=block)]:
        assert torch.isfinite(y).all()
        assert torch.equal(y[0],torch.zeros(256,dtype=torch.float64))
        torch.testing.assert_close(y,ref,rtol=1e-12,atol=1e-12)

def test_merge_tree_order_and_identity():
    q,k,v=fixture(49)
    states=[block_state(q,slice_cache(k,a,b),slice_cache(v,a,b),1/16) for a,b in [(0,16),(16,32),(32,49)]]
    neutral=empty_state(6,256,dtype=torch.float64,device=q.device)
    ref=dense(q,k,v,1/16)
    for s in [merge(merge(states[0],states[1]),states[2]),merge(states[2],merge(states[1],states[0])),merge(neutral,merge(states[0],merge(states[1],states[2])))]:
        torch.testing.assert_close(finalize(s),ref,rtol=1e-12,atol=1e-12)

def test_extreme_logits_are_stable():
    q,k,v=fixture(33);q=q*10000
    y=streaming(q,k,v,1/16,block_size=16)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y,dense(q,k,v,1/16),rtol=1e-12,atol=1e-12)

def test_no_dense_decoder_or_full_staged_call(monkeypatch):
    import reference.executor as e
    q,k,v=fixture(33)
    def forbidden(*a,**kw):raise AssertionError('streaming reconstructed a dense baseline')
    monkeypatch.setattr(e,'dequantize_int4_groupwise',forbidden)
    monkeypatch.setattr(e,'staged',forbidden)
    assert torch.isfinite(e.streaming(q,k,v,1/16)).all()

def test_invalid_inputs():
    q,k,v=fixture(17)
    with pytest.raises(ValueError):streaming(q,k,v,1.,block_size=0)
    with pytest.raises(ValueError):streaming(q,k,v,1.,mask=torch.ones(6,17))
    with pytest.raises(ValueError):streaming(q,k,v,1.,compute_dtype=torch.float16)
