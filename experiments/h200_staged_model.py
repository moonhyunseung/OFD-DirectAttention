"""Staged packed K/V integration gate, not fused online attention or production serving.
Uses existing frozen offset-binary quantizer; no approximation to the quantized operator is introduced.
Run: python experiments/h200_staged_model.py --out /absolute/new/output [--kernel-only]
The fixed prompts below are integration fixtures, not downstream quality evaluation datasets.
"""
import argparse,copy,gc,hashlib,json,random,subprocess,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,triton
import torch.nn.functional as F
import triton.language as tl
from upstream.directk.quant import quantize_int4_groupwise
from transformers import AutoModelForCausalLM,AutoTokenizer,AttentionInterface
from torch.nn.attention.bias import causal_lower_right

MODEL='Qwen/Qwen3.8-27B'
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
DERIVED_FROM=dict(DirectK='1d25e3350a96d9d6d6e2057b5a2e836fb64c134d',
                 DirectV='641287013ba072a7c2bc2da5520a865ca095c9e1')
# K/V kernels below add a batch*KV-head index and a capacity stride to the independently tested stages.
# K uses inline signed code conversion and group-major cache; V uses the held-out selected offset/128 policy.
@triton.jit
def score_batched(Q,P,S,O,L:tl.constexpr,CAP:tl.constexpr):
    ctx=tl.program_id(1);t=tl.program_id(0)*64+tl.arange(0,64)
    h=tl.arange(0,16);j=tl.arange(0,32);acc=tl.full((16,64),0,tl.float32)
    for g in tl.static_range(8):
        q=tl.load(Q+ctx*1536+h[:,None]*256+g*32+j[None,:],h[:,None]<6,0)
        byte=tl.load(P+ctx*128*CAP+g*16*CAP+(j[:,None]//2)*CAP+t[None,:],t[None,:]<L,0).to(tl.int32)
        c=(byte>>((j[:,None]%2)*4))&15
        code=(c-8).to(tl.float16)
        dot=tl.dot(q,code)
        s=tl.load(S+ctx*8*CAP+g*CAP+t,t<L,0).to(tl.float32)
        acc=acc+dot*s[None,:]
    tl.store(O+ctx*6*L+h[:,None]*L+t[None,:],acc*0.0625,(h[:,None]<6)&(t[None,:]<L))
@triton.jit
def value_batched(P,V,S,O,L:tl.constexpr,CAP:tl.constexpr,NCTX:tl.constexpr):
    block=tl.program_id(0);ch=tl.program_id(1);g=tl.program_id(2);ctx=ch//6
    t=block*128+tl.arange(0,128);j=tl.arange(0,32)
    p=tl.load(P+ch*L+t,t<L,0).to(tl.float32)
    s=tl.load(S+ctx*CAP*8+t*8+g,t<L,0).to(tl.float32);w=p*s
    byte=tl.load(V+ctx*CAP*128+t[:,None]*128+g*16+j[None,:]//2,t[:,None]<L,0).to(tl.int32)
    c=(byte>>((j[None,:]%2)*4))&15
    y=tl.sum(w[:,None]*c.to(tl.float32),0)-8*tl.sum(w,0)
    tl.store(O+block*NCTX*1536+ch*256+g*32+j,y)

class PackedStore:
    def __init__(self,k,v,steps,mode):
        self.b,self.h,self.pos,d=k.shape
        if d!=256:raise ValueError('Only d=256 is in this frozen adapter contract')
        self.n=self.b*self.h;self.cap=self.pos+steps;self.mode=mode
        self.kp=torch.zeros(self.n,8,16,self.cap,device=k.device,dtype=torch.uint8)
        self.ks=torch.zeros(self.n,8,self.cap,device=k.device,dtype=torch.float16)
        self.vp=torch.zeros(self.n,self.cap,8,16,device=k.device,dtype=torch.uint8)
        self.vs=torch.zeros(self.n,self.cap,8,device=k.device,dtype=torch.float16)
        ko=quantize_int4_groupwise(k);vo=quantize_int4_groupwise(v)
        self.kp[:,:,:,:self.pos].copy_(ko.packed.reshape(self.n,self.pos,8,16).permute(0,2,3,1))
        self.ks[:,:,:self.pos].copy_(ko.scale.reshape(self.n,self.pos,8).transpose(1,2))
        self.vp[:,:self.pos].copy_(vo.packed.reshape(self.n,self.pos,8,16))
        self.vs[:,:self.pos].copy_(vo.scale.reshape(self.n,self.pos,8))
    def append(self,k,v):
        if k.shape[2]!=1:raise ValueError('Decode-only adapter, one token per update')
        ko=quantize_int4_groupwise(k);vo=quantize_int4_groupwise(v)
        self.kp[:,:,:,self.pos].copy_(ko.packed.reshape(self.n,8,16))
        self.ks[:,:,self.pos].copy_(ko.scale.reshape(self.n,8))
        self.vp[:,self.pos].copy_(vo.packed.reshape(self.n,8,16))
        self.vs[:,self.pos].copy_(vo.scale.reshape(self.n,8));self.pos+=1
    def decoded(self,dtype=torch.float32):
        pk=self.kp[:,:,:,:self.pos].permute(0,3,1,2)
        pv=self.vp[:,:self.pos]
        def dec(p,s):
            c=torch.stack(((p&15).to(dtype),(p>>4).to(dtype)),-1).reshape(self.n,self.pos,8,32)
            return ((c-8)*s.to(dtype)[:,:,:,None]).reshape(self.n,self.pos,256)
        return dec(pk,self.ks[:,:,:self.pos].transpose(1,2)),dec(pv,self.vs[:,:self.pos])
    def direct(self,q,L=None):
        L=self.pos if L is None else L
        z=torch.empty(self.n,6,L,device=q.device,dtype=torch.float32)
        score_batched[(triton.cdiv(L,64),self.n)](q,self.kp,self.ks,z,L,self.cap,num_warps=4,enable_fp_fusion=False)
        p=torch.softmax(z,dim=-1)
        partial=torch.empty(triton.cdiv(L,128),self.n,6,256,device=q.device,dtype=torch.float32)
        value_batched[(triton.cdiv(L,128),self.n*6,8)](p,self.vp,self.vs,partial,L,self.cap,self.n,enable_fp_fusion=False)
        return partial.sum(0)
    def attention(self,q):
        if q.dtype!=torch.float16 or q.shape[1]!=self.h*6 or q.shape[2]!=1:
            raise ValueError('Adapter requires FP16 Q, r=6, one decode token')
        qq=q[:,:,0,:].reshape(self.n,6,256).contiguous()
        if self.mode=='direct':y=self.direct(qq)
        else:
            k,v=self.decoded()
            p=torch.softmax(torch.matmul(qq.float(),k.transpose(-1,-2))/16,dim=-1)
            y=torch.matmul(p,v)
        return y.to(q.dtype).reshape(self.b,self.h*6,256)[:,None,:,:]
    def precompile(self,steps):
        if self.mode=='direct':
            q=torch.zeros(self.n,6,256,device=self.kp.device,dtype=torch.float16)
            for L in range(self.pos+1,self.pos+steps+1):self.direct(q,L)
    @property
    def kv_bytes(self):return sum(x.numel()*x.element_size() for x in [self.kp,self.ks,self.vp,self.vs])

class NativeStore:
    def __init__(self,k,v,steps):
        b,h,t,d=k.shape;self.pos=t
        self.k=torch.empty(b,h,t+steps,d,device=k.device,dtype=k.dtype);self.v=torch.empty_like(self.k)
        self.k[:,:,:t].copy_(k);self.v[:,:,:t].copy_(v)
    def append(self,k,v):
        self.k[:,:,self.pos:self.pos+1].copy_(k);self.v[:,:,self.pos:self.pos+1].copy_(v);self.pos+=1
    def attention(self,q):
        y=F.scaled_dot_product_attention(q,self.k[:,:,:self.pos],self.v[:,:,:self.pos],enable_gqa=True,scale=1/16)
        return y.transpose(1,2)
    def precompile(self,steps):pass
    @property
    def kv_bytes(self):return (self.k.numel()+self.v.numel())*self.k.element_size()

def move_cache(cache,device):
    for layer in cache.layers:
        for key,val in vars(layer).items():
            if isinstance(val,torch.Tensor):setattr(layer,key,val.to(device))
    return cache

def install(cache,mode,steps):
    stores={}
    for i,layer in enumerate(cache.layers):
        if getattr(layer,'keys',None) is None:continue
        k,v=layer.keys,layer.values
        store=NativeStore(k,v,steps) if mode=='native' else PackedStore(k,v,steps,mode)
        stores[i]=store;layer.keys=None;layer.values=None
        def update(self,k,v,*args,_s=store,**kw):
            _s.append(k,v);return k,v
        def length(self,*args,_s=store,**kw):return _s.pos
        layer.update=types.MethodType(update,layer);layer.get_seq_length=types.MethodType(length,layer)
    if len(stores)!=16:raise RuntimeError(f'Expected 16 full-attention layers; observed {len(stores)}')
    return stores

def kernel_gate(out):
    torch.manual_seed(1760);rows=[]
    for b,h in [(1,1),(1,4),(2,4)]:
        for L in [1,17,129,1024]:
            k=torch.randn(b,h,L,256,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
            store=PackedStore(k,v,3,'direct')
            for step in range(3):
                store.append(torch.randn(b,h,1,256,device='cuda',dtype=torch.float16),torch.randn(b,h,1,256,device='cuda',dtype=torch.float16))
                q=torch.randn(b*h,6,256,device='cuda',dtype=torch.float16)
                y=store.direct(q);kd,vd=store.decoded(torch.float64)
                ref=torch.softmax(torch.matmul(q.double(),kd.transpose(-1,-2))/16,dim=-1)@vd
                err=(y.double()-ref).abs();rel=torch.linalg.vector_norm(y.double()-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-300)
                ok=bool(torch.isfinite(y).all() and rel<1e-4)
                rows.append(dict(batch=b,kv_heads=h,initial_length=L,step=step,max_abs=err.max().item(),relative_l2=rel.item(),passed=ok))
    (out/'kernel_gate.json').write_text(json.dumps(rows,indent=2))
    if not all(r['passed'] for r in rows):raise RuntimeError('Batched staged composition gate failed')
    print('BATCHED_STAGED_GATE_PASS',len(rows),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--kernel-only',action='store_true')
    a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1760);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    m=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),derived_stage_commits=DERIVED_FROM,
        quantizer_lock=json.loads((ROOT/'manifests/upstream_lock.json').read_text()),model_id=MODEL,model_revision=REVISION,
        torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),
        scope='staged full-model teacher-forced integration and feasibility timing; fixed repeated prompt is not downstream quality evidence',
        model_gate='direct vs decoded-quantized logits relL2<=0.002 and abs NLL delta<=0.01; native replacement relL2<=0.002',
        limitations=['no fused online softmax','logits/probabilities and V partial workspace materialized','reference PyTorch append quantization included in both quantized paths','prefill, conversion, transfer and JIT excluded','optional linear-attention fallbacks retained in raw log','no optimized quantized-serving baseline'])
    def save():(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
    save()
    try:
        kernel_gate(out)
        if a.kernel_only:m['status']='KERNEL_GATE_PASS';save();return
        active=None
        def attention(module,q,k,v,attention_mask,dropout=0.,scaling=None,**kwargs):
            if active is not None and module.layer_idx in active:return active[module.layer_idx].attention(q),None
            if q.shape[2]>1:
                bias=None if q.shape[2]==k.shape[2] else causal_lower_right(q.shape[2],k.shape[2])
                y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,is_causal=bias is None,enable_gqa=True,scale=scaling)
            else:y=F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=True,scale=scaling)
            return y.transpose(1,2),None
        AttentionInterface.register('ofd_staged_gate',attention)
        tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True,trust_remote_code=False)
        print('MODEL_LOADING',MODEL,REVISION,flush=True)
        model,loading=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,local_files_only=True,dtype=torch.float16,
            device_map='cuda',attn_implementation='ofd_staged_gate',trust_remote_code=False,output_loading_info=True)
        if loading.get('missing_keys'):raise RuntimeError(str(loading['missing_keys'][:8]))
        model.eval();m['parameter_bytes']=sum(x.numel()*x.element_size() for x in model.parameters());save()
        prompt='A research system must separate exact representations, floating-point execution, and measured hardware behavior. Every experiment fixes the inputs and compares the same mathematical operator. def square(x): return x * x. '
        ids=tok.encode(prompt,add_special_tokens=False)
        tokens=(ids*((8192+33)//len(ids)+1))[:8192+33]
        (out/'tokens.json').write_text(json.dumps(dict(token_ids=tokens,sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),fixture='generated repeated prose/code integration text; not a quality benchmark')))
        results=[]
        for context in [128,2048,8192]:
            steps=8 if context==128 else 16;warm=2 if context==128 else 4;rounds=1 if context==128 else 3
            active=None;cache=None
            with torch.inference_mode():
                for start in range(0,context,512):
                    xx=torch.tensor([tokens[start:min(start+512,context)]],device='cuda')
                    res=model(input_ids=xx,past_key_values=cache,use_cache=True,logits_to_keep=1);cache=res.past_key_values;del res,xx
            template=move_cache(cache,'cpu');del cache;gc.collect();torch.cuda.empty_cache()
            inputs=[torch.tensor([[tokens[context+s]]],device='cuda') for s in range(steps)]
            positions=[torch.tensor([[context+s]],device='cuda') for s in range(steps)]
            target=torch.tensor([tokens[context+s+1] for s in range(warm,steps)],device='cuda')
            refs={}
            modes=['unmodified','native','decoded_quant','direct'] if context==128 else ['native','decoded_quant','direct']
            for rnd in range(rounds):
                order=modes.copy()
                if rnd:random.Random(1760+rnd).shuffle(order)
                for mode in order:
                    current=move_cache(copy.deepcopy(template),'cuda')
                    active=None if mode=='unmodified' else install(current,mode,steps)
                    if active:
                        for store in active.values():store.precompile(steps)
                    torch.cuda.synchronize();events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(steps-warm)]
                    logits=[];torch.cuda.reset_peak_memory_stats();wall=0
                    with torch.inference_mode():
                        for s in range(steps):
                            if s==warm:torch.cuda.synchronize();wall=time.perf_counter()
                            if s>=warm:events[s-warm][0].record()
                            res=model(input_ids=inputs[s],position_ids=positions[s],attention_mask={'full_attention':None,'linear_attention':None},past_key_values=current,use_cache=True,logits_to_keep=1)
                            if s>=warm:events[s-warm][1].record();logits.append(res.logits[:,0,:])
                            current=res.past_key_values;del res
                    torch.cuda.synchronize();ms=(time.perf_counter()-wall)*1000/(steps-warm)
                    y=torch.cat(logits).float()
                    if not torch.isfinite(y).all():raise RuntimeError('Nonfinite model logits')
                    nll=F.cross_entropy(y,target).item()
                    if mode in ['unmodified','decoded_quant','native'] and mode not in refs:refs[mode]=dict(y=y.cpu(),nll=nll)
                    refname='decoded_quant' if mode=='direct' else 'unmodified' if context==128 and mode=='native' else None
                    rel=None;delta=None;agree=None
                    if refname:
                        ref=refs[refname];ry=ref['y'].to(y.device);rel=((y-ry).norm()/ry.norm()).item()
                        delta=nll-ref['nll'];agree=(y.argmax(-1)==ry.argmax(-1)).float().mean().item()
                        if rel>0.002 or abs(delta)>0.01:raise RuntimeError(f'Model numerical gate failed: {mode} rel={rel} nll_delta={delta}')
                    row=dict(context=context,round=rnd,mode=mode,steps=steps,warmup=warm,wall_ms=ms,gpu_event_ms=sum(a.elapsed_time(b) for a,b in events)/(steps-warm),
                        nll=nll,comparison=refname,logit_relative_l2=rel,nll_delta=delta,argmax_agreement=agree,
                        kv_bytes=sum(x.kv_bytes for x in active.values()) if active else None,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    results.append(row);(out/'model_rows.json').write_text(json.dumps(results,indent=2));print('MODEL_ROW',json.dumps(row),flush=True)
                    active=None;del current,y,logits
                    gc.collect();torch.cuda.empty_cache()
            del template,inputs,positions,refs
            gc.collect();torch.cuda.empty_cache()
        m.update(status='PASS',model_rows=len(results));save();print('STAGED_MODEL_PASS',out,flush=True)
    except Exception as e:m.update(status='FAIL',error=repr(e));save();raise
if __name__=='__main__':main()
