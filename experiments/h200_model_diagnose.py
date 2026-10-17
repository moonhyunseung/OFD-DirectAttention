"""Untimed causal diagnosis of the staged-model gate failure.
Separates same-state operator errors from quantized-cache feedback.
Frozen-append mode is a counterfactual control, never a deployable benchmark.
"""
import argparse,copy,gc,hashlib,json,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AttentionInterface,AutoModelForCausalLM,AutoTokenizer
from torch.nn.attention.bias import causal_lower_right
import h200_staged_model as base
EXPECTED='2a4ff236ad8527cfcc2ac2051d9a447327e53ff646e3ea3fb9046805dbd878b0'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()==EXPECTED
    data=json.loads(a.tokens.read_text());tokens=data['token_ids']
    assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest()==data['sha256']
    torch.manual_seed(1760);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    m=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=EXPECTED,
        token_sha256=data['sha256'],model=base.MODEL,revision=base.REVISION,context=128,steps=8,warmup=2,
        scope='untimed same-state versus trajectory-feedback diagnosis',model_gate_unchanged='relL2<=0.002; abs NLL delta<=0.01',
        frozen_append='counterfactual uses reference packed new K/V and stored scales, while queries and native linear states evolve independently',
        torch=torch.__version__,gpu=torch.cuda.get_device_name())
    def save():(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
    save();active=None;mode='';records=[];base_rows={};base_q={}
    original_append=base.PackedStore.append
    original_attention=base.PackedStore.attention
    def append(self,k,v):
        original_append(self,k,v)
        key=(self.layer,self.pos);idx=self.pos-1
        cur=[self.kp[:,:,:,idx],self.ks[:,:,idx],self.vp[:,idx],self.vs[:,idx]]
        if mode=='reference':base_rows[key]=[x.detach().cpu().clone() for x in cur]
        else:
            refs=[x.to(cur[0].device) for x in base_rows[key]]
            record=dict(kind='append',mode=mode,layer=self.layer,step=self.pos-129,
                        K_bytes_changed=int((cur[0]!=refs[0]).sum()),K_scales_changed=int((cur[1]!=refs[1]).sum()),
                        V_bytes_changed=int((cur[2]!=refs[2]).sum()),V_scales_changed=int((cur[3]!=refs[3]).sum()))
            records.append(record)
            if mode=='frozen_append':
                for dst,src in zip(cur,refs):dst.copy_(src)
    def attention(self,q):
        key=(self.layer,self.pos)
        if mode=='reference':
            base_q[key]=q.detach().cpu()
            qq=q[:,:,0,:].reshape(self.n,6,256).contiguous()
            kd,vd=self.decoded()
            zr=torch.matmul(qq.float(),kd.transpose(-1,-2))/16
            z=torch.empty_like(zr)
            base.score_batched[(base.triton.cdiv(self.pos,64),self.n)](qq,self.kp,self.ks,z,self.pos,self.cap,num_warps=4,enable_fp_fusion=False)
            pr=torch.softmax(zr,-1);p=torch.softmax(z,-1)
            yr=pr@vd;yd=self.direct(qq)
            def rel(x,y):return ((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-30)).item()
            records.append(dict(kind='same_state',layer=self.layer,step=self.pos-129,q_absmax=q.abs().max().item(),
                score_max_abs=(z-zr).abs().max().item(),score_rel=rel(z,zr),probability_rel=rel(p,pr),
                output_rel=rel(yd,yr),output_max_abs=(yd-yr).abs().max().item(),
                half_output_rel=rel(yd.half(),yr.half()),half_values_changed=int((yd.half()!=yr.half()).sum())))
            return yr.half().reshape(self.b,self.h*6,256)[:,None,:,:]
        ref=base_q[key].to(q.device)
        records.append(dict(kind='query',mode=mode,layer=self.layer,step=self.pos-129,
            relative_l2=((q.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30)).item(),values_changed=int((q!=ref).sum())))
        return original_attention(self,q)
    base.PackedStore.append=append;base.PackedStore.attention=attention
    def attn(module,q,k,v,attention_mask,dropout=0.,scaling=None,**kw):
        if active is not None and module.layer_idx in active:return active[module.layer_idx].attention(q),None
        if q.shape[2]>1:
            bias=None if q.shape[2]==k.shape[2] else causal_lower_right(q.shape[2],k.shape[2])
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,is_causal=bias is None,enable_gqa=True,scale=scaling)
        else:y=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,scale=scaling)
        return y.transpose(1,2),None
    AttentionInterface.register('ofd_diagnose',attn)
    try:
        model=AutoModelForCausalLM.from_pretrained(base.MODEL,revision=base.REVISION,local_files_only=True,dtype=torch.float16,
            device_map='cuda',attn_implementation='ofd_diagnose',trust_remote_code=False).eval()
        with torch.inference_mode():
            res=model(input_ids=torch.tensor([tokens[:128]],device='cuda'),use_cache=True,logits_to_keep=1)
            template=base.move_cache(res.past_key_values,'cpu');del res;gc.collect();torch.cuda.empty_cache()
            outputs={}
            for mode in ['reference','direct','frozen_append']:
                cache=base.move_cache(copy.deepcopy(template),'cuda');active=base.install(cache,'decoded_quant' if mode=='reference' else 'direct',8)
                for i,s in active.items():s.layer=i
                ys=[]
                for step in range(8):
                    res=model(input_ids=torch.tensor([[tokens[128+step]]],device='cuda'),position_ids=torch.tensor([[128+step]],device='cuda'),
                        attention_mask={'full_attention':None,'linear_attention':None},past_key_values=cache,use_cache=True,logits_to_keep=1)
                    cache=res.past_key_values;ys.append(res.logits[:,0,:].float().cpu());del res
                outputs[mode]=torch.cat(ys);active=None;del cache;gc.collect();torch.cuda.empty_cache()
                print('DIAG_MODE_DONE',mode,flush=True)
        ref=outputs['reference'][2:];target=torch.tensor(tokens[131:137])
        # target step s predicts tokens[129+s]; warmup removes s=0,1 => tokens[131:137].
        nll=F.cross_entropy(ref,target).item();metrics={}
        for name,y in outputs.items():
            yy=y[2:];metrics[name]=dict(logit_relative_l2=((yy-ref).norm()/ref.norm()).item(),
                nll_delta=F.cross_entropy(yy,target).item()-nll,argmax_agreement=(yy.argmax(-1)==ref.argmax(-1)).float().mean().item())
        same=[x for x in records if x['kind']=='same_state']
        summary=dict(same_state_max_output_rel=max(x['output_rel'] for x in same),
            same_state_max_half_output_rel=max(x['half_output_rel'] for x in same),same_state_max_score_abs=max(x['score_max_abs'] for x in same),
            changed_appends={md:sum(x.get('K_bytes_changed',0)+x.get('V_bytes_changed',0) for x in records if x.get('mode')==md and x['kind']=='append') for md in ['direct','frozen_append']})
        (out/'local_records.json').write_text(json.dumps(records,indent=2));(out/'metrics.json').write_text(json.dumps(dict(metrics=metrics,summary=summary),indent=2))
        m.update(status='DIAGNOSED',metrics=metrics,summary=summary);save();print(json.dumps(dict(metrics=metrics,summary=summary),indent=2),flush=True)
    except Exception as e:m.update(status='FAIL',error=repr(e));save();raise
if __name__=='__main__':main()
