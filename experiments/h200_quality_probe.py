"""Independent held-out text diagnostic after a model-fidelity gate failure.
This does not change that gate, promote the backend, or establish task accuracy/serving performance.
Measures paired teacher-forced NLL, top-1 prediction agreement and logit drift.
No dataset text, weights or full logits are committed by this program.
"""
import argparse,copy,gc,hashlib,json,subprocess,random
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,AttentionInterface
from torch.nn.attention.bias import causal_lower_right
from datasets import load_dataset
import h200_staged_model as base
SPECS=[
 dict(id='Salesforce/wikitext',revision='b08601e04326c79dfdd32d625aee71d232d685c3',subset='wikitext-2-raw-v1',domain='prose'),
 dict(id='google-research-datasets/mbpp',revision='4bb6404fdc6cacfda99d4ac4205087b89d32030c',subset='full',domain='code'),
 dict(id='openai/gsm8k',revision='740312add88f781978c0658806c59bc2815b9866',subset='main',domain='math')]
def text(row):
    if 'code' in row:return '\n'.join([row.get('text',''),row.get('test_setup_code',''),row['code'],*row.get('test_list',[]),*row.get('challenge_test_list',[])])
    if 'question' in row:return row['question']+'\n'+row.get('answer','')
    return row.get('text','')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    expected='2a4ff236ad8527cfcc2ac2051d9a447327e53ff646e3ea3fb9046805dbd878b0'
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()==expected
    torch.manual_seed(1770);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    m=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=expected,
        model_id=base.MODEL,model_revision=base.REVISION,context_lengths=[512,2048],steps=64,warmup=0,
        scope='quality diagnostic only, prior 0.002 model-logit gate remains failed',
        non_claims=['not perplexity equivalence over a corpus','not code execution accuracy','not math exact-match accuracy','not model/serving speed','not production acceptance'],
        datasets=SPECS,seed=1770,torch=torch.__version__,gpu=torch.cuda.get_device_name())
    def save():(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
    save();active=None
    def attention(module,q,k,v,attention_mask,dropout=0.,scaling=None,**kwargs):
        if active is not None and module.layer_idx in active:return active[module.layer_idx].attention(q),None
        if q.shape[2]>1:
            bias=None if q.shape[2]==k.shape[2] else causal_lower_right(q.shape[2],k.shape[2])
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,is_causal=bias is None,enable_gqa=True,scale=scaling)
        else:y=F.scaled_dot_product_attention(q,k,v,enable_gqa=True,scale=scaling)
        return y.transpose(1,2),None
    AttentionInterface.register('ofd_quality_probe',attention)
    try:
        tok=AutoTokenizer.from_pretrained(base.MODEL,revision=base.REVISION,local_files_only=True,trust_remote_code=False)
        items=[]
        for spec in SPECS:
            ds=load_dataset(spec['id'],spec['subset'],revision=spec['revision'],split='test')
            tokens=[];ids=[]
            for i,row in enumerate(ds):
                txt=text(row)
                if not txt.strip():continue
                tokens.extend(tok.encode(txt,add_special_tokens=False)+[tok.eos_token_id]);ids.append(i)
                if len(tokens)>=2113:break
            if len(tokens)<2113:raise RuntimeError('Insufficient unique test data; do not cycle')
            tokens=tokens[:2113];item=dict(spec=spec,row_ids=ids,tokens=tokens,token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
            items.append(item)
        (out/'data_manifest.json').write_text(json.dumps([{k:v for k,v in x.items() if k!='tokens'} for x in items],indent=2))
        model=AutoModelForCausalLM.from_pretrained(base.MODEL,revision=base.REVISION,local_files_only=True,dtype=torch.float16,
            device_map='cuda',attn_implementation='ofd_quality_probe',trust_remote_code=False).eval()
        rows=[]
        with torch.inference_mode():
            for item in items:
                tokens=item['tokens']
                for context in [512,2048]:
                    active=None;cache=None
                    for start in range(0,context,512):
                        res=model(input_ids=torch.tensor([tokens[start:min(context,start+512)]],device='cuda'),past_key_values=cache,use_cache=True,logits_to_keep=1)
                        cache=res.past_key_values;del res
                    template=base.move_cache(cache,'cpu');del cache;gc.collect();torch.cuda.empty_cache();outputs={}
                    modes=['native','decoded_quant','direct']
                    # No performance measurement; reference is computed before paired diagnostics.
                    for mode in modes:
                        current=base.move_cache(copy.deepcopy(template),'cuda');active=base.install(current,mode,64)
                        if mode=='direct':
                            next(iter(active.values())).precompile(64)
                        ys=[]
                        for step in range(64):
                            res=model(input_ids=torch.tensor([[tokens[context+step]]],device='cuda'),position_ids=torch.tensor([[context+step]],device='cuda'),
                                attention_mask={'full_attention':None,'linear_attention':None},past_key_values=current,use_cache=True,logits_to_keep=1)
                            current=res.past_key_values;ys.append(res.logits[:,0,:].float().cpu());del res
                        outputs[mode]=torch.cat(ys);active=None;del current;gc.collect();torch.cuda.empty_cache()
                        print('PROBE_MODE',item['spec']['domain'],context,mode,flush=True)
                    target=torch.tensor(tokens[context+1:context+65])
                    losses={name:F.cross_entropy(y,target,reduction='none') for name,y in outputs.items()}
                    ref=outputs['decoded_quant'];direct=outputs['direct'];native=outputs['native']
                    delta=losses['direct']-losses['decoded_quant']
                    row=dict(domain=item['spec']['domain'],context=context,tokens=64,
                        mean_nll={name:l.mean().item() for name,l in losses.items()},direct_minus_decoded_nll=delta.mean().item(),
                        paired_delta_min=delta.min().item(),paired_delta_max=delta.max().item(),
                        direct_decoded_argmax_agreement=(direct.argmax(-1)==ref.argmax(-1)).float().mean().item(),
                        native_direct_argmax_agreement=(native.argmax(-1)==direct.argmax(-1)).float().mean().item(),
                        direct_decoded_logit_rel_l2=((direct-ref).norm()/ref.norm()).item(),
                        per_step=dict(direct_minus_decoded_nll=delta.tolist(),direct_ref_rel_l2=((direct-ref).norm(dim=-1)/ref.norm(dim=-1)).tolist(),
                                      native_nll=losses['native'].tolist(),decoded_nll=losses['decoded_quant'].tolist(),direct_nll=losses['direct'].tolist()))
                    rows.append(row);(out/'probe_rows.json').write_text(json.dumps(rows,indent=2))
                    print('PROBE_ROW',json.dumps({k:v for k,v in row.items() if k!='per_step'}),flush=True)
                    del template,outputs,losses,ref,direct,native
                    gc.collect();torch.cuda.empty_cache()
        m.update(status='DIAGNOSTIC_COMPLETED',cases=len(rows),paired_target_tokens=sum(x['tokens'] for x in rows));save();print('QUALITY_PROBE_COMPLETED',out,flush=True)
    except Exception as e:m.update(status='FAIL',error=repr(e));save();raise
if __name__=='__main__':main()
