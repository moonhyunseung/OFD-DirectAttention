"""Run clean-source CPU reference tests and archive numerical characterization."""
import json,subprocess,sys,platform,hashlib,uuid
from pathlib import Path
from datetime import datetime,timezone
import torch,numpy,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from numerical_profile import run

def main():
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit source before formal verification')
    out=ROOT/'results'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]);out.mkdir()
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    m=dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status='RUNNING',command=sys.argv,python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=numpy.__version__,pytest=pytest.__version__,device='cpu',threads=1,seed=1729,performance_claim=False,production_numerical_contract_closed=False,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in tracked if p.endswith('.py')})
    def save():(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
    save()
    with (out/'run.log').open('w') as log:
        rc=subprocess.run([sys.executable,'-m','pytest','-q','tests'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        m['test_return_code']=rc
        try:m['numerical']=run(out)
        except Exception:
            import traceback;traceback.print_exc(file=log);m['status']='FAIL';save();raise
    m['status']='PASS' if rc==0 and m['numerical']['all_finite'] else 'FAIL';save()
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(out,m['status'],m['numerical'])
    if m['status']!='PASS':raise RuntimeError('Gate failed; evidence retained')
if __name__=='__main__':main()
