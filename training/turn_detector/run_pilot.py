"""Colab-only pilot driver; consumes completed caches, never benchmark datasets."""
from pathlib import Path
import os,time,json,sys,subprocess
ROOT=Path('/content/turn-recreation')
if not ROOT.exists():raise RuntimeError('Colab only')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
manifest=ROOT/'cache-streaming-v2/manifest.json'
expected=len(json.loads((ROOT/'selection.json').read_text()))
start=time.monotonic()
while True:
    records=json.loads(manifest.read_text()) if manifest.exists() else []
    if len(records)==expected:break
    if time.monotonic()-start>7200:raise TimeoutError('Caches did not complete')
    time.sleep(10)
from leakage_guard import validate_manifest
print('TRAINING_LEAKAGE_AUDIT',json.dumps(validate_manifest(manifest)),flush=True)
def run(name,argv):
    out=ROOT/'runs'/name;out.parent.mkdir(exist_ok=True)
    if out.exists():raise RuntimeError('Refuse to overwrite experiment '+name)
    print('RUN_START',name,flush=True)
    subprocess.run([sys.executable,*argv,'--manifest',str(manifest),'--out',str(out)],check=True,cwd=ROOT)
    print('RUN_DONE',name,(out/'best.json').read_text(),flush=True)
run('baseline-v1',['baseline.py','sweep'])
for variant in ('mlp','vap'):
    run(variant+'-v1-seed42',['heads.py','train','--variant',variant,'--epochs','12','--seed','42'])
print('PILOT_COMPLETE',flush=True)
