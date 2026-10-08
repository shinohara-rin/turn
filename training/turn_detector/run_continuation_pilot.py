"""Remote-only evaluation of the fixed 100-update encoder VAP intervention."""
from pathlib import Path
import json,os,time,subprocess,sys

ROOT=Path('/content/turn-recreation')
if not ROOT.exists():raise RuntimeError('Colab only')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
manifest=ROOT/'cache-streaming-v2/manifest.json'
checkpoint=ROOT/'runs/encoder-vap-v1/checkpoint.pt'
expected=len(json.loads((ROOT/'selection.json').read_text()))
start=time.monotonic()
while True:
    records=json.loads(manifest.read_text()) if manifest.exists() else []
    log=ROOT/'runs/encoder-vap-v1/train.log'
    # Last update log precedes atomic checkpoint save; wait for process exit too.
    pidfile=log.with_name('train.pid')
    proc=Path('/proc/'+pidfile.read_text()) if pidfile.exists() else None
    try:
        alive=proc is not None and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:
        alive=False
    if len(records)==expected and checkpoint.exists() and not alive:break
    if time.monotonic()-start>7200:raise TimeoutError('Original cache or continuation did not finish')
    time.sleep(15)
import torch
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
if state['update']!=100:raise RuntimeError('Continuation did not reach frozen 100-update budget')
del state
cache=ROOT/'cache-vap-update100'
print('CONTINUED_CACHE_START',flush=True)
subprocess.run([sys.executable,'cache_batch.py','--batch-size','4',
                '--encoder-checkpoint',str(checkpoint),'--output-dir',str(cache),
                '--training-manifest',str(ROOT/'continuation-training-manifest.json')],cwd=ROOT,check=True)
out=ROOT/'runs/continued-mlp-v1-seed42'
argv=[sys.executable,'heads.py','train','--variant','mlp','--epochs','12','--seed','42',
      '--manifest',str(cache/'manifest.json'),'--out',str(out)]
if out.exists():argv+=['--resume']
print('CONTINUED_HEAD_START',flush=True)
subprocess.run(argv,cwd=ROOT,check=True)
print('CONTINUED_PILOT_COMPLETE',(out/'best.json').read_text(),flush=True)
