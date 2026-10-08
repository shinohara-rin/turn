"""Remote sequential paired full-data experiment; resume exact existing jobs only."""
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path('/content/turn-recreation')
if not ROOT.is_dir():raise RuntimeError('Remote Colab required')
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
def run(argv):
    print('START',argv,flush=True)
    subprocess.run([sys.executable,*argv],cwd=ROOT,check=True)
selection=ROOT/'selection-full-train-dev.json'
if not selection.exists():run(['scale_cache.py','--split',str(ROOT/'split.json'),'--out',str(selection)])
records=json.loads(selection.read_text())
assert {p:sum(r['split']==p for r in records) for p in ('train','dev')}=={'train':131,'dev':16}
common=['cache_batch.py','--selection-manifest',str(selection),'--batch-size','16']
run(common+['--output-dir',str(ROOT/'cache-streaming-full-v1'),'--reuse-cache-manifest',str(ROOT/'cache-streaming-v2/manifest.json')])
for kind,cache in [('frozen','cache-streaming-full-v1'),('continued','cache-vap-update100-full-v1')]:
    # Surface frozen full-data quality before the second feature pass completes.
    if kind=='continued':
        run(common+['--output-dir',str(ROOT/'cache-vap-update100-full-v1'),'--encoder-checkpoint',str(ROOT/'runs/encoder-vap-v1/checkpoint.pt'),'--training-manifest',str(ROOT/'continuation-training-manifest.json'),'--policy-cache-manifest',str(ROOT/'cache-streaming-full-v1/manifest.json'),'--reuse-cache-manifest',str(ROOT/'cache-vap-update100/manifest.json')])
    manifest=ROOT/cache/'manifest.json'
    assert len(json.loads(manifest.read_text()))==147
    for seed in (42,17):
        out=ROOT/'runs'/f'full-{kind}-mlp-seed{seed}'
        argv=['heads.py','train','--variant','mlp','--epochs','12','--seed',str(seed),'--manifest',str(manifest),'--out',str(out)]
        if out.exists():argv+=['--resume']
        run(argv)
        print('FULL_HEAD_COMPLETE',kind,seed,(out/'best.json').read_text(),flush=True)
print('FULL_SCALE_COMPLETE',flush=True)
