from pathlib import Path
import os,sys,json,concurrent.futures,hashlib
os.environ['HF_TOKEN']=Path('/content/.hf_token').read_text().strip()
os.environ['HF_HOME']='/content/hf'
from huggingface_hub import HfApi,hf_hub_download
root=Path('/content/turn-recreation'); root.mkdir(exist_ok=True)
repo='otoearth/otoSpeech-full-duplex-turn-104h'; revision='46f520297f434edf804389f82f9075a59d2f8268'
files=HfApi().list_repo_files(repo,repo_type='dataset',revision=revision)
metadata=[x for x in files if x.endswith('/metadata.json')]
def get(f):
    p=hf_hub_download(repo,f,repo_type='dataset',revision=revision)
    d=json.loads(Path(p).read_text());d['_dir']=f.rsplit('/',1)[0];return d
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool: rows=list(pool.map(get,metadata))
(root/'metadata.json').write_text(json.dumps(rows))
parent={}
def find(x):
    parent.setdefault(x,x)
    if parent[x]!=x:parent[x]=find(parent[x])
    return parent[x]
for r in rows:
    a,b=r['speaker_1_actor_id'],r['speaker_2_actor_id']
    if not a or not b:raise ValueError('missing actor id')
    parent[find(a)]=find(b)
groups={}
for r in rows:groups.setdefault(find(r['speaker_1_actor_id']),[]).append(r['_dir'])
print(json.dumps({'conversations':len(rows),'speakers':len(parent),'component_sizes':sorted([len(x) for x in groups.values()],reverse=True),'metadata_schema':list(rows[0])}),flush=True)
(root/'components.json').write_text(json.dumps(list(groups.values())))
sys.path.insert(0,'/content/turnbench')
try:
    import nemo.collections.asr as nemo_asr
    model=nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet_realtime_eou_120m-v1')
    print('MODEL_READY',type(model).__name__,str(model.cfg.preprocessor),str(model.encoder.streaming_cfg),flush=True)
except Exception as e:
    import traceback;traceback.print_exc()
