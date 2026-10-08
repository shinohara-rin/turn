from pathlib import Path
import subprocess,sys,os,json,tarfile,hashlib,shutil
assert Path('/content').is_dir()
os.chmod('/content/.hf_token',0o600)
subprocess.run([sys.executable,'-m','pip','install','-q','onnx==1.23.2','onnxruntime==1.30.0','silero-vad==6.2.0','pyyaml','huggingface_hub'],check=True)
from huggingface_hub import hf_hub_download
import torch,yaml
root=Path('/content/webgpu-playground');root.mkdir(exist_ok=True);out=root/'assets';out.mkdir(exist_ok=True)
token=Path('/content/.hf_token').read_text().strip()
def fetch(repo,name,revision):return Path(hf_hub_download(repo,name,revision=revision,token=token))
nemo=fetch('nvidia/parakeet_realtime_eou_120m-v1','parakeet_realtime_eou_120m-v1.nemo','a7e2b4629593dce0ec19f600e00e9904353fda2d')
assert hashlib.file_digest(nemo.open('rb'),'sha256').hexdigest()=='6603a22a53b7c1a4bac4736cb24628fb568a7102ba931a28c799e2e72f109893'
with tarfile.open(nemo) as t:
 config=next(m for m in t.getmembers() if m.name.endswith('model_config.yaml'))
 cfg=yaml.safe_load(t.extractfile(config));(root/'model_config.json').write_text(json.dumps(cfg,indent=2))
 weights=next(m for m in t.getmembers() if m.name.endswith('model_weights.ckpt'))
 with t.extractfile(weights) as src,(root/'model_weights.ckpt').open('wb') as dst:shutil.copyfileobj(src,dst)
state=torch.load(root/'model_weights.ckpt',map_location='cpu',weights_only=False)
head=fetch('shinohararin/pardon-turn-detector-v1','turn_head.pt','27622509159f24a99abc182593f5d624fafb6e70')
assert hashlib.file_digest(head.open('rb'),'sha256').hexdigest()=='cf911ffa6c2c21dd38d57f36784f7b948f67e642c1c04a0bdcd14028c24ffa40'
shutil.copyfile(head,root/'turn_head.pt')
h=torch.load(head,map_location='cpu',weights_only=False)
print(json.dumps({'preprocessor':cfg['preprocessor'],'head':{k:h[k] for k in ['dim','hidden','auxiliary']},'preprocessor_keys':[(k,list(v.shape)) for k,v in state.items() if k.startswith('preprocessor')]}),flush=True)
for n in ['startup.onnx','steady.onnx','weights.bin']:
 shutil.copyfile(fetch('shinohararin/pardon-turn-detector-v2-experiments','browser-compression/v1/export/fp32-dedup/'+n,'d400823b62a3170cb2aaa1c56871848b9622f71d'),out/n)
(root/'prepared.json').write_text(json.dumps({'prepared':True,'data_accessed':False}))
print('PREPARED',flush=True)
