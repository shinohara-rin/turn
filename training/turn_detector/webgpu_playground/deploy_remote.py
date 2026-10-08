"""Publish only the authorized public browser deployment, from remote CPU."""
from pathlib import Path
import tarfile,shutil,json,hashlib
from huggingface_hub import HfApi
root=Path('/content/webgpu-playground');site=root/'site';site.mkdir(exist_ok=True)
with tarfile.open('/content/playground-site.tar.gz') as t:t.extractall(site,filter='data')
shutil.copytree(root/'assets',site/'assets',dirs_exist_ok=True)
api=HfApi(token=Path('/content/.hf_token').read_text().strip());repo='shinohararin/pardon-turn-webgpu'
api.create_repo(repo_id=repo,repo_type='space',space_sdk='static',private=False,exist_ok=True)
# Staging root is a strict allowlist: no checkpoints, datasets, auth or training state.
assert all(p.suffix not in {'.pt','.ckpt','.nemo','.parquet'} and 'token' not in p.name.lower() for p in site.rglob('*') if p.is_file())
c=api.upload_folder(repo_id=repo,repo_type='space',folder_path=str(site),commit_message='Publish local WebGPU turn detector with single-speaker and stereo modes')
receipt={'repo':repo,'revision':c.oid,'url':'https://shinohararin-pardon-turn-webgpu.static.hf.space/','files':{str(p.relative_to(site)):{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in site.rglob('*') if p.is_file()}}
(root/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'repo':repo,'revision':c.oid,'files':len(receipt['files'])}),flush=True)
