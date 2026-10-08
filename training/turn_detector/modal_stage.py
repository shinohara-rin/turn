"""Stage HF datasets directly into persistent Modal Volume, CPU only.
Run: uv tool run --from modal modal run --detach modal_stage.py
No dataset content traverses the local machine. HF credentials use ephemeral Secret.
"""
import os
from pathlib import Path
import modal
app=modal.App('turnbench-dataset-staging')
volume=modal.Volume.from_name('turnbench-datasets',create_if_missing=True)
if modal.is_local():
    token=os.environ.get('HF_TOKEN') or (Path.home()/'.cache/huggingface/token').read_text().strip()
    secret=modal.Secret.from_dict({'HF_TOKEN':token})
else:
    secret=modal.Secret.from_dict({})
image=modal.Image.debian_slim(python_version='3.11').pip_install('huggingface_hub==1.17.0','hf_xet')
@app.function(image=image,cpu=2,memory=4096,timeout=86400,volumes={'/datasets':volume},secrets=[secret])
def stage():
    import json,time
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from huggingface_hub import HfApi,hf_hub_download,get_hf_file_metadata,hf_hub_url
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    root=Path('/datasets'); api=HfApi(); report=[]
    repos=[('otoearth/otoSpeech-full-duplex-turn-104h','46f520297f434edf804389f82f9075a59d2f8268'),('mundo-ai/turn-benchmark-dev','8fa18a24be51528a45397b35cbcaecd84202062b'),('mundo-ai/turn-benchmark-test',None)]
    for repo,revision in repos:
        started=time.time(); dest=root/repo
        try:
            info=api.dataset_info(repo,revision=revision,files_metadata=True);rev=info.sha
            # Separated channels preserve all audio and avoid duplicate combined rendition.
            entries=[x for x in info.siblings if not x.rfilename.endswith('combined_audio.wav')]
            probe=next(x.rfilename for x in entries if x.rfilename.endswith(('.wav','.parquet')))
            get_hf_file_metadata(hf_hub_url(repo,probe,repo_type='dataset',revision=rev),token=os.environ['HF_TOKEN'])
            total=sum(x.size or 0 for x in entries);completed=0;size=0
            def download(entry):
                path=hf_hub_download(repo,entry.rfilename,repo_type='dataset',revision=rev,local_dir=dest)
                actual=Path(path).stat().st_size
                if entry.size is not None and actual!=entry.size:raise RuntimeError('file size mismatch')
                return actual
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs=[pool.submit(download,e) for e in entries]
                for job in as_completed(jobs):
                    size+=job.result();completed+=1
                    if completed%40==0 or completed==len(entries):
                        progress={'repo':repo,'revision':rev,'completed':completed,'files':len(entries),'bytes':size,'expected_bytes':total,'elapsed_s':round(time.time()-started)}
                        (root/'progress.json').write_text(json.dumps(progress,indent=2));volume.commit();print(json.dumps(progress),flush=True)
            result={'repo':repo,'revision':rev,'status':'complete','files':completed,'bytes':size,'path':str(dest),'excluded':['*/combined_audio.wav'] if 'otoSpeech' in repo else []}
        except Exception as e:
            result={'repo':repo,'status':'failed','error_type':type(e).__name__,'http_status':getattr(getattr(e,'response',None),'status_code',None)}
        report.append(result);(root/'staging-report.json').write_text(json.dumps(report,indent=2));volume.commit();print(json.dumps(result),flush=True)
    return report
@app.local_entrypoint()
def main():
    print(stage.remote())
