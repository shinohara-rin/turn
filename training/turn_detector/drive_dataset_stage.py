"""CPU Colab: download pinned datasets directly into Google Drive, no local host copy."""
from pathlib import Path
import os,json,time,concurrent.futures
ROOT=Path('/content/drive/MyDrive/turn-detector-recreation/datasets')
REPOS=[('otoearth/otoSpeech-full-duplex-turn-104h','46f520297f434edf804389f82f9075a59d2f8268'),('mundo-ai/turn-benchmark-dev','8fa18a24be51528a45397b35cbcaecd84202062b'),('mundo-ai/turn-benchmark-test','0c12a77a0134449757e4fd4144913bfcff0c376d')]

def main():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive not mounted')
    os.environ['HF_TOKEN']=Path('/content/.hf_token').read_text().strip()
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    from huggingface_hub import HfApi,hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars();ROOT.mkdir(parents=True,exist_ok=True);report=[]
    for repo,revision in REPOS:
        info=HfApi().dataset_info(repo,revision=revision,files_metadata=True)
        entries=[e for e in info.siblings if not e.rfilename.endswith('combined_audio.wav')]
        dest=ROOT/repo;start=time.time();n=0;size=0;total=sum(e.size or 0 for e in entries)
        def download(e):
            for attempt in range(4):
                try:
                    result=Path(hf_hub_download(repo,e.rfilename,repo_type='dataset',revision=revision,local_dir=dest))
                    actual=result.stat().st_size
                    if e.size is not None and actual!=e.size:raise IOError('Size mismatch')
                    return actual
                except Exception:
                    if attempt==3:raise
                    time.sleep(2**attempt)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for future in concurrent.futures.as_completed([pool.submit(download,e) for e in entries]):
                size+=future.result();n+=1
                if n%20==0 or n==len(entries):
                    elapsed=time.time()-start
                    status={'repo':repo,'revision':revision,'files':n,'expected_files':len(entries),'bytes':size,'expected_bytes':total,'elapsed_s':round(elapsed),'mb_per_s':round(size/1e6/max(1,elapsed),2),'remaining_s_estimate':round((total-size)/max(1,size/max(1,elapsed)))}
                    (ROOT/'progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        row={'repo':repo,'revision':revision,'status':'complete','files':n,'bytes':size,'path':str(dest)}
        report.append(row);(ROOT/'staging-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    print('DRIVE_DATASETS_COMPLETE',flush=True)
if __name__=='__main__':main()
