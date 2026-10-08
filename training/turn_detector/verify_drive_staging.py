"""Remote independent Drive inventory check; never decodes data or labels."""
from pathlib import Path
from workspace import hf_token
import json,os,time
from drive_dataset_stage import ROOT,REPOS

def main():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Mounted remote Drive required')
    os.environ['HF_TOKEN']=hf_token()
    from huggingface_hub import HfApi
    report=json.loads((ROOT/'staging-report.json').read_text())
    if len(report)!=len(REPOS):raise ValueError('Staging incomplete')
    rows=[]
    for repo,revision in REPOS:
        declared=next(r for r in report if r['repo']==repo)
        if declared['status']!='complete' or declared['revision']!=revision:raise ValueError('Staging identity mismatch')
        entries=[e for e in HfApi().dataset_info(repo,revision=revision,files_metadata=True).siblings if not e.rfilename.endswith('combined_audio.wav')]
        total=0
        for entry in entries:
            size=(ROOT/repo/entry.rfilename).stat().st_size
            if size!=entry.size:raise ValueError('Persisted size mismatch: '+repo+'/'+entry.rfilename)
            total+=size
        if len(entries)!=declared['files'] or total!=declared['bytes']:raise ValueError('Report inventory mismatch')
        rows.append(dict(repo=repo,revision=revision,files=len(entries),bytes=total,verified='independent mounted Drive file inventory and upstream sizes',contents_decoded=False))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(checked_at=time.time(),datasets=rows,test_contents_used=False,full_content_hash_check=False)
    Path('/content/turn-recreation/drive-staging-verification.json').write_text(json.dumps(result,indent=2))
    print('DRIVE_INVENTORY_VERIFIED',flush=True)
if __name__=='__main__':main()
