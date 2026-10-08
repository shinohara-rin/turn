"""Periodic, verified copy of completed research artifacts to mounted Drive."""
from pathlib import Path
import os,time,json,hashlib,shutil,traceback
SOURCE=Path('/content/turn-recreation')
DEST=Path('/content/drive/MyDrive/turn-detector-recreation/runs/2026-10-07-pilot')
ALLOW_ROOT={'.py','.json','.md','.log'}
ALLOW_DIR={'cache-streaming-v2','cache-vap-update100','cache-silero-last-v1','cache-streaming-full-v1','cache-vap-update100-full-v1','cache-phase128-baseline-v1','cache-phase128-v1','cache-phase128-full-v1','cache-phase128-full-batched-v1','runs','.arbor'}

def backup_once():
    if not Path('/content/drive/MyDrive').is_dir():raise RuntimeError('Drive is not mounted; do not write a fake local backup')
    DEST.mkdir(parents=True,exist_ok=True)
    index_path=DEST/'backup-index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {}
    copied=0;size=0;now=time.time()
    candidates=[]
    for f in SOURCE.iterdir():
        if f.is_file() and f.suffix in ALLOW_ROOT:candidates.append(f)
        elif f.is_dir() and f.name in ALLOW_DIR:candidates.extend(x for x in f.rglob('*') if x.is_file())
    candidates.sort(key=lambda f:(f.suffix!='.pt',str(f))) # checkpoints first
    for src in candidates:
        if src.name.endswith(('.tmp','.partial','.pid')) or '__pycache__' in src.parts:continue
        before=src.stat()
        if now-before.st_mtime<3:continue # never copy actively-written checkpoint
        rel=str(src.relative_to(SOURCE));key=f'{before.st_size}:{before.st_mtime_ns}'
        if index.get(rel,{}).get('version')==key:continue
        dst=DEST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_name(dst.name+'.partial')
        shutil.copyfile(src,tmp)
        after=src.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            tmp.unlink(missing_ok=True);continue
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if hashlib.sha256(tmp.read_bytes()).hexdigest()!=digest:raise IOError('Backup checksum mismatch: '+rel)
        os.replace(tmp,dst);index[rel]={'version':key,'sha256':digest,'bytes':before.st_size,'verified_at':time.time()}
        copied+=1;size+=before.st_size
    tmp=index_path.with_suffix('.tmp');tmp.write_text(json.dumps(index,indent=2));os.replace(tmp,index_path)
    status={'time':time.time(),'files_copied':copied,'bytes_copied':size,'total_files':len(index),'checkpoints':sum(k.endswith('.pt') for k in index),'destination':str(DEST),'verified_by':'sha256 readback'}
    (SOURCE/'backup-status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)

if __name__=='__main__':
    while True:
        try:backup_once()
        except Exception:traceback.print_exc()
        time.sleep(30)
