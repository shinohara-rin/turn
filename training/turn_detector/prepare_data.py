"""Remote-only otoSpeech ingestion. Never run on the host; downloads stay /content."""
from pathlib import Path
import os,sys,json,re,hashlib,concurrent.futures,argparse
import numpy as np
from dataclasses import asdict
ROOT=Path('/content/turn-recreation')
REPO='otoearth/otoSpeech-full-duplex-turn-104h'
REV='46f520297f434edf804389f82f9075a59d2f8268'

def make_split():
    rows=json.loads((ROOT/'metadata.json').read_text())
    actors=sorted({r[f'speaker_{s}_actor_id'] for r in rows for s in (1,2)})
    rng=np.random.default_rng(20261007); rng.shuffle(actors)
    n=len(actors); assignments={a:('train' if i<int(n*.6) else 'dev' if i<int(n*.8) else 'gate') for i,a in enumerate(actors)}
    splits={k:[] for k in ('train','dev','gate','excluded_cross_partition')}
    for r in rows:
        a,b=[assignments[r[f'speaker_{s}_actor_id']] for s in (1,2)]
        splits[a if a==b else 'excluded_cross_partition'].append(r['_dir'])
    for k in splits:splits[k].sort(key=lambda x:hashlib.sha256(('pilot-v1:'+x).encode()).hexdigest())
    out={'seed':20261007,'assignments':assignments,'splits':splits,'revision':REV,'unit':'speaker; discard cross-partition pairs'}
    (ROOT/'split.json').write_text(json.dumps(out,indent=2))
    print('SPLIT',json.dumps({k:len(v) for k,v in splits.items()}),flush=True)
    return out

def download(cid):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    directory=ROOT/'oto'/cid;directory.mkdir(parents=True,exist_ok=True)
    for speaker in (1,2):
        for suffix in ('annotation_a.srt','audio.wav'):
            name=f'speaker_{speaker}_{suffix}'
            if (directory/name).exists():continue
            hf_hub_download(REPO,f'{cid}/{name}',repo_type='dataset',revision=REV,local_dir=ROOT/'oto')
    return cid

def annotations(cid, directory=None):
    import srt
    result={}
    directory=Path(directory) if directory is not None else ROOT/'oto'/cid
    for s in (1,2):
        segments=[]
        for entry in srt.parse((directory/f'speaker_{s}_annotation_a.srt').read_text()):
            m=re.match(r'\[([^]]+)\]\s*(.*)',entry.content,re.S)
            if not m:raise ValueError(f'unknown annotation format {cid}')
            segments.append((entry.start.total_seconds(),entry.end.total_seconds(),m[1],m[2]))
        result[s]=segments
    return result

def gold_and_activity(cid,times,directory=None):
    # Single-annotator training labels use the published floor construction,
    # explicitly without claiming three-annotator benchmark consensus.
    sys.path.insert(0,'/content/turnbench')
    from turnbench.gold import ConsensusEvent,ConsensusViews,build_conversation_events,CANONICAL,TURN_CANONICAL
    ann=annotations(cid,directory=directory)
    fine=[];turn=[];activity=np.zeros((len(times),2),np.float32)
    for s,segs in ann.items():
        for start,end,label,text in segs:
            if label in CANONICAL:fine.append(ConsensusEvent(s,start,end,CANONICAL[label]))
            if label in TURN_CANONICAL:turn.append(ConsensusEvent(s,start,end,'Turn'))
            if label not in ('Awkward Silence','Non-Speech Noise','Channel Bleed'):
                activity[(times>=start)&(times<end),s-1]=1
    events=build_conversation_events(ConsensusViews(turn,[],fine,[]))
    return asdict(events),activity

def main():
    if not Path('/content').exists():raise RuntimeError('Remote Colab only')
    os.environ['HF_TOKEN']=Path('/content/.hf_token').read_text().strip()
    os.environ['HF_HOME']='/content/hf'
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=16);ap.add_argument('--dev',type=int,default=6);args=ap.parse_args()
    split=json.loads((ROOT/'split.json').read_text()) if (ROOT/'split.json').exists() else make_split()
    # Gate audio/labels deliberately remain unopened during search.
    selected=[{'id':cid,'split':part} for part,n in [('train',args.train),('dev',args.dev)] for cid in split['splits'][part][:n]]
    (ROOT/'selection.json').write_text(json.dumps(selected,indent=2))
    for r in selected:
        download(r['id']);print('DOWNLOADED',r['split'],r['id'],flush=True)
    print('DATA_READY',json.dumps({p:sum(r['split']==p for r in selected) for p in ('train','dev')}),flush=True)

if __name__=='__main__':main()
