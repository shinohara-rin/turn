"""Remote feature materialization; datasets, waveforms and features stay on Colab."""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
ROOT=Path('/content/turn-recreation')

def causal_vad(audio):
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(2)
    detector=load_silero_vad().eval()
    detector.reset_states()
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(audio)-511,512):
            p=detector(torch.from_numpy(audio[start:start+512].T.copy()),16000)
            rows.append(p.cpu().numpy().reshape(2))
    raw=np.asarray(rows,dtype=np.float32)
    # Five causal 32ms frames per decision. Gate averages recent evidence.
    t=len(raw)//5
    return raw[:t*5].reshape(t,5,2).mean(axis=1)

def main():
    import soundfile as sf,torch
    from encoder import ParakeetStreamingEncoder,ParakeetPrefixEncoder,causal_resample
    from prepare_data import download,gold_and_activity
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['streaming','prefix'],default='streaming');ap.add_argument('--limit',type=int);args=ap.parse_args()
    if not ROOT.exists():raise RuntimeError('Remote Colab only')
    os.environ['HF_TOKEN']=Path('/content/.hf_token').read_text().strip()
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    torch.set_num_threads(2)
    out=ROOT/f'cache-{args.method}';out.mkdir(exist_ok=True)
    selected=json.loads((ROOT/'selection.json').read_text())
    if args.limit:selected=selected[:args.limit]
    encoder=(ParakeetStreamingEncoder if args.method=='streaming' else ParakeetPrefixEncoder)(device='cuda',batch_size=32)
    manifest=[]
    start_all=time.monotonic()
    for record in selected:
        cid=record['id'];npz=out/f'{cid}.npz';events_path=out/f'{cid}.events.json'
        if not npz.exists():
            start=time.monotonic(); download(cid)
            streams=[sf.read(ROOT/'oto'/cid/f'speaker_{s}_audio.wav',dtype='float32') for s in (1,2)]
            assert streams[0][1]==streams[1][1] and len(streams[0][0])==len(streams[1][0])
            audio=np.stack([s[0] for s in streams],axis=1); sr=streams[0][1]
            x,delay=causal_resample(audio,sr);del audio,streams
            n=len(x)//2560;x=x[:n*2560]
            seq=encoder.extract(x,16000)
            vad=causal_vad(x)
            assert len(vad)==len(seq.features)
            events,activity=gold_and_activity(cid,seq.available_at_s)
            energy=np.log(np.sqrt((x.reshape(n,2560,2)**2).mean(axis=1))+1e-7)[...,None]
            events_path.write_text(json.dumps(events))
            np.savez_compressed(npz,times=seq.available_at_s,features=seq.features.astype(np.float16),vad=vad,extras=energy,annotation_activity=activity)
            (out/f'{cid}.provenance.json').write_text(json.dumps({**seq.metadata,'source_sr':sr,'source_resample_delay_s':delay,'labels':'single-annotator otoSpeech; published TurnBench floor builder'},indent=2))
            print(json.dumps({'event':'cached','id':cid,'split':record['split'],'duration_s':len(x)/16000,'frames':n,'elapsed_s':round(time.monotonic()-start,2),'total_elapsed_s':round(time.monotonic()-start_all,2)}),flush=True)
        manifest.append({**record,'source':'otoearth/otoSpeech-full-duplex-turn-104h','revision':'46f520297f434edf804389f82f9075a59d2f8268','npz':str(npz),'events':str(events_path)})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('CACHE_COMPLETE',len(manifest),flush=True)

if __name__=='__main__':main()
