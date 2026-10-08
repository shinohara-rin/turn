"""Remote-only staged approved caches and zero-audio recurrent trajectory."""
from pathlib import Path
import os,json,hashlib,shutil,time,subprocess,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
R=Path('/content/single-speaker');R.mkdir(exist_ok=True)
D=Path('/content/drive/MyDrive/turn-detector-recreation/runs/2026-10-07-pilot')
O=Path('/content/drive/MyDrive/turn-detector-recreation/runs/single-speaker-v1');assert D.is_dir();O.mkdir(exist_ok=True)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def persist(p):
 q=O/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(p)==sha(q)
index=json.loads((D/'backup-index.json').read_text())
manifest='cache-vap-update100-full-v1/manifest.json'
initial=[manifest,'split.json','metadata.json','cache-vap-update100-full-v1/encoder-identity.json','cache-vap-update100-full-v1/causality.json']
receipts={}
def copy(rel):
 src=D/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() and sha(dst)==index[rel]['sha256']:
  receipts[rel]=index[rel]['sha256'];return
 h=sha(src);assert index[rel]['sha256']==h,(rel,'archive mismatch');shutil.copyfile(src,dst);assert sha(dst)==h;receipts[rel]=h
for rel in initial:copy(rel)
sys.path.insert(0,str(R));from leakage_guard import validate_manifest
proof=validate_manifest(R/manifest);assert proof['counts']=={'train':131,'dev':16,'gate':0}
rows=json.loads((R/manifest).read_text());maxsteps=0
for i,row in enumerate(rows):
 assert row['split'] in ('train','dev')
 for key in ('npz','events'):
  original=Path(row[key]);rel=str(original.relative_to('/content/turn-recreation')) if original.is_absolute() else str(Path(manifest).parent/original)
  copy(rel);assert receipts[rel]==row[key+'_sha256'];row[key]=str(R/rel)
 with np.load(R/Path(manifest).parent/row['npz'],allow_pickle=False) as f:maxsteps=max(maxsteps,len(f['times']))
 if i%20==0:print('staged',i,flush=True)
(R/manifest).write_text(json.dumps(rows,indent=2))
(R/'input-receipt.json').write_text(json.dumps({'audit':proof,'files':receipts,'max_steps':maxsteps,'test_accessed':False},indent=2));persist(R/'input-receipt.json')
from huggingface_hub import hf_hub_download
TOKEN=Path('/content/.hf_token').read_text().strip();os.chmod('/content/.hf_token',0o600)
repo='shinohararin/pardon-turn-detector-v2-experiments';rev='d400823b62a3170cb2aaa1c56871848b9622f71d';prefix='browser-compression/v1/export/fp32-dedup/'
m=R/'model';m.mkdir(exist_ok=True)
rep=json.loads(Path(hf_hub_download(repo,prefix+'repack.json',revision=rev,token=TOKEN)).read_text())
for name,v in rep['files'].items():
 src=Path(hf_hub_download(repo,prefix+name,revision=rev,token=TOKEN));assert sha(src)==v['sha256'];shutil.copyfile(src,m/name)
h=Path(hf_hub_download('shinohararin/pardon-turn-detector-v1','turn_head.pt',revision='27622509159f24a99abc182593f5d624fafb6e70',token=TOKEN));assert sha(h)=='cf911ffa6c2c21dd38d57f36784f7b948f67e642c1c04a0bdcd14028c24ffa40';shutil.copyfile(h,R/'baseline.pt')
# Verify the complete zero-waveform frontend algebra using actual PCM zeros.
# Mel weights do not affect all-zero STFT power; the log guard is pinned.
import torch
pcm=torch.zeros(2,5120);pre=torch.cat([pcm[:,:1],pcm[:,1:]-.97*pcm[:,:-1]],1)
power=torch.stft(pre,n_fft=512,hop_length=160,win_length=400,window=torch.hann_window(400),center=True,pad_mode='reflect',return_complex=True).abs().pow(2)
assert torch.count_nonzero(power)==0
zero_mel=np.float32(torch.log(torch.tensor(2**-24)).item())
import onnxruntime as ort,torch
from silero_vad import load_silero_vad
from heads import causal_history
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
a=ort.InferenceSession(str(m/'startup.onnx'),opts,providers=['CPUExecutionProvider']);b=ort.InferenceSession(str(m/'steady.onnx'),opts,providers=['CPUExecutionProvider'])
v=load_silero_vad().eval();v.reset_states();torch.set_num_threads(2)
cache=[np.zeros((2,17,70,512),np.float32),np.zeros((2,17,512,8),np.float32),np.zeros(2,np.int64)]
emb=[];vad=[];fixed=None;consecutive=0;start=time.monotonic()
for step in range(maxsteps):
 n=9 if step==0 else 25
 values=(a if step==0 else b).run(None,dict(processed_signal=np.full((2,128,n),zero_mel,np.float32),processed_signal_length=np.full(2,n,np.int64),cache_last_channel=cache[0],cache_last_time=cache[1],cache_last_channel_len=cache[2]))
 same=all(np.array_equal(x,y) for x,y in zip(cache,values[2:]));cache=values[2:]
 z=np.stack([values[0][k,:,int(values[1][k])-1] for k in range(2)]);assert np.array_equal(z[0],z[1]);emb.append(z[0].astype(np.float16))
 with torch.no_grad():pv=[float(v(torch.zeros(2,512),16000)[0]) for _ in range(5)]
 vad.append(np.float32(np.mean(pv)))
 # Require complete encoder state and native VAD states to settle; last 32
 # decision means identical is an additional check, not a substitute for state.
 vh=hashlib.sha256(v._state.numpy().tobytes()+v._context.numpy().tobytes()).hexdigest()
 consecutive=consecutive+1 if same and step and vh==lastvh else 0;lastvh=vh
 if step%50==0:print('silence',step,'of',maxsteps,'seconds',round(time.monotonic()-start,1),'fixed',consecutive,flush=True)
 if consecutive>=32:
  fixed=step;break
if fixed is not None:
 emb.extend([emb[-1]]*(maxsteps-len(emb)));vad.extend([vad[-1]]*(maxsteps-len(vad)))
features=np.asarray(emb);vp=np.asarray(vad);times=np.arange(1,maxsteps+1)*.16;hist=causal_history(times,np.repeat(vp[:,None],2,1))[:,0];other=np.concatenate([features.astype(np.float32),hist,np.full((maxsteps,1),np.log(1e-7),np.float32)],1)
np.save(R/'silence-other.npy',other);persist(R/'silence-other.npy')
(R/'silence-receipt.json').write_text(json.dumps({'steps':maxsteps,'computed_steps':step+1,'fixed_point_verified_step':fixed,'seconds':time.monotonic()-start,'sha256':sha(R/'silence-other.npy'),'encoder_revision':rev,'runtime':ort.__version__,'silero_state_checked':True,'frontend_zero_mel':float(np.float32(np.log(2**-24))),'embedding_rounding':'float16 then float32','source_sha256':sha(__file__)},indent=2));persist(R/'silence-receipt.json')
subprocess.run(['git','clone','-q','https://github.com/sesameailabs/turnbench.git','/content/single-speaker/turnbench'],check=True)
subprocess.run(['git','-C','/content/single-speaker/turnbench','checkout','-q','38a6f874322430cb3ca71d8a52aa1e636e88bad8'],check=True)
(R/'PREPARED').write_text('complete');print('PREPARED',flush=True)
