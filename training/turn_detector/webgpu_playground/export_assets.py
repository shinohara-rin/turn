"""Remote-only deployment packaging and synthetic full-pipeline reference."""
from pathlib import Path
import json,hashlib,shutil,subprocess,sys
import numpy as np,torch,onnxruntime as ort
root=Path('/content/webgpu-playground');out=root/'assets';assert root.is_dir()
torch.set_num_threads(2)
s=torch.load(root/'model_weights.ckpt',map_location='cpu',weights_only=False)
window=s['preprocessor.featurizer.window'];fb=s['preprocessor.featurizer.fb']
(out/'frontend.json').write_text(json.dumps({'sampleRate':16000,'fftSize':512,'hop':160,'window':window.tolist(),'mel':fb.squeeze(0).tolist(),'preemphasis':.97,'logGuard':2**-24,'normalize':'NA','chunkFrames':[9,16],'prefixFrames':9,'decisionSamples':2560}))
h=torch.load(root/'turn_head.pt',map_location='cpu',weights_only=False)
class Head(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(h['dim'],h['hidden']),torch.nn.GELU(),torch.nn.Dropout(.1),torch.nn.Linear(h['hidden'],h['hidden']),torch.nn.GELU(),torch.nn.Linear(h['hidden'],1));self.net.load_state_dict(h['model']);self.register_buffer('mean',torch.as_tensor(h['mean']));self.register_buffer('std',torch.as_tensor(h['std']))
 def forward(self,x):return self.net((x-self.mean)/self.std).squeeze(-1).sigmoid()
head=Head().eval();torch.onnx.export(head,(torch.zeros(2,1042),),str(out/'head.onnx'),input_names=['features'],output_names=['probabilities'],opset_version=17,dynamo=False)
import silero_vad
from silero_vad import load_silero_vad
vad=load_silero_vad().eval();vad.reset_states()
vp=Path(silero_vad.__file__).parent/'data'/'silero_vad.onnx';shutil.copyfile(vp,out/'vad.onnx')
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
vsession=ort.InferenceSession(str(out/'vad.onnx'),opts,providers=['CPUExecutionProvider'])
print('VAD',[(x.name,x.shape) for x in vsession.get_inputs()],[(x.name,x.shape) for x in vsession.get_outputs()],flush=True)
sessions={p:ort.InferenceSession(str(out/(p+'.onnx')),opts,providers=['CPUExecutionProvider']) for p in ['startup','steady']}
hsession=ort.InferenceSession(str(out/'head.onnx'),opts,providers=['CPUExecutionProvider'])
rng=np.random.default_rng(202610082);wave=rng.normal(0,.025,(64000,2)).astype(np.float32);wave[16000:32000,1]=0;wave[48000:]=0
# Synthetic fixture is generated here, not sampled from any dataset.
(out/'check-pcm.bin').write_bytes(wave.T.copy().tobytes())
x=torch.from_numpy(wave.T.copy());pre=torch.cat([x[:,:1],x[:,1:]-.97*x[:,:-1]],1)
raw=torch.log(torch.matmul(fb,torch.stft(pre,n_fft=512,hop_length=160,win_length=400,window=window,center=True,pad_mode='reflect',return_complex=True).abs().pow(2))+2**-24)
cache=[np.zeros((2,17,70,512),np.float32),np.zeros((2,17,512,8),np.float32),np.zeros(2,np.int64)];idx=0;last=np.zeros((2,512),np.float32);state=np.zeros((2,2,128),np.float32);context=np.zeros((2,64),np.float32);vs=[];refs=[];vaderr=[];headerr=[]
sys.path.insert(0,'/content');from heads import causal_history,speaker_features,commit_events
for step in range(25):
 end=(step+1)*2560;chunklen=9 if step==0 else 16;chunkend=idx+chunklen
 if (chunkend-1)*160+257<=end:
  chunk=raw[:,:,max(0,idx-9):chunkend].numpy().copy();v=sessions['startup' if step==0 else 'steady'].run(None,dict(processed_signal=chunk,processed_signal_length=np.full(2,chunk.shape[2],np.int64),cache_last_channel=cache[0],cache_last_time=cache[1],cache_last_channel_len=cache[2]));cache=v[2:];last=np.stack([v[0][i,:,int(v[1][i])-1] for i in range(2)]).astype(np.float16).astype(np.float32);idx+=chunklen
 vals=[]
 for offset in range(end-2560,end,512):
  block=wave[offset:offset+512].T.copy();vp,vsn=vsession.run(None,{'input':np.concatenate([context,block],1),'state':state,'sr':np.array(16000,np.int64)});context=block[:,-64:];state=vsn;native=vad(torch.from_numpy(block),16000).detach().numpy().reshape(2);vaderr.append(float(np.max(abs(native-vp.reshape(2)))));vals.append(vp.reshape(2))
 vs.append(np.array(vals).mean(0));times=np.arange(1,len(vs)+1)*2560/16000;history=causal_history(times,np.asarray(vs));energy=np.log(np.sqrt(np.mean(wave[end-2560:end]**2,axis=0))+1e-7)[None,:,None];features=speaker_features(last[None],history[-1:],energy)[0];prob=hsession.run(None,{'features':features})[0];native=head(torch.from_numpy(features)).detach().numpy();headerr.append(float(np.max(abs(native-prob))))
 refs.append({'step':step,'time':float(end/16000),'vad':vs[-1].tolist(),'energy':energy.reshape(2).tolist(),'encoded':last.tolist(),'probabilities':prob.tolist(),'melProbe':raw[:,[0,32,64,127],:min(4,raw.shape[2])].tolist() if step==0 else None})
assert max(vaderr)<1e-4 and max(headerr)<1e-5
(out/'check.json').write_text(json.dumps({'syntheticOnly':True,'samples':len(wave),'layout':'channel-major float32','steps':refs,'tolerances':{'probability':.003,'vad':.0001,'encodedAbsolute':.003,'encodedRelative':.001},'vadNativeMaxError':max(vaderr),'headNativeMaxError':max(headerr)}))
# Clearly synthetic spoken example, created remotely; no dataset redistribution.
subprocess.run(['bash','-lc','command -v espeak || (apt-get update -qq && apt-get install -y -qq espeak)'],check=True,stdout=subprocess.DEVNULL)
subprocess.run(['espeak','-s','155','-w',str(root/'speech.wav'),'I think we should take the train. Actually, wait a moment. We could go tomorrow instead. That is everything I wanted to say.'],check=True)
import soundfile as sf
from scipy.signal import resample_poly
speech,sr=sf.read(root/'speech.wav',dtype='float32');speech=resample_poly(speech,16000,sr);speech=np.concatenate([np.zeros(8000),speech,np.zeros(48000)]);sf.write(out/'demo.wav',speech,16000,subtype='PCM_16')
manifest={'version':1,'encoderRevision':'d400823b62a3170cb2aaa1c56871848b9622f71d','headSha256':'cf911ffa6c2c21dd38d57f36784f7b948f67e642c1c04a0bdcd14028c24ffa40','precision':'fp32','threshold':0.8214424509124978,'recommitSeconds':1.5,'decisionSamples':2560,'sampleRate':16000,'vadVersion':silero_vad.__version__,'vadNativeMaxError':max(vaderr),'headNativeMaxError':max(headerr),'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in out.iterdir() if p.is_file()}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),flush=True);print('EXPORT_COMPLETE',flush=True)
