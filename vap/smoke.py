import time, torch, soundfile as sf, numpy as np
from baselines.vap.predict import _load_model, _step_extraction
torch.set_num_threads(4)
m=_load_model("pretrained","cpu")
w,sr=sf.read("baselines/vap/VoiceActivityProjection/example/student_long_female_en-US-Wavenet-G.wav",dtype="float32")
if w.ndim==2: a,b=w[:,0],w[:,1]
else: a,b=w,np.zeros_like(w)
print("sr",sr,"dur",len(a)/sr)
# tile to 180s with alternating speakers
seg=len(a); n=int(180*sr)//seg+1
A=np.concatenate([a if i%2==0 else np.zeros_like(a) for i in range(n)])[:int(180*sr)]
B=np.concatenate([b if i%2==0 else a for i in range(n)])[:int(180*sr)]
x=torch.from_numpy(np.stack([A,B]))[None]
t=time.time(); out=_step_extraction(x,m,"cpu"); dt=time.time()-t
print("frames",out["p_now"].shape,"rt factor",180/dt)
p=out["p_now"][0].numpy(); print("p_now mean per spk",p.mean(0))
