"""Bounded matched continuation versus missing-other-speaker adaptation."""
from pathlib import Path
import os,sys,time,json,hashlib,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, torch
import torch.nn.functional as F
R=Path('/content/single-speaker');assert (R/'PREPARED').exists();sys.path.insert(0,str(R));sys.path.insert(0,str(R/'turnbench'))
from heads import causal_history,speaker_features,eot_targets,make_model,commit_events
from leakage_guard import validate_manifest
from evaluate_phase128_pair import verify_upstream,matched_score,summarize,paired_matches
from turnbench import score as api
from turnbench.gold import AnchorEvent,Interval
api.AnchorEvent=AnchorEvent;api.Interval=Interval
O=R/'results';O.mkdir(exist_ok=True);D=Path('/content/drive/MyDrive/turn-detector-recreation/runs/single-speaker-v1/results');assert D.parent.is_dir();D.mkdir(exist_ok=True)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def save_json(name,v):
 p=O/name;p.write_text(json.dumps(v,indent=2));persist(p)
def persist(p):
 q=D/p.name;shutil.copyfile(p,q);assert sha(p)==sha(q)
for source in ('prepare.py','run.py','PROTOCOL.md','input-receipt.json','silence-receipt.json','heads.py','leakage_guard.py','evaluate_phase128_pair.py'):
 shutil.copyfile(R/source,O/source);persist(O/source)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
manifest=R/'cache-vap-update100-full-v1/manifest.json';audit=validate_manifest(manifest);assert audit['counts']=={'train':131,'dev':16,'gate':0};scorer=verify_upstream(R/'turnbench')
base=torch.load(R/'baseline.pt',map_location='cpu',weights_only=False);assert sha(R/'baseline.pt')=='cf911ffa6c2c21dd38d57f36784f7b948f67e642c1c04a0bdcd14028c24ffa40';assert base['dim']==1042 and base['hidden']==128 and not base['auxiliary']
mean,std=np.asarray(base['mean'],np.float32),np.asarray(base['std'],np.float32)
silence=np.load(R/'silence-other.npy');assert silence.shape[1]==521
rows=json.loads(manifest.read_text());offset=0;training=[];dev=[]
for row in rows:
 assert row['encoder_checkpoint_sha256']=='2cd76b022820e769ee95c6da08d075f5c30eadf70b0b414ff92a3c6c5c89489f'
 for key in ('npz','events'):assert sha(row[key])==row[key+'_sha256'], 'Input changed after staging'
assert json.loads((R/'silence-receipt.json').read_text())['encoder_revision']=='d400823b62a3170cb2aaa1c56871848b9622f71d'
# Only metadata and one approved record at a time are materialized in RAM.
for row in rows:
 with np.load(manifest.parent/row['npz'],allow_pickle=False) as z:n=len(z['times'])*2
 if row['split']=='train':training.append((row,offset,n));offset+=n
x=np.lib.format.open_memmap(R/'train-x.npy',mode='w+',dtype='float32',shape=(offset,1042));y=np.zeros(offset,np.float32);w=np.zeros(offset,np.float32);stepids=np.zeros(offset,np.int32)
for row,start,n in training:
 with np.load(manifest.parent/row['npz'],allow_pickle=False) as z:
  events=json.loads((manifest.parent/row['events']).read_text());t,v=z['times'],z['vad'];xx=speaker_features(z['features'],causal_history(t,v),z['extras']);yy,ww=eot_targets(t,v,events)
  assert np.allclose(t,np.arange(1,len(t)+1)*.16,atol=.002)
  x[start:start+n]=xx.reshape(-1,1042);y[start:start+n]=yy.reshape(-1);w[start:start+n]=ww.reshape(-1);stepids[start:start+n]=np.repeat(np.arange(len(t)),2)
x.flush();w/=max(float(w.mean()),1e-8)
for row in rows:
 if row['split']!='dev':continue
 with np.load(manifest.parent/row['npz'],allow_pickle=False) as z:
  t,v=z['times'],z['vad'];xx=speaker_features(z['features'],causal_history(t,v),z['extras']);events=json.loads((manifest.parent/row['events']).read_text())
  dev.append(dict(id=row['id'],times=t,vad=v,x=xx,events=events))
# Mono views are evaluated simultaneously for convenience; both retain their own
# own-channel streams. Each row's other half is the independent silence stream.
def predict(model,mono):
 model.eval();probs=[]
 with torch.no_grad():
  for r in dev:
   a=r['x'].reshape(-1,1042);parts=[]
   for start in range(0,len(a),4096):
    b=a[start:start+4096].copy()
    if mono:b[:,521:]=silence[(np.arange(start,start+len(b))//2)]
    parts.append(model(torch.from_numpy((b-mean)/std))[:,0].sigmoid().numpy())
   probs.append(np.concatenate(parts).reshape(-1,2))
 return probs
THRESHOLD=.8214424509124978
allmatches={};scores={}
def evaluate(name,model):
 for mode in ('stereo','mono'):
  probs=predict(model,mode=='mono');total=api.TaskScore();matches=[]
  for record,p in zip(dev,probs):
   pred=commit_events(record['times'],record['vad'],p,THRESHOLD,1.5);s,m=matched_score(record['events'],pred,api);api.merge(total,s);matches.append(m)
  scores[name+'/'+mode]=summarize(total);allmatches[name+'/'+mode]=matches
  print(name,mode,json.dumps(scores[name+'/'+mode]),flush=True)
 save_json('scores-progress.json',scores)
def model():
 m=make_model(1042,128,False);m.load_state_dict(base['model']);return m
baseline=model();evaluate('frozen',baseline)
assert {k:scores['frozen/stereo'][k] for k in ('tp','fn','fp','tn')}=={'tp':958,'fn':73,'fp':129,'tn':1801}, 'Historical baseline replay mismatch; stop for diagnosis'
start=time.monotonic();effort=[]
for seed in (17,42):
 for arm in ('control','mono'):
  name=f'{arm}-{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed);m=model();m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01);losses=[];t0=time.monotonic()
  for update in range(1,201):
   if time.monotonic()-start>1200:raise RuntimeError('Registered 20-minute training/evaluation budget exhausted')
   index=rng.integers(0,len(x),2048);b=x[index].copy()
   if arm=='mono':b[:,521:]=silence[stepids[index]]
   z=m(torch.from_numpy((b-mean)/std))[:,0];loss=(F.binary_cross_entropy_with_logits(z,torch.from_numpy(y[index]),reduction='none')*torch.from_numpy(w[index])).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
   if update%50==0:
    p=O/f'{name}-u{update}.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),mean=mean,std=std,dim=1042,hidden=128,auxiliary=False,seed=seed,arm=arm,update=update,baseline_sha256=sha(R/'baseline.pt'),rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),operating_point={'threshold':THRESHOLD,'recommit_s':1.5}),p);persist(p);print('checkpoint',name,update,'seconds',round(time.monotonic()-t0,2),flush=True)
  effort.append(dict(name=name,updates=200,seconds=time.monotonic()-t0,mean_loss=float(np.mean(losses))))
  evaluate(name,m)
pairs={}
for seed in (17,42):
 for ref in ('frozen',f'control-{seed}'):
  for mode in ('mono','stereo'):
   a=allmatches[ref+'/'+mode];b=allmatches[f'mono-{seed}/'+mode];p=[paired_matches(aa,bb) for aa,bb in zip(a,b)];gains=[v['gain_ms'] for pp in p for v in pp['anchors'] if v['gain_ms'] is not None];pairs[f'mono-{seed}_vs_{ref}/{mode}']={k:sum(pp[k] for pp in p) for k in ('common_detected','gained','lost','missed_both')};pairs[f'mono-{seed}_vs_{ref}/{mode}'].update(common_latency_gain_ms_median=float(np.median(gains)) if gains else None,per_conversation=[{k:v for k,v in pp.items() if k!='anchors'} for pp in p])
report={'scope':'Exploratory original16dev missing-interlocutor screen; no test or consumed-gate access','audit':audit,'scorer':scorer,'source_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL.md'),'versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},'normalizer_sha256':hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest(),'baseline_sha256':sha(R/'baseline.pt'),'silence':json.loads((R/'silence-receipt.json').read_text()),'policy':{'threshold':THRESHOLD,'recommit_s':1.5,'gate':'mean','phase_ms':160},'updates_per_arm':200,'training_rows':len(x),'normalizer':'frozen selected baseline normalizer','scores':scores,'pairs':pairs,'effort':effort,'automatic_promotion':False}
save_json('report.json',report)
from huggingface_hub import HfApi,hf_hub_download
api_hf=HfApi(token=Path('/content/.hf_token').read_text().strip());repo='shinohararin/pardon-turn-detector-v2-experiments';assert api_hf.repo_info(repo).private
c=api_hf.upload_folder(repo_id=repo,folder_path=str(O),path_in_repo='single-speaker/v1',commit_message='Archive bounded mono adaptation versus matched continuation controls')
files={p.name:sha(p) for p in O.iterdir() if p.is_file()}
for name,h in files.items():assert sha(hf_hub_download(repo,'single-speaker/v1/'+name,revision=c.oid,token=api_hf.token))==h
save_json('archive.json',{'repo':repo,'revision':c.oid,'verified_files':files,'private':True})
print('COMPLETE',flush=True)
