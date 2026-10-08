"""Batch independent conversations for causal remote feature extraction.

Frozen default is unchanged. For an approved continuation checkpoint:
  python cache_batch.py --encoder-checkpoint /content/vap.pt \
      --output-dir /content/turn-recreation/cache-vap-update100
Training-manifest defaults to the original frozen cache manifest; it must match
the checkpoint's recorded audit exactly. Never reuse a frozen output directory.
"""
from pathlib import Path
from workspace import hf_token
import os,json,time,argparse,hashlib
import numpy as np
from cache_identity import sha256_file,validate_continuation,validate_cache_identity
ROOT=Path('/content/turn-recreation')

def main():
    import torch,soundfile as sf
    from encoder import ParakeetStreamingEncoder,causal_resample
    from vad_batch import causal_vad_batch
    from scale_cache import (validate_selection,verified_drive_directory,cleanup_created_fallback,
                             reuse_verified_cache,VAD_PROTOCOL)
    from prepare_data import download,gold_and_activity,REPO,REV
    os.environ['HF_TOKEN']=hf_token()
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    torch.set_num_threads(2)
    ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--encoder-checkpoint',type=Path)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--selection-manifest',type=Path,default=ROOT/'selection.json')
    ap.add_argument('--reuse-cache-manifest',type=Path)
    ap.add_argument('--policy-cache-manifest',type=Path,default=ROOT/'cache-streaming-v2'/'manifest.json')
    ap.add_argument('--drive-dataset-root',type=Path,default=Path('/content/drive/MyDrive/turn-detector-recreation/datasets/otoearth/otoSpeech-full-duplex-turn-104h'))
    ap.add_argument('--keep-fallback-audio',action='store_true')
    ap.add_argument('--training-manifest',type=Path,default=ROOT/'cache-streaming-v2'/'manifest.json')
    args=ap.parse_args()
    if not Path('/content').is_dir():raise RuntimeError('Remote Colab only')
    if args.batch_size<1:raise ValueError('batch-size must be positive')
    out=args.output_dir if args.output_dir is not None else ROOT/'cache-streaming-v2'
    if args.selection_manifest.resolve()!=(ROOT/'selection.json').resolve() and out.resolve()==(ROOT/'cache-streaming-v2').resolve():
        raise ValueError('Explicit scaling selection requires a distinct output directory')
    records=validate_selection(json.loads(args.selection_manifest.read_text()),ROOT/'split.json')
    if len(records)>22 and out.resolve()==(ROOT/'cache-streaming-v2').resolve():raise ValueError('Expanded selection must preserve pilot output')
    if args.reuse_cache_manifest is not None and args.reuse_cache_manifest.parent.resolve()==out.resolve():raise ValueError('Reuse source must differ from output')
    # Put two dev conversations early so a pilot is possible before all caches.
    train=[r for r in records if r['split']=='train'];dev=[r for r in records if r['split']=='dev']
    selected=train[:4]+dev[:2]+train[4:]+dev[2:]
    continuation=None;identity={};policy_records={}
    if args.encoder_checkpoint is not None:
        if args.output_dir is None or out.resolve() in {(ROOT/'cache-streaming-v2').resolve(),args.training_manifest.parent.resolve()}:
            raise ValueError('Continued encoder requires an explicit output directory distinct from frozen training caches')
        from continue_vap import Recipe,approved_training_records,MODEL_REPO,MODEL_REVISION
        import continue_vap
        _,expected_audit=approved_training_records(args.training_manifest,ROOT/'split.json',Recipe().expected_train_conversations)
        checkpoint_sha=sha256_file(args.encoder_checkpoint)
        continuation=torch.load(args.encoder_checkpoint,map_location='cpu',weights_only=False)
        if sha256_file(args.encoder_checkpoint)!=checkpoint_sha:
            raise ValueError('Continuation checkpoint changed while loading')
        continuation_source=sha256_file(continue_vap.__file__)
        validate_continuation(continuation,expected_audit,continuation_source,MODEL_REPO,MODEL_REVISION)
        # The intervention changes encoder weights only. Reuse exact original
        # VAD, energy and supervision so gate drift cannot confound comparison.
        from leakage_guard import validate_manifest
        policy_manifest=args.policy_cache_manifest
        validate_manifest(policy_manifest,split_path=ROOT/'split.json')
        policy_records={r['id']:r for r in json.loads(policy_manifest.read_text())}
        identity={'encoder_checkpoint_sha256':checkpoint_sha,'encoder_checkpoint_update':continuation['update'],
                  'base_model_repo':MODEL_REPO,'base_model_revision':MODEL_REVISION,
                  'continuation_source_sha256':continuation_source,
                  'continuation_manifest_sha256':expected_audit['manifest_sha256'],
                  'continuation_split_sha256':expected_audit['split_sha256'],
                  'policy_cache_manifest_sha256':sha256_file(policy_manifest),
                  'continuation_recipe_sha256':hashlib.sha256(json.dumps(continuation['recipe'],sort_keys=True).encode()).hexdigest(),
                  'cache_entrypoint_sha256':sha256_file(__file__)}
        # Only encoder weights are needed; release optimizer/head/RNG tensors.
        continuation={'encoder':continuation['encoder']}
    if policy_records and any(r['id'] not in policy_records or policy_records[r['id']]['split']!=r['split'] for r in selected):
        raise ValueError('Policy cache must cover every selected train/dev conversation before continued extraction')
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'encoder-identity.json'
    if marker.exists():
        validate_cache_identity(json.loads(marker.read_text()),identity)
    elif identity and any(out.iterdir()):
        raise ValueError('Refusing continued extraction into an unmarked nonempty directory')
    if identity and not marker.exists():
        temporary=marker.with_suffix('.tmp');temporary.write_text(json.dumps(identity,indent=2));os.replace(temporary,marker)
    selection_audit={'selection_sha256':sha256_file(args.selection_manifest),'split_sha256':sha256_file(ROOT/'split.json'),
                     'counts':{p:sum(r['split']==p for r in records) for p in ('train','dev')},'gate_or_test_selected':False}
    audit_path=out/'selection-audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text())!=selection_audit:raise ValueError('Frozen extraction selection changed')
    audit_path.write_text(json.dumps(selection_audit,indent=2))
    source_hash=sha256_file(Path(__file__).with_name('encoder.py'))
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    if manifest:
        from leakage_guard import validate_manifest
        validate_manifest(manifest_path,split_path=ROOT/'split.json')
        approved={(r['id'],r['split']) for r in selected}
        for r in manifest:
            if (r['id'],r['split']) not in approved:raise ValueError('Cached selection changed')
            provenance=json.loads((out/f"{r['id']}.provenance.json").read_text())
            validate_cache_identity(provenance.get('cache_assembly_identity',provenance),identity)
            if r.get('encoder_checkpoint_sha256')!=identity.get('encoder_checkpoint_sha256'):raise ValueError('Manifest encoder identity mismatch')
            if provenance['encoder_source_sha256']!=source_hash:raise ValueError('Cached encoder code changed')
            for key in ('npz','events'):
                if sha256_file(r[key])!=r[key+'_sha256']:
                    raise ValueError('Cached file checksum mismatch: '+r[key])
    if args.reuse_cache_manifest is not None:
        imported=reuse_verified_cache(args.reuse_cache_manifest,out,selected,identity,source_hash,ROOT/'split.json',policy_records=policy_records)
        existing={r['id']:r for r in manifest}
        for r in imported:
            if r['id'] in existing:
                if any(existing[r['id']][k]!=r[k] for k in ('npz_sha256','events_sha256')):raise ValueError('Reused cache differs from existing output')
            else:manifest.append(r)
        temporary=manifest_path.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2));os.replace(temporary,manifest_path)
    done={r['id'] for r in manifest}
    selected=[r for r in selected if r['id'] not in done]
    if not selected:
        print('CACHE_ALREADY_COMPLETE',len(manifest),flush=True);return
    if continuation is None:
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import ASRModel
        from continue_vap import MODEL_REPO,MODEL_REVISION,MODEL_FILE
        base_path=hf_hub_download(MODEL_REPO,MODEL_FILE,revision=MODEL_REVISION)
        encoder=ParakeetStreamingEncoder(model=ASRModel.restore_from(base_path),device='cuda')
    else:
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import ASRModel
        from continue_vap import MODEL_FILE
        base_path=hf_hub_download(MODEL_REPO,MODEL_FILE,revision=MODEL_REVISION)
        model=ASRModel.restore_from(base_path)
        model.encoder.load_state_dict(continuation['encoder'],strict=True)
        encoder=ParakeetStreamingEncoder(model=model,device='cuda')
        del continuation
        from encoder import test_causality,test_batch_causality
        evidence={'encoder_checkpoint_sha256':checkpoint_sha,
                  'synthetic_only':True,
                  'checks':[test_causality(encoder),test_causality(encoder,sample_rate=24000),
                            test_batch_causality(encoder)]}
        (out/'causality.json').write_text(json.dumps(evidence,indent=2))
        print('CONTINUED_ENCODER_CAUSALITY',json.dumps(evidence),flush=True)
    start_all=time.monotonic()
    for i in range(0,len(selected),args.batch_size):
        group=selected[i:i+args.batch_size];audio=[];meta=[];created_fallback=[];directories=[];start=time.monotonic()
        for r in group:
            cid=r['id'];directory=verified_drive_directory(args.drive_dataset_root,cid)
            if directory is None:
                directory=ROOT/'oto'/cid;existed=directory.exists();download(cid)
                if not existed:created_fallback.append(directory)
            directories.append(directory)
            # Read/resample one conversation at a time to bound raw-wave memory.
            streams=[sf.read(directory/f'speaker_{s}_audio.wav',dtype='float32') for s in (1,2)]
            assert streams[0][1]==streams[1][1] and len(streams[0][0])==len(streams[1][0])
            sr=streams[0][1];x,delay=causal_resample(np.stack([s[0] for s in streams],axis=1),sr)
            x=x[:len(x)//2560*2560];audio.append(x);meta.append((sr,delay));del streams
        sequences=encoder.extract_batch(audio,16000)
        group_vad=causal_vad_batch(audio) if not policy_records else [None]*len(group)
        for r,x,seq,(sr,delay),directory,new_vad in zip(group,audio,sequences,meta,directories,group_vad):
            cid=r['id']
            if policy_records:
                prior=policy_records[cid]
                if prior['split']!=r['split']:raise ValueError('Policy cache split changed')
                for key in ('npz','events'):
                    if sha256_file(prior[key])!=prior[key+'_sha256']:raise ValueError('Policy cache checksum mismatch')
                with np.load(prior['npz'],allow_pickle=False) as cached:
                    if not np.array_equal(seq.available_at_s,cached['times']):raise ValueError('Continued encoder decision times changed')
                    vad=cached['vad'];energy=cached['extras'];activity=cached['annotation_activity']
                events=json.loads(Path(prior['events']).read_text())
            else:
                vad=new_vad
                events,activity=gold_and_activity(cid,seq.available_at_s,directory=directory)
                energy=np.log(np.sqrt((x.reshape(len(vad),2560,2)**2).mean(axis=1))+1e-7)[...,None]
            n=len(vad)
            if len(seq.features)!=n:raise ValueError('Feature/VAD grid mismatch')
            npz=out/f'{cid}.npz';event_path=out/f'{cid}.events.json'
            event_path.write_text(json.dumps(events))
            np.savez_compressed(npz,times=seq.available_at_s,features=seq.features.astype(np.float16),vad=vad,extras=energy,annotation_activity=activity)
            provenance={**seq.metadata,'base_model_repo':MODEL_REPO,'base_model_revision':MODEL_REVISION,**identity,'source':REPO,'revision':REV,'id':cid,'split':r['split'],'source_sr':sr,'source_resample_delay_s':delay,'vad_protocol':VAD_PROTOCOL,'raw_dataset_directory':str(directory),'encoder_source_sha256':source_hash,'labels':'single-annotator otoSpeech; published TurnBench floor builder'}
            (out/f'{cid}.provenance.json').write_text(json.dumps(provenance))
            manifest.append({**r,**({'encoder_checkpoint_sha256':identity['encoder_checkpoint_sha256']} if identity else {}),'source':REPO,'revision':REV,'npz':str(npz),'events':str(event_path),'npz_sha256':sha256_file(npz),'events_sha256':sha256_file(event_path)})
        temporary=manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,indent=2));os.replace(temporary,manifest_path)
        from leakage_guard import validate_manifest
        audit=validate_manifest(out/'manifest.json',split_path=ROOT/'split.json');(out/'leakage-audit.json').write_text(json.dumps(audit,indent=2))
        print(json.dumps({'event':'batch_cached','ids':[r['id'] for r in group],'seconds_audio':sum(len(x)/16000 for x in audio),'elapsed_s':round(time.monotonic()-start,2),'completed':len(manifest),'total':len(selected)+len(done),'elapsed_total_s':round(time.monotonic()-start_all,2)}),flush=True)
        del audio,sequences,group_vad
        if not args.keep_fallback_audio:
            for directory in created_fallback:cleanup_created_fallback(directory)
    print('CACHE_COMPLETE',len(manifest),flush=True)

if __name__=='__main__':main()
