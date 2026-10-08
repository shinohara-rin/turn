"""Causal full-duration phase arrays and conservative cached-prefix replay check."""
import numpy as np


def native_silero_all(audio,detector):
    """All complete32ms frames, including partial160ms tail. No final padding."""
    import torch
    audio=np.asarray(audio,np.float32)
    if audio.ndim!=2 or audio.shape[1]!=2 or not np.isfinite(audio).all():raise ValueError('Expected finite stereo16k audio')
    detector.eval();detector.reset_states();rows=[]
    with torch.inference_mode():
        for start in range(0,len(audio)-511,512):
            rows.append(detector(torch.from_numpy(audio[start:start+512].T.copy()),16000).cpu().numpy().reshape(2))
    return np.asarray(rows,np.float32).reshape(-1,2)


def inputs_at_decisions(audio,raw,decisions):
    requested=np.asarray(decisions)
    ends=requested.astype(np.int64)
    if requested.ndim!=1 or np.any(requested!=ends) or not np.array_equal(ends,np.arange(2048,len(audio)+1,2560)):
        raise ValueError('Not a complete valid128-phase decision grid')
    if raw.shape!=(len(audio)//512,2):raise ValueError('Native32ms coverage incomplete')
    vad=[];energy=[]
    for end in ends:
        right=end//512
        vad.append(raw[max(0,right-5):right].mean(0))
        window=audio[max(0,end-2560):end]
        energy.append(np.log(np.sqrt((window**2).mean(0))+1e-7))
    return np.asarray(vad,np.float32).reshape(-1,2),np.asarray(energy,np.float32).reshape(-1,2,1)


def verify_replay_prefix(cached,fresh):
    """Same1e-3 abs/relative allowance as existing encoder batch qualification.

    Cached values are float16; fresh values are float32. No tolerance fitting or
    silent replacement of cached rows. Return error evidence, fail on mismatch.
    """
    if cached.dtype!=np.float16 or fresh.shape!=cached.shape:raise ValueError('Replay prefix shape/dtype differs')
    np.testing.assert_allclose(fresh,cached.astype(np.float32),atol=1e-3,rtol=1e-3)
    delta=float(np.max(abs(fresh-cached.astype(np.float32)))) if cached.size else 0.
    return {'passed':True,'atol':1e-3,'rtol':1e-3,'max_absolute_delta':delta,
            'reference':'cached float16 vs fresh float32; same existing batch-equivalence tolerance'}


def qualify_phase_encoder(encoder,*,batch_size=4):
    """Remote synthetic full-tail extraction, same-chunk reuse and causality."""
    if batch_size<1:raise ValueError('Qualification batch size must be positive')
    rng=np.random.default_rng(128147)
    audio=rng.normal(0,.03,(2560*12+2048,2)).astype(np.float32)
    old_count=len(audio)//2560
    old=encoder.extract(audio[:old_count*2560])
    ends=np.arange(2048,len(audio)+1,2560,dtype=np.int64)
    full=encoder.extract(audio,decision_samples=ends)
    np.testing.assert_array_equal(full.available_at_s,ends/16000)
    parity=verify_replay_prefix(old.features.astype(np.float16),full.features[:old_count])
    cutoff=2560*6
    changed=audio.copy();changed[cutoff:]=rng.uniform(-1.064,1.064,changed[cutoff:].shape)
    mutated=encoder.extract(changed,decision_samples=ends)
    mask=ends<=cutoff
    np.testing.assert_allclose(full.features[mask],mutated.features[mask],atol=1e-5,rtol=1e-5)
    truncated=encoder.extract(audio[:cutoff],decision_samples=ends[mask])
    np.testing.assert_allclose(full.features[mask],truncated.features,atol=1e-5,rtol=1e-5)
    batch_audio=[audio];batch_ends=[ends];independent=[full]
    for i in range(1,batch_size):
        longer=np.concatenate((audio,rng.normal(0,.03,(2560*i,2)).astype(np.float32)))
        longer_ends=np.arange(2048,len(longer)+1,2560,dtype=np.int64)
        batch_audio.append(longer);batch_ends.append(longer_ends)
        independent.append(encoder.extract(longer,decision_samples=longer_ends))
    batched=encoder.extract_batch(batch_audio,decision_samples=batch_ends)
    for reference,actual in zip(independent,batched):
        np.testing.assert_array_equal(reference.available_at_s,actual.available_at_s)
        np.testing.assert_allclose(reference.features,actual.features,atol=1e-3,rtol=1e-3)
    if len(full.features)!=old_count+1:raise AssertionError('Synthetic missingtail was not actually extracted')
    return {'passed':True,'synthetic_only':True,'checks':['missing_tail','prefix_reuse','suffix_mutation','truncation','exact_grid','unequal_length_batch_parity'],
            'qualified_batch_size':batch_size,'prefix_parity':parity,'phase_grid_contract':'phase128-full-v1'}


def qualify_native_silero(detector):
    """Actual staged Silero model: old native parity plus new-tail causality."""
    import torch
    from vad_batch import causal_vad_batch
    rng=np.random.default_rng(328)
    audio=rng.normal(0,.03,(2560*20+2048+31,2)).astype(np.float32)
    audio[1000,0]=1.064;audio[1200,1]=-1.064
    full=native_silero_all(audio,detector)
    caller_threads=torch.get_num_threads()
    try:legacy=causal_vad_batch([audio],detector=detector,reduction='raw')[0]
    finally:torch.set_num_threads(caller_threads)
    np.testing.assert_allclose(full[:len(legacy)],legacy,atol=1e-6,rtol=1e-5)
    cutoff=2560*6
    changed=audio.copy();changed[cutoff:]=rng.uniform(-1.064,1.064,changed[cutoff:].shape)
    np.testing.assert_allclose(full[:cutoff//512],native_silero_all(changed,detector)[:cutoff//512],atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(full[:cutoff//512],native_silero_all(audio[:cutoff],detector),atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(full,native_silero_all(audio[:-31],detector),atol=1e-6,rtol=1e-5)
    if len(full)!=len(legacy)+4:raise AssertionError('Synthetic native tail missing')
    return {'passed':True,'synthetic_only':True,'checks':['legacy_prefix_parity','suffix_mutation','truncation','no_padding','native_tail','reset'],
            'max_prefix_delta':float(np.max(abs(full[:len(legacy)]-legacy))),'phase_grid_contract':'phase128-full-v1'}
