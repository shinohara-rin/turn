"""Remote synthetic encoder-only ONNX recurrence qualification (no dataset access).

Export separate fixed-shape startup/steady graphs with explicit batch-first
caches. FP32 ONNX Runtime CPU consumes its OWN previous caches for the entire
synthetic sequence and compares every output/cache/length to native NeMo steps.
Optional FP16 conversion is size-only, NOT browser/GPU numerical qualification.

Requires an already available .nemo base model and exact continued encoder file.
No implicit network download or compute allocation. Model artifacts stay remote.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import time
import numpy as np

INPUTS=('processed_signal','processed_signal_length','cache_last_channel','cache_last_time','cache_last_channel_len')
OUTPUTS=('encoded','encoded_length','cache_last_channel_next','cache_last_time_next','cache_last_channel_len_next')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def externalize_shared(paths,directory):
    """Deduplicate raw initializers across phase graphs in one external data file.

    Keeping two graphs must not silently double the delivered parameter bytes.
    Small typed constants remain inline. This does not imply shared GPU weights
    across two browser sessions; WebGPU runtime memory still needs measurement.
    """
    import onnx
    from onnx.external_data_helper import set_external_data
    directory=Path(directory);directory.mkdir(exist_ok=False)
    offsets={};outputs=[]
    with (directory/'weights.bin').open('wb') as stream:
        for path in paths:
            model=onnx.load(path,load_external_data=True)
            for tensor in model.graph.initializer:
                if not tensor.HasField('raw_data') or len(tensor.raw_data)<1024:continue
                raw=tensor.raw_data;key=(tensor.data_type,tuple(tensor.dims),hashlib.sha256(raw).hexdigest())
                if key not in offsets:
                    offset=stream.tell();stream.write(raw);offsets[key]=(offset,len(raw))
                offset,length=offsets[key]
                set_external_data(tensor,location='weights.bin',offset=offset,length=length)
                tensor.ClearField('raw_data')
            target=directory/path.name
            target.write_bytes(model.SerializeToString());outputs.append(target)
        stream.flush();os.fsync(stream.fileno())
    for path in outputs:onnx.checker.check_model(str(path))
    return outputs


def make_wrapper(encoder,drop):
    import torch
    class StreamingStep(torch.nn.Module):
        def __init__(self):super().__init__();self.encoder=encoder
        def forward(self,processed_signal,processed_signal_length,cache_last_channel,cache_last_time,cache_last_channel_len):
            values=self.encoder.cache_aware_stream_step(
                processed_signal=processed_signal,processed_signal_length=processed_signal_length,
                cache_last_channel=cache_last_channel.transpose(0,1),
                cache_last_time=cache_last_time.transpose(0,1),
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=False,drop_extra_pre_encoded=drop)
            return values[0],values[1],values[2].transpose(0,1),values[3].transpose(0,1),values[4]
    return StreamingStep().eval()


def synthetic_chunks(model,steps,seed):
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    buffer=CacheAwareStreamingAudioBuffer(model,online_normalization=True)
    buffer.preprocessor.eval();f=buffer.preprocessor.featurizer
    if getattr(f,'stft_pad_amount',None) is not None or getattr(f,'frame_splicing',1)!=1:
        raise ValueError('Unsupported preprocessing availability contract')
    if str(getattr(f,'normalize',None)).lower() not in ('none','false','na'):
        raise ValueError('Raw preprocessing must not use whole-utterance normalization')
    f.dither=0.
    generator=torch.Generator().manual_seed(seed)
    samples=round(max(4.,steps*.5+2)*16000)
    wave=torch.randn(2,samples,generator=generator)*.03
    # Distinct channels exercise independent recurrent state, not duplicated input.
    wave[1,samples//3:2*samples//3]=0
    with torch.inference_mode():raw,length=buffer.preprocessor(input_signal=wave,length=torch.full((2,),samples,dtype=torch.long))
    complete=min(int(length.min()),max(0,(samples-1-int(f.n_fft)//2)//int(f.hop_length)+1))
    for channel in range(2):buffer.append_processed_signal(raw[channel:channel+1,:,:complete])
    iterator=iter(buffer);chunks=[];cfg=model.encoder.streaming_cfg
    def step_value(value,step):return int(value[min(step,1)]) if isinstance(value,(list,tuple)) else int(value)
    for step in range(steps):
        if buffer.buffer_idx+step_value(cfg.chunk_size,step)>complete:raise ValueError('Synthetic duration insufficient for requested complete chunks')
        chunk,chunk_length=next(iterator);chunks.append((chunk.detach().clone(),chunk_length.detach().clone()))
    if len({tuple(c[0].shape) for c in chunks[1:]})!=1:raise ValueError('Steady chunk shapes differ; fixed graph invalid')
    return chunks


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--nemo',type=Path,required=True);p.add_argument('--continued',type=Path,required=True)
    p.add_argument('--base-sha256',required=True);p.add_argument('--continued-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--fp16-size',action='store_true');p.add_argument('--atol',type=float,default=1e-4);p.add_argument('--rtol',type=float,default=1e-4)
    a=p.parse_args()
    if not Path('/content').is_dir():raise RuntimeError('Run remotely; no local model download/execution')
    if not 3<=a.steps<=64:raise ValueError('Synthetic screen requires3..64 recurrent steps')
    if sha(a.nemo)!=a.base_sha256:raise ValueError('Base checkpoint identity differs')
    if sha(a.continued)!=a.continued_sha256:raise ValueError('Continued checkpoint identity differs')
    a.out.mkdir(parents=True,exist_ok=False);os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,onnx,onnxruntime as ort
    from nemo.collections.asr.models import ASRModel
    from nemo.core.classes.common import typecheck
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    model=ASRModel.restore_from(str(a.nemo),map_location='cpu').float().eval()
    state=torch.load(a.continued,map_location='cpu',weights_only=False)
    model.encoder.load_state_dict(state['encoder'],strict=True);del state
    for param in model.parameters():param.requires_grad_(False)
    model.preprocessor.featurizer.dither=0.
    encoder=model.encoder;chunks=synthetic_chunks(model,a.steps,a.seed)
    native_initial=encoder.get_initial_cache_state(batch_size=2)
    initial=tuple(c.transpose(0,1).contiguous() if i<2 else c for i,c in enumerate(native_initial))
    drops={'startup':0,'steady':int(encoder.streaming_cfg.drop_extra_pre_encoded)}
    wrappers={k:make_wrapper(encoder,v) for k,v in drops.items()}
    with torch.inference_mode():startup=wrappers['startup'](*chunks[0],*initial)
    examples={'startup':(*chunks[0],*initial),'steady':(*chunks[1],*startup[2:])}
    export_dir=a.out/'raw-fp32';export_dir.mkdir();paths=[]
    for phase in ('startup','steady'):
        path=export_dir/(phase+'.onnx')
        with torch.inference_mode(),typecheck.disable_checks():
            torch.onnx.export(wrappers[phase],examples[phase],str(path),input_names=list(INPUTS),output_names=list(OUTPUTS),
                              opset_version=17,dynamo=False,do_constant_folding=True)
        onnx.checker.check_model(str(path));paths.append(path)
    paths=externalize_shared(paths,a.out/'fp32')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session_started=time.perf_counter()
    sessions={phase:ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']) for phase,path in zip(('startup','steady'),paths)}
    session_seconds=time.perf_counter()-session_started
    for session in sessions.values():
        if [v.name for v in session.get_inputs()]!=list(INPUTS) or [v.name for v in session.get_outputs()]!=list(OUTPUTS):
            raise ValueError('Export dropped/reordered recurrent inputs or outputs')
    native_cache=tuple(v.clone() for v in native_initial);runtime_cache=[v.numpy().copy() for v in initial]
    rows=[]
    with torch.inference_mode():
        for step,(chunk,length) in enumerate(chunks):
            phase='startup' if step==0 else 'steady'
            start=time.perf_counter()
            reference=encoder.cache_aware_stream_step(processed_signal=chunk,processed_signal_length=length,
                cache_last_channel=native_cache[0],cache_last_time=native_cache[1],cache_last_channel_len=native_cache[2],
                keep_all_outputs=False,drop_extra_pre_encoded=drops[phase])
            native_seconds=time.perf_counter()-start;native_cache=reference[2:]
            expected=[v.numpy() for v in reference[:2]]+[v.transpose(0,1).numpy() for v in reference[2:4]]+[reference[4].numpy()]
            feed=dict(zip(INPUTS,[chunk.numpy(),length.numpy(),*runtime_cache]))
            start=time.perf_counter();actual=sessions[phase].run(list(OUTPUTS),feed);runtime_seconds=time.perf_counter()-start
            errors={}
            for name,ref,got in zip(OUTPUTS,expected,actual):
                if np.issubdtype(ref.dtype,np.integer):np.testing.assert_array_equal(got,ref)
                else:np.testing.assert_allclose(got,ref,atol=a.atol,rtol=a.rtol,err_msg=f'{phase} step{step} {name}')
                errors[name]=float(np.max(np.abs(got-ref))) if ref.size else 0.
            runtime_cache=actual[2:] # Independent ONNX recurrence; never substitute native caches.
            rows.append(dict(step=step,phase=phase,max_absolute_errors=errors,native_seconds=native_seconds,onnx_seconds=runtime_seconds,cache_length=actual[4].tolist()))
    result=dict(status='fp32_synthetic_recurrence_passed',synthetic_only=True,browser_webgpu_tested=False,quality_tested=False,
        continued_sha256=a.continued_sha256,base_nemo_sha256=sha(a.nemo),helper_sha256=sha(__file__),steps=rows,
        tolerances=dict(atol=a.atol,rtol=a.rtol),session_creation_seconds=session_seconds,
        steady_onnx_p50_seconds=float(np.median([v['onnx_seconds'] for v in rows[1:]])),
        encoder_parameters=sum(p.numel() for p in encoder.parameters()),
        interfaces={phase:dict(inputs=[dict(name=name,shape=list(v.shape),dtype=str(v.dtype)) for name,v in zip(INPUTS,examples[phase])],drop_extra_pre_encoded=drops[phase]) for phase in examples},
        cache_layout='batch,layers,time,dim and batch,layers,dim,kernel; lengths int64; fixed batch2',
        streaming_config={k:str(v) for k,v in vars(encoder.streaming_cfg).items()},
        installed_sources={name:sha(inspect.getfile(value)) for name,value in [('encoder',type(encoder)),('stream_step',encoder.cache_aware_stream_step)]},
        versions={name:importlib.metadata.version(name) for name in ('torch','nemo_toolkit','onnx','onnxruntime')})
    if a.fp16_size:
        from onnxconverter_common import float16
        raw=a.out/'raw-fp16';raw.mkdir();half=[]
        for path in paths:
            graph=float16.convert_float_to_float16(onnx.load(path),keep_io_types=True)
            out=raw/path.name;onnx.save(graph,out);half.append(out)
        externalize_shared(half,a.out/'fp16')
        result['fp16']=dict(status='converted_for_size_only',parity_tested=False,browser_tested=False,keep_io_types=True)
    result['bundles']={precision:{f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in (a.out/precision).iterdir()} for precision in ('fp32','fp16') if (a.out/precision).exists()}
    (a.out/'qualification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','interfaces','streaming_config','installed_sources')}),flush=True)

if __name__=='__main__':main()
