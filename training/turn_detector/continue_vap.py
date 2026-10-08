"""Experimental actual encoder continuation. Remote Colab only; no benchmark input.

Uses NeMo serving chunks, fixed eval-mode normalization, TBPTT=1, and gradients
only in the final two encoder layers plus a joint VAP head. Defaults are a
bounded pilot, not recovered Ooma hyperparameters. Run --synthetic-smoke first.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn

MODEL_REPO = 'nvidia/parakeet_realtime_eou_120m-v1'
MODEL_REVISION = 'a7e2b4629593dce0ec19f600e00e9904353fda2d'
MODEL_FILE = 'parakeet_realtime_eou_120m-v1.nemo'


@dataclass(frozen=True)
class Recipe:
    seed: int = 20261007
    batch_size: int = 4
    warmup_steps: int = 40
    accumulation_steps: int = 64
    step_s: float = .16
    top_layers: int = 2
    encoder_lr: float = 1e-5
    head_lr: float = 3e-4
    weight_decay: float = .01
    max_updates: int = 100
    save_every: int = 25
    expected_train_conversations: int = 16
    bins_s: tuple = (.2, .4, .6, .8)
    occupancy_threshold: float = .5


def setup_trainable(model, recipe):
    model.eval()
    model.requires_grad_(False)
    if len(model.encoder.layers) < recipe.top_layers:
        raise ValueError('Insufficient encoder layers')
    for layer in model.encoder.layers[-recipe.top_layers:]:
        layer.requires_grad_(True)
    dim = int(model.encoder.d_model)
    head = nn.Sequential(nn.Linear(2*dim, 256), nn.GELU(), nn.Linear(256, 256)).to(model.device)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': recipe.encoder_lr},
        {'params': head.parameters(), 'lr': recipe.head_lr},
    ], weight_decay=recipe.weight_decay)
    return head, optimizer


def union_segments(segments):
    merged = []
    for start, end in sorted(segments):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def activity_code(segments, time_s, recipe):
    """Exact interval occupancy, channel-major little-endian 8-bit VAP code.

    Training targets may use future annotation; this function never reads audio.
    Bins are consecutive lengths, total two seconds, not cumulative endpoints.
    """
    code = 0
    for channel in range(2):
        left = time_s
        for bin_index, width in enumerate(recipe.bins_s):
            right = left + width
            occupied = sum(max(0., min(b, right)-max(a, left)) for a, b in segments[channel])
            if occupied / width >= recipe.occupancy_threshold - 1e-9:
                code |= 1 << (channel*4 + bin_index)
            left = right
    return code


def approved_training_records(manifest, split_path, expected):
    # This guard runs before any waveform or annotation is opened.
    from leakage_guard import validate_manifest
    audit = validate_manifest(manifest, split_path=split_path)
    records = json.loads(Path(manifest).read_text())
    train = [r for r in records if r['split'] == 'train']
    if len(train) != expected:
        raise ValueError(f'Expected frozen {expected}-conversation train set, found {len(train)}')
    for record in train:
        if not str(record['id']).isdigit():
            raise ValueError('Unexpected conversation path identifier')
    audit['loaded_partitions'] = ['train']
    audit['training_ids'] = [r['id'] for r in train]
    return train, audit


def read_segments(directory):
    import srt
    import re
    ignored = {'Awkward Silence', 'Non-Speech Noise', 'Channel Bleed'}
    channels = []
    for speaker in (1, 2):
        intervals = []
        for item in srt.parse((directory/f'speaker_{speaker}_annotation_a.srt').read_text()):
            match = re.match(r'\[([^]]+)\]', item.content)
            if not match:
                raise ValueError('Unknown Otospeech annotation format')
            if match[1] not in ignored:
                intervals.append((item.start.total_seconds(), item.end.total_seconds()))
        channels.append(union_segments(intervals))
    return channels


def draw_crops(records, root, rng, recipe):
    import soundfile as sf
    from encoder import causal_resample
    waves, labels, provenance = [], [], []
    seconds = (recipe.warmup_steps+recipe.accumulation_steps)*recipe.step_s
    for _ in range(recipe.batch_size):
        record = records[int(rng.integers(len(records)))]; cid = record['id']
        directory = root/'oto'/cid
        paths = [directory/f'speaker_{s}_audio.wav' for s in (1, 2)]
        info = [sf.info(p) for p in paths]
        if info[0].samplerate != info[1].samplerate or info[0].frames != info[1].frames:
            raise ValueError('Channel alignment mismatch')
        sr = info[0].samplerate
        duration = info[0].frames/sr
        max_start = math.floor((duration-seconds-sum(recipe.bins_s))/recipe.step_s)
        if max_start < 0:
            raise ValueError('Conversation too short for crop and future targets')
        offset = int(rng.integers(max_start+1))*recipe.step_s
        start = round(offset*sr); count = round(seconds*sr)
        channels = [sf.read(p, start=start, frames=count, dtype='float32')[0] for p in paths]
        x, delay = causal_resample(np.stack(channels, axis=1), sr)
        x = x[:round(seconds*16000)]
        if len(x) != round(seconds*16000):
            raise ValueError('Truncated input crop')
        segments = read_segments(directory)
        target = [activity_code(segments, offset+(recipe.warmup_steps+j+1)*recipe.step_s, recipe)
                  for j in range(recipe.accumulation_steps)]
        # Whole-conversation/crop speaker swap, never a time-dependent permutation.
        swapped = bool(rng.integers(2))
        if swapped:
            x = x[:, ::-1].copy()
            target = [((y & 15) << 4) | (y >> 4) for y in target]
        waves.append(x); labels.append(target)
        provenance.append({'id': cid, 'offset_s': offset, 'input_seconds': seconds,
                           'resampler_signal_delay_s': delay, 'channels_swapped': swapped})
    return waves, np.asarray(labels, dtype=np.int64), provenance


def train_crop(model, head, optimizer, waves, targets, recipe):
    """One optimizer update; detach every cache after every 160ms chunk."""
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    model.eval()  # Gradients remain enabled; BN/context/dropout stay serving-consistent.
    head.train(); optimizer.zero_grad(set_to_none=True)
    buffer = CacheAwareStreamingAudioBuffer(model, online_normalization=True)
    buffer.preprocessor.eval()
    front = buffer.preprocessor.featurizer
    if getattr(front, 'frame_splicing', 1) != 1 or getattr(front, 'stft_pad_amount', None) is not None:
        raise ValueError('Unsupported frontend availability calculation')
    if str(getattr(front, 'normalize', None)).lower() not in ('none', 'na', 'false'):
        raise ValueError('Full-utterance frontend normalization forbidden')
    if len(waves) != recipe.batch_size or len({len(x) for x in waves}) != 1:
        raise ValueError('Expected equally sized independent conversation crops')
    if targets.shape != (recipe.batch_size, recipe.accumulation_steps):
        raise ValueError('Unexpected target shape')
    device = model.device; batch = 2*len(waves); n = len(waves[0])
    signal = torch.from_numpy(np.stack(waves).transpose(0, 2, 1).reshape(batch, n)).to(device)
    hop, n_fft = int(front.hop_length), int(front.n_fft)
    # Only deterministic local STFT is precomputed; no future target audio loaded.
    with torch.no_grad():
        raw, raw_len = buffer.preprocessor(input_signal=signal,
                                           length=torch.full((batch,), n, device=device, dtype=torch.long))
        complete = min(int(raw_len.min()), (n-1-n_fft//2)//hop+1)
        for channel in range(batch):
            buffer.append_processed_signal(raw[channel:channel+1, :, :complete])
    encoder = model.encoder; cfg = encoder.streaming_cfg
    cache = encoder.get_initial_cache_state(batch_size=batch)
    iterator = iter(buffer); step = 0; count = 0; loss_sum = 0.; scored_indices = []
    y = torch.as_tensor(targets, device=device)
    with torch.backends.cudnn.flags(allow_tf32=False):
        while buffer.buffer_idx < complete:
            size = cfg.chunk_size[min(step, 1)] if isinstance(cfg.chunk_size, (list, tuple)) else cfg.chunk_size
            chunk_end = buffer.buffer_idx+size
            if chunk_end > complete:
                break
            available_sample = (chunk_end-1)*hop+n_fft//2+1
            decision_index = math.ceil(available_sample/round(recipe.step_s*16000))
            target_index = decision_index-recipe.warmup_steps-1
            scored = 0 <= target_index < recipe.accumulation_steps
            chunk, length = next(iterator)
            with torch.set_grad_enabled(scored):
                encoded, encoded_len, *next_cache = encoder.cache_aware_stream_step(
                    processed_signal=chunk, processed_signal_length=length,
                    cache_last_channel=cache[0], cache_last_time=cache[1], cache_last_channel_len=cache[2],
                    keep_all_outputs=False, drop_extra_pre_encoded=0 if step == 0 else cfg.drop_extra_pre_encoded)
                if scored:
                    last = encoded[torch.arange(batch, device=device), :, encoded_len.long()-1]
                    logits = head(last.reshape(len(waves), -1))
                    loss = nn.functional.cross_entropy(logits, y[:, target_index])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    (loss/recipe.accumulation_steps).backward()
                    loss_sum += float(loss.detach()); count += 1; scored_indices.append(target_index)
            cache = tuple(t.detach() for t in next_cache)
            step += 1
    if scored_indices != list(range(recipe.accumulation_steps)):
        raise RuntimeError(f'Serving grid did not yield each target exactly once: {scored_indices}')
    trainable = [p for p in model.encoder.parameters() if p.requires_grad]
    frozen_gradients = any(p.grad is not None for p in model.parameters() if not p.requires_grad)
    encoder_grad = sum(float(p.grad.detach().float().square().sum()) for p in trainable if p.grad is not None)**.5
    if frozen_gradients or not math.isfinite(encoder_grad) or encoder_grad <= 0:
        raise RuntimeError('Encoder gradient audit failed')
    norm = torch.nn.utils.clip_grad_norm_(trainable+list(head.parameters()), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'vap_cross_entropy': loss_sum/count, 'encoder_grad_l2': encoder_grad,
            'preclip_grad_l2': float(norm), 'frozen_gradients': frozen_gradients,
            'decisions_per_conversation': count, 'tbptt_chunks': 1}


def checkpoint(path, model, head, optimizer, rng, recipe, audit, update):
    payload = {'format_version': 1, 'encoder': model.encoder.state_dict(), 'head': head.state_dict(),
               'optimizer': optimizer.state_dict(), 'recipe': asdict(recipe), 'audit': audit,
               'update': update, 'numpy_generator': rng.bit_generator.state,
               'numpy_global': np.random.get_state(), 'python_rng': random.getstate(),
               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
               'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def resume(path, model, head, optimizer, rng, recipe, audit):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['recipe'] != asdict(recipe) or state['audit'] != audit or state['model_revision'] != MODEL_REVISION:
        raise ValueError('Resume recipe/data/model provenance changed')
    if state['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Resume code provenance changed; fork an explicit new experiment')
    model.encoder.load_state_dict(state['encoder']); head.load_state_dict(state['head'])
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_generator']; np.random.set_state(state['numpy_global'])
    random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return int(state['update'])


def synthetic_smoke(model):
    # These checks require no annotations, datasets, or benchmark access.
    recipe = Recipe(batch_size=1, warmup_steps=4, accumulation_steps=4, max_updates=1)
    assert activity_code([[], []], 0, recipe) == 0
    assert activity_code([[(0, 2)], []], 0, recipe) == 15
    assert activity_code([[(0, 2)], [(0, 2)]], 0, recipe) == 255
    assert activity_code([[(0, .1)], []], 0, recipe) == 1
    torch.manual_seed(recipe.seed); rng = np.random.default_rng(recipe.seed)
    head, optimizer = setup_trainable(model, recipe)
    before = {name: p.detach().clone() for name, p in model.encoder.named_parameters() if p.requires_grad}
    frozen = next(p for p in model.encoder.parameters() if not p.requires_grad).detach().clone()
    wave = rng.normal(0, .03, (round(8*.16*16000), 2)).astype(np.float32)
    targets = np.asarray([[0, 15, 240, 255]], dtype=np.int64)
    metrics = train_crop(model, head, optimizer, [wave], targets, recipe)
    changed = [name for name, p in model.encoder.named_parameters()
               if p.requires_grad and not torch.equal(before[name], p.detach())]
    assert changed
    assert torch.equal(frozen, next(p for p in model.encoder.parameters() if not p.requires_grad))
    assert not any(module.training for module in model.modules())
    metrics.update({'synthetic_only': True, 'updated_parameter_tensors': len(changed),
                    'serving_eval_mode': True, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
    # Prove optimizer/RNG checkpoint round trip without retaining a smoke model.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='vap-smoke-', dir='/content') as directory:
        path = Path(directory)/'checkpoint.pt'
        audit = {'synthetic_only': True, 'test_data_loaded': False}
        checkpoint(path, model, head, optimizer, rng, recipe, audit, 1)
        expected = rng.random(4)
        update = resume(path, model, head, optimizer, rng, recipe, audit)
        np.testing.assert_array_equal(rng.random(4), expected)
        assert update == 1 and not path.with_name(path.name+'.tmp').exists()
        metrics['checkpoint_rng_roundtrip'] = True
    from encoder import ParakeetStreamingEncoder, test_causality
    metrics['post_update_causality'] = test_causality(ParakeetStreamingEncoder(model=model))
    return metrics


def main():
    if not Path('/content').is_dir():
        raise RuntimeError('Run only on remote Colab; no local models or datasets')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/content/turn-recreation'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--synthetic-smoke', action='store_true')
    args = parser.parse_args(); recipe = Recipe()
    torch.set_num_threads(2); random.seed(recipe.seed); np.random.seed(recipe.seed); torch.manual_seed(recipe.seed)
    if not args.synthetic_smoke:
        if args.manifest is None or args.checkpoint is None:
            parser.error('Training requires --manifest and --checkpoint')
        records, audit = approved_training_records(args.manifest, args.root/'split.json', recipe.expected_train_conversations)
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel
    model_file = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    model = ASRModel.restore_from(model_file).to('cuda').eval()
    if args.synthetic_smoke:
        print(json.dumps(synthetic_smoke(model)), flush=True); return
    head, optimizer = setup_trainable(model, recipe); rng = np.random.default_rng(recipe.seed)
    start = resume(args.checkpoint, model, head, optimizer, rng, recipe, audit) if args.checkpoint.exists() else 0
    for update in range(start+1, recipe.max_updates+1):
        waves, targets, crops = draw_crops(records, args.root, rng, recipe)
        metrics = train_crop(model, head, optimizer, waves, targets, recipe)
        print(json.dumps({'update': update, **metrics, 'crops': crops}), flush=True)
        if update % recipe.save_every == 0 or update == recipe.max_updates:
            checkpoint(args.checkpoint, model, head, optimizer, rng, recipe, audit, update)


if __name__ == '__main__':
    main()
