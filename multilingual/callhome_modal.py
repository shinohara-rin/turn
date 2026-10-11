"""CallHome es / ja / zh -> ssl_turn training data (features + labels) on the ssl-turn-work volume.

Source: talkbank/callhome (gated, CC-BY-NC-SA-4.0) configs spa / jpn / zho: mono 16 kHz telephone
calls (~10 min annotated each) with utterance start/end and speaker.

Per call:
  1. Speech timing: each speaker's utterances intersected with Silero VAD on the mix, so pauses
     inside an utterance become real silences (holds). Calls where a third speaker holds > 5% of
     speech are skipped; minor extra speakers are dropped.
  2. Labels: multilingual/rule_labels.py (Turn / Backchannel by timing rules) through the pinned
     TurnBench gold builder, then labels.floor_targets / floor_projection exactly as prep.oto_item
     does for oto. Written to /work/labels/callhome/{cid}.npz with `activity` (VAP targets) and a
     zero-weight `fine` (no fine labels here).
  3. Pseudo-stereo: pseudo_stereo.gated_stereo (mix at full gain while that speaker talks, about
     -24 dB bleed otherwise), then each channel through encode_asr.encode_channel (the r019 streaming
     FastConformer) -> /work/feats_asr/callhome/{cid}.npy [T, 2, 1024] fp16.

    modal run multilingual/callhome_modal.py                 # all 15 shards in parallel (L4)
"""
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ssl_turn' / 'pipeline'))
sys.path.insert(0, str(ROOT / 'multilingual'))
from common import VOLUMES, WORK, work  # noqa: E402

DATASET = 'talkbank/callhome'
LANGS = {'spa': 'es', 'jpn': 'ja', 'zho': 'zh'}
SHARDS = 5
FRAME_S = 0.08
SR = 16000

app = modal.App('multilingual-callhome')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential', 'git')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                      'scipy==1.15.3', 'soundfile==0.13.1', 'silero-vad==5.1.2', 'typer', 'srt', 'huggingface_hub', 'datasets')
         .pip_install('turnbench @ git+https://github.com/SesameAILabs/turnbench@38a6f874322430cb3ca71d8a52aa1e636e88bad8',
                      extra_options='--no-deps')
         .add_local_dir(str(ROOT / 'ssl_turn'), '/root/ssl_turn', ignore=['**/__pycache__', 'web/**'])
         .add_local_python_source('common', 'encode_asr', 'rule_labels'))
secret = modal.Secret.from_dict({'HF_TOKEN': os.environ.get('HF_TOKEN', '')})


def intersect(a, b):
    """Two sorted interval lists -> their intersection."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


@app.function(image=image, volumes=VOLUMES, secrets=[secret], gpu='L4', cpu=8, memory=32768, timeout=7200)
def shard(config, index, limit=0):
    import io
    import json
    import time
    import zlib
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from silero_vad import get_speech_timestamps, load_silero_vad
    sys.path.insert(0, '/root/ssl_turn')
    import encode_asr
    import labels as lb
    import pseudo_stereo
    import rule_labels

    t0 = time.time()
    torch.set_num_threads(8)
    path = hf_hub_download(DATASET, f'{config}/data-{index:05d}-of-{SHARDS:05d}.parquet', repo_type='dataset',
                           token=os.environ['HF_TOKEN'])
    vad = load_silero_vad()
    enc = encode_asr.load_model()
    for d in ('labels/callhome', 'feats_asr/callhome', 'meta/callhome'):
        os.makedirs(f'{WORK}/{d}', exist_ok=True)
    pf = pq.ParquetFile(path)
    report, row = [], 0
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            if limit and row >= limit:
                break
            cid = f'{LANGS[config]}_{index}_{row}'
            row += 1
            wav, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
            assert sr == SR, sr
            wav = wav.mean(1) if wav.ndim > 1 else wav
            dur = len(wav) / SR
            by = {}
            for a, b, s in zip(r['timestamps_start'], r['timestamps_end'], r['speakers']):
                if b > a:
                    by.setdefault(s, []).append((a, min(b, dur)))
            talk = sorted(by, key=lambda s: -sum(b - a for a, b in by[s]))
            total = sum(b - a for s in by for a, b in by[s])
            extra = sum(b - a for s in talk[2:] for a, b in by[s]) / max(total, 1e-6)
            if len(talk) < 2 or extra > 0.05:
                report.append(dict(cid=cid, skipped=f'{len(talk)} speakers, extra share {extra:.2f}'))
                continue
            ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR, threshold=0.5,
                                       min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=30)
            speech = [(t['start'] / SR, t['end'] / SR) for t in ts]
            segs = []  # refined (speaker 1/2, start, end)
            for k, s in enumerate(talk[:2]):
                for a, b in intersect(union(by[s]), speech):
                    if b - a >= 0.05:
                        segs.append((k + 1, a, b))
            labeled = rule_labels.label(segs)
            events = rule_labels.events(labeled)
            T = int(np.floor(dur / FRAME_S))
            times = (np.arange(T) + 1) * FRAME_S
            y = lb.floor_targets(times, [(s, a, b, l) for s, a, b, l in labeled], events)
            future, future_w = lb.floor_projection(y['floor'], y['floor_w'])
            activity = pseudo_stereo.frame_activity([(a, b, s - 1) for s, a, b in segs], T)
            np.savez_compressed(f'{WORK}/labels/callhome/{cid}.npz', floor=y['floor'], floor_w=y['floor_w'],
                                act=y['act'], act_w=y['act_w'], future=future, future_w=future_w, activity=activity,
                                times=times, fine=np.zeros_like(y['act']))
            stereo = pseudo_stereo.gated_stereo(wav, SR, [(a, b, s - 1) for s, a, b in segs],
                                                rng=np.random.default_rng(zlib.crc32(cid.encode())))
            Tf = len(wav) // encode_asr.FRAME
            feats = np.stack([encode_asr.encode_channel(enc, np.ascontiguousarray(stereo[:Tf * encode_asr.FRAME, c]), Tf)
                              for c in (0, 1)], 1)
            np.save(f'{WORK}/feats_asr/callhome/{cid}.tmp.npy', feats)
            os.replace(f'{WORK}/feats_asr/callhome/{cid}.tmp.npy', f'{WORK}/feats_asr/callhome/{cid}.npy')
            n_bc = sum(l == 'Backchannel' for *_, l in labeled)
            report.append(dict(cid=cid, lang=LANGS[config], dur=round(dur, 1), segs=len(segs), backchannels=n_bc,
                               eot=len(events['eot_positive_events']), frames=Tf))
            print(report[-1], f'{time.time() - t0:.0f} s', flush=True)
    json.dump(report, open(f'{WORK}/meta/callhome/{config}_{index}.json', 'w'))
    work.commit()
    return report


@app.local_entrypoint()
def main(configs: str = 'spa,jpn,zho', shards: str = '0,1,2,3,4', limit: int = 0):
    jobs = [(c, int(i), limit) for c in configs.split(',') for i in shards.split(',')]
    out = []
    for rep in shard.starmap(jobs, return_exceptions=True):
        if isinstance(rep, Exception):
            print('FAILED', repr(rep)[:1500])
            continue
        out += rep
    ok = [r for r in out if 'skipped' not in r]
    print(f'{len(ok)} calls written, {len(out) - len(ok)} skipped')
    for lang in LANGS.values():
        rs = [r for r in ok if r['lang'] == lang]
        print(lang, len(rs), 'calls', round(sum(r['dur'] for r in rs) / 3600, 1), 'h',
              sum(r['eot'] for r in rs), 'EOT events', sum(r['backchannels'] for r in rs), 'backchannels')
