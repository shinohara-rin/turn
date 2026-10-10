"""Score ssl_turn FastConformer heads on LiveKit eot-bench (one language) on a Modal L4.

    modal run eotbench/modal_eot.py --run r019_asr_bgaug \
        --models fine1_bal1_s1,fine1_bal1_s2,bgaug_s1,bgaug_s2 --out tracks.npz

eot-bench rows are single user turns (16 kHz mono). As in eot-bench's VAP adapter, the user
goes in channel 0 and the agent channel 1 is silent (zeros). Each turn is encoded once with
encode_asr.encode_channel (causal streaming FastConformer, 80 ms frames) and the head runs over
the whole turn in one forward pass; turns are shorter than the head's exact-context reach
(model.context_frames), and the encoder and head are causal, so frame t equals the score of
the audio prefix that ends at (t + 1) * 80 ms. to_harness.py reads the score at each eot-bench
grid timestamp from the last frame available by then.

Output npz: {model}/{variant}/{row id} -> float16 [T] user-channel score track.
"""
import sys
from pathlib import Path

import modal

SRC = Path(__file__).resolve().parent.parent / 'ssl_turn'
sys.path.insert(0, str(SRC / 'pipeline'))
from common import VOLUMES, WORK  # noqa: E402

DATASET = 'livekit/eot-bench-data'
REVISION = 'ca9d98a9686b920a2d8c9eb984224ba9be74e4dd'  # pinned like eot-bench's committed runs

app = modal.App('ssl-turn-eotbench')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                      'scipy==1.15.3', 'soundfile==0.13.1', 'huggingface_hub')
         .add_local_dir(str(SRC), '/root/ssl_turn', ignore=['**/__pycache__', 'pipeline/**'])
         .add_local_python_source('common', 'encode_asr'))


def build_head(cfg):
    """train.build_head for no-tap feature configs (feats 'asr')."""
    import model as m
    assert cfg.get('feats') == 'asr' and not cfg.get('taps') and not cfg.get('enroll') and not cfg.get('tune'), cfg
    return m.TurnModel(tap_layers=0, final_dim=1024, dim=cfg.get('dim', 256), heads=cfg.get('heads', 4),
                       layers=cfg.get('layers', 4), window_s=cfg.get('window_s', 20.0),
                       dropout=cfg.get('dropout', 0.1))


def variants(floor, future, silent):
    """Channel-0 (user) EOT scores, as score.score_variants: floor/future [T, 4] / [T, H, 4]
    over (HELD_0, HELD_1, OPEN, CONTESTED); silent [T] = user p(SILENT)."""
    released = floor[:, 2] + floor[:, 1]            # user no longer holds: OPEN or held by the agent
    released04 = future[:, 0, 2] + future[:, 0, 1]
    return {'eot': released, 'eot_q': released * silent, 'eot_f04_q': released04 * silent}


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=16384, timeout=3600)
def score(run, models, language='en'):
    import io
    import sys
    import time
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, '/root/ssl_turn')
    import encode_asr
    import model as m

    t0 = time.time()
    path = hf_hub_download(DATASET, f'data/{language}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=REVISION)
    table = pq.read_table(path, columns=['id', 'audio'])
    enc = encode_asr.load_model()
    heads, ctx = {}, {}
    for name in models:
        ck = torch.load(f'{WORK}/runs/{run}/{name}.pt', map_location='cuda')
        heads[name] = build_head(ck['cfg']).cuda().eval()
        heads[name].load_state_dict(ck['state'])
        ctx[name] = m.context_frames(ck['cfg'].get('layers', 4), ck['cfg'].get('window_s', 20.0))
    out, audio_s = {}, 0.0
    for rid, audio in zip(table.column('id').to_pylist(), table.column('audio').to_pylist()):
        wav, sr = sf.read(io.BytesIO(audio['bytes']), dtype='float32')
        assert sr == 16000, sr
        wav = wav.mean(1) if wav.ndim > 1 else wav
        T = len(wav) // encode_asr.FRAME
        audio_s += len(wav) / sr
        user = encode_asr.encode_channel(enc, wav[:T * encode_asr.FRAME], T)
        agent = encode_asr.encode_channel(enc, np.zeros(T * encode_asr.FRAME, np.float32), T)
        x = torch.from_numpy(np.stack([user, agent], 1))[None].cuda()   # [1, T, 2, 1024]
        for name, net in heads.items():
            assert T <= ctx[name], (rid, T)  # one pass is exact
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                o = net(None, x)
            floor = o['floor'][0].float().softmax(-1).cpu().numpy()             # [T, 4]
            future = o['future'][0].float().softmax(-1).cpu().numpy()           # [T, H, 4]
            silent = o['act'][0].float().softmax(-1)[:, 0, 0].cpu().numpy()     # [T] user p(SILENT)
            for v, track in variants(floor, future, silent).items():
                out[f'{name}/{v}/{rid}'] = np.clip(track, 0, 1).astype(np.float16)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    print(f'{len(table)} turns, {audio_s / 60:.1f} min audio, {time.time() - t0:.0f} s wall')
    return buf.getvalue()


@app.local_entrypoint()
def main(run: str = 'r019_asr_bgaug', models: str = 'fine1_bal1_s1,fine1_bal1_s2,bgaug_s1,bgaug_s2',
         language: str = 'en', out: str = 'tracks.npz'):
    data = score.remote(run, models.split(','), language)
    with open(out, 'wb') as f:
        f.write(data)
    print('wrote', out)


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=16384, timeout=1800)
def causality_check(run='r019_asr_bgaug', name='bgaug_s2', n_turns=20, cuts=5, language='en', seed=0):
    """Max |score(full turn) - score(audio prefix)| at the prefix's last frame, over random cuts.
    Zero (up to bf16 noise) means one full-turn pass equals scoring each prefix separately."""
    import io
    import sys
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, '/root/ssl_turn')
    import encode_asr
    path = hf_hub_download(DATASET, f'data/{language}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=REVISION)
    table = pq.read_table(path, columns=['id', 'audio']).slice(0, n_turns)
    enc = encode_asr.load_model()
    ck = torch.load(f'{WORK}/runs/{run}/{name}.pt', map_location='cuda')
    net = build_head(ck['cfg']).cuda().eval()
    net.load_state_dict(ck['state'])
    rng = np.random.default_rng(seed)

    def track(wav):
        T = len(wav) // encode_asr.FRAME
        user = encode_asr.encode_channel(enc, wav[:T * encode_asr.FRAME], T)
        agent = encode_asr.encode_channel(enc, np.zeros(T * encode_asr.FRAME, np.float32), T)
        x = torch.from_numpy(np.stack([user, agent], 1))[None].cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(None, x)
        floor = o['floor'][0].float().softmax(-1).cpu().numpy()
        silent = o['act'][0].float().softmax(-1)[:, 0, 0].cpu().numpy()
        return variants(floor, floor[:, None], silent)['eot_q']

    diffs = []
    for audio in table.column('audio').to_pylist():
        wav, _ = sf.read(io.BytesIO(audio['bytes']), dtype='float32')
        full = track(wav)
        for t in rng.integers(10, len(full), cuts):
            # prefix with audio before (t + 1) * 80 ms (+ an odd sample tail) -> frame t must match
            pre = track(wav[:(t + 1) * encode_asr.FRAME + int(rng.integers(0, encode_asr.FRAME))])
            diffs.append(abs(float(full[t]) - float(pre[t])))
    return dict(n=len(diffs), max=max(diffs), mean=float(np.mean(diffs)))
