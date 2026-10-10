"""Background-speech robustness of the ssl_turn floor model on TurnBench dev.

Same mixtures as bgspeech/ (VAP): a far-field "podcast playing" (another TB dev
conversation, both speakers) is added to one channel ("user") of each evaluated
conversation at a given SNR. Only that channel is re-encoded with Cat; the clean
channel's cached features are reused. Inference runs in the same GPU container
(no mixed features are stored), writing /work/runs/<out_run>/probs.npz with keys
'<model>@<cond>/tbdev/<cid>/{post,silent,fine}' for score.py-style scoring.

    modal run bgmix.py --plan plan.json --masks masks.npz --run r012_fine \
        --models fine1_bal1_s1,fine1_bal1_s2 --out-run bg_r012
"""
import json
import time
from pathlib import Path

import modal

from common import VOLUMES, WORK, _code, gpu_image, setup_path, work

image = gpu_image
if modal.is_local():  # in the container this module lives at /root/bgmix.py
    image = gpu_image.add_local_file(str(Path(__file__).resolve().parents[2] / 'bgspeech' / 'mixing.py'),
                                     '/root/bgspeech/mixing.py')
app = modal.App('ssl-turn-bgmix')
# name: (SNR dB, gated to user activity, background style)
#   far:   another conversation (both speakers) played through a far-field room + loudspeaker
#   near:  one other talker, dry (no room, no loudspeaker): someone talking right by the mic
#   tv:    far-field dialogue with a music bed 5 dB below it
#   music: far-field music only (MUSAN fma, some with vocals)
CONDITIONS = {'snr10': (10.0, False, 'far'), 'snr5': (5.0, False, 'far'), 'snr0': (0.0, False, 'far'),
              'gate5': (5.0, True, 'far'), 'gate0': (0.0, True, 'far'), 'snrm5': (-5.0, False, 'far'),
              'near5': (5.0, False, 'near'), 'near0': (0.0, False, 'near'),
              'tv5': (5.0, False, 'tv'), 'tv0': (0.0, False, 'tv'), 'music0': (0.0, False, 'music')}


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=8, memory=32768, timeout=7200)
def encode_infer(plan, masks, run, names, out_run, conds, batch_waves=16):
    import hashlib
    import os
    import sys
    import numpy as np
    import torch
    setup_path()
    sys.path.insert(0, '/root/bgspeech')
    sys.path.insert(0, '/root/ssl_turn/pipeline')
    import cat_encoder as ce
    import mixing
    import train as tr
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sr = ce.SAMPLE_RATE
    ids = plan['subset']

    # 1. Mixed user channels (24 kHz, same recipe as the VAP run; rng seeded per conversation).
    t0 = time.time()
    audio = {c: np.load(f'/work/audio/tbdev/{c}.npy').astype(np.float32) for c in ids}
    donors = {}
    for c in ids:
        for d in plan['items'][c]['donors']:
            if d not in donors:
                donors[d] = np.load(f'/work/audio/tbdev/{d}.npy').astype(np.float32)
    styles = {CONDITIONS[c][2] for c in conds}
    music = np.load('/work/musan/fma_24k.npy').astype(np.float32) if styles & {'tv', 'music'} else None
    seed = lambda c, tag: np.random.default_rng(int(hashlib.sha256((c + tag).encode()).hexdigest()[:8], 16))
    waves = []
    for c in ids:
        user = plan['items'][c]['user']
        x = audio[c][:, user - 1]
        dons = [donors[d] for d in plan['items'][c]['donors']]
        bgs = {}
        if 'far' in styles:  # same rng stream as the VAP run and the first ssl_turn run
            bgs['far'] = mixing.background_track([d[:, 0] + d[:, 1] for d in dons], len(x), sr, seed(c, ''))
        if 'near' in styles:
            bgs['near'] = mixing.concat_offset([d[:, 0] for d in dons], len(x), sr, seed(c, ':near'))
        if 'tv' in styles:
            rng = seed(c, ':tv')
            talk = mixing.concat_offset([d[:, 0] + d[:, 1] for d in dons], len(x), sr, rng)
            bed = mixing.concat_offset([music], len(x), sr, rng)
            bed *= mixing.active_rms(talk, sr) / (mixing.active_rms(bed, sr) + 1e-9) * 10 ** (-5 / 20)
            bgs['tv'] = mixing.playback(talk + bed, sr, rng)
        if 'music' in styles:
            rng = seed(c, ':music')
            bgs['music'] = mixing.playback(mixing.concat_offset([music], len(x), sr, rng), sr, rng)
        mask = masks[c]
        for cond in conds:
            snr, gated, style = CONDITIONS[cond]
            bg = bgs[style]
            b = bg * mixing.smooth_gate(mask, 100.0, len(x), sr) if gated else bg
            waves.append(((c, cond), mixing.mix(x, sr, b, snr, mask, bg_level=bg)))
    del donors, music
    print(f'mixed {len(waves)} channels in {time.time() - t0:.0f}s', flush=True)

    # 2. Cat-encode only the mixed channels.
    enc = ce.build('/work/models/cat', '/work/models/cat/cat_encoder.safetensors', device='cuda')
    cols = tr.columns([15, 23, 31])
    tr.LOADED = [15, 23, 31]
    feats = {}
    waves.sort(key=lambda w: -len(w[1]))
    t0, done = time.time(), 0.0
    for b in range(0, len(waves), batch_waves):
        group = waves[b:b + batch_waves]
        n = (max(len(w) for _, w in group) + ce.HOP - 1) // ce.HOP * ce.HOP
        x = np.zeros((len(group), n), np.float32)
        for i, (_, w) in enumerate(group):
            x[i, :len(w)] = w
        with torch.no_grad():
            out = enc.stream(torch.from_numpy(x).cuda(), 50, out_device='cpu', out_dtype=torch.float16)
        f = torch.cat([out['taps'].flatten(2), out['final']], -1).numpy()
        for i, (key, w) in enumerate(group):
            feats[key] = f[i, :len(w) // ce.HOP][:, cols].copy()
            done += len(w) / sr
        print(f'batch {b // batch_waves}: {done / (time.time() - t0):.0f} channel-s/s', flush=True)
    del enc, waves
    torch.cuda.empty_cache()

    # 3. Inference with the saved checkpoints, clean (cached features) and each condition.
    models, configs = {}, {}
    for nme in names:
        ck = torch.load(f'/work/runs/{run}/{nme}.pt', map_location='cuda')
        configs[nme] = ck['cfg']
        models[nme] = tr.build_model(ck['cfg']).cuda()
        models[nme].load_state_dict(ck['state'])
    clean = {c: np.ascontiguousarray(np.load(f'/work/feats/tbdev/{c}.npy', mmap_mode='r')[..., cols]) for c in ids}
    enrolls = None
    if any(c.get('enroll') == 'ecapa' for c in configs.values()):
        with np.load('/work/enroll_ecapa/tbdev.npz') as z:
            enrolls = {('tbdev', c): torch.from_numpy(z[c][:, 0]).cuda() for c in ids}
    elif any(c.get('enroll') for c in configs.values()):  # enrollment clip: the speaker's clean first 20 s
        enrolls = {}
        with np.load('/work/bg/tbdev_activity.npz') as z:
            for c in ids:
                act = torch.from_numpy(z[c][:len(clean[c])]).cuda()
                enrolls[('tbdev', c)] = tr.enroll_vector(torch.from_numpy(clean[c][:len(act)]).cuda(), act)
    probs = {}
    for cond in ['clean'] + list(conds):
        Xs, offs = [], [0]
        for c in ids:
            x = clean[c]
            if cond != 'clean':
                m = feats[(c, cond)]
                T = min(len(x), len(m))
                x = x[:T].copy()
                x[:, plan['items'][c]['user'] - 1] = m[:T]
            Xs.append(x)
            offs.append(offs[-1] + len(x))
        X = torch.from_numpy(np.concatenate(Xs)).cuda()
        out = tr.infer_all(models, configs, (('tbdev', X, offs, ids),), 'cuda', enrolls)
        for k, v in out.items():
            mdl, rest = k.split('/', 1)
            probs[f'{mdl}@{cond}/{rest}'] = v
        del X
    os.makedirs(f'/work/runs/{out_run}', exist_ok=True)
    np.savez_compressed(f'/work/runs/{out_run}/probs.npz', **probs)
    json.dump(dict(plan=plan, run=run, models=names, conds=conds), open(f'/work/runs/{out_run}/bgmix.json', 'w'))
    work.commit()
    return len(probs)


AUG_TAPS = [15, 23, 31]  # feats_aug columns: these taps then final (train.columns(AUG_TAPS))


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=8, memory=32768, timeout=7200)
def encode_aug(cids, donor_ids, snr_range=(-5.0, 20.0), batch_waves=16, style='far'):
    """Training-time background augmentation for otoSpeech conversations.

    For each conversation and each channel c, add a far-field 'podcast' made of 2
    random donor conversations (both speakers) to channel c only, at an SNR drawn
    from `snr_range` against c's active speech, and Cat-encode that channel. Saves
    /work/feats_aug/oto/<cid>.npy, float16 [T, 2, 4608]: channel c = c with its own
    background (the other channel's clean features stay in /work/feats/oto). Labels
    are unchanged: background talkers never hold the floor.

    style='near' instead adds one dry donor talker (no room, no loudspeaker: someone talking
    right next to the mic) and writes /work/feats_aug_near/oto.
    """
    import hashlib
    import os
    import sys
    import numpy as np
    import torch
    setup_path()
    sys.path.insert(0, '/root/bgspeech')
    sys.path.insert(0, '/root/ssl_turn/pipeline')
    import cat_encoder as ce
    import mixing
    import train as tr
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sr = ce.SAMPLE_RATE
    cols = tr.columns(AUG_TAPS)
    enc = ce.build('/work/models/cat', '/work/models/cat/cat_encoder.safetensors', device='cuda')
    out_dir = '/work/feats_aug/oto' if style == 'far' else f'/work/feats_aug_{style}/oto'
    os.makedirs(out_dir, exist_ok=True)
    meta, t0, done = {}, time.time(), 0.0
    for g in range(0, len(cids), batch_waves // 2):
        group = cids[g:g + batch_waves // 2]
        waves = []
        for cid in group:
            a = np.load(f'/work/audio/oto/{cid}.npy').astype(np.float32)
            act = np.load(f'/work/labels/oto/{cid}.npz')['activity']  # [T, 2] on the 80 ms grid
            tag = 'aug' if style == 'far' else f'aug-{style}'
            rng = np.random.default_rng(int(hashlib.sha256(f'{tag}:{cid}'.encode()).hexdigest()[:8], 16))
            for c in (0, 1):
                picks = rng.choice([d for d in donor_ids if d != cid], 2, replace=False)
                dons = []
                for d in picks:
                    x = np.load(f'/work/audio/oto/{d}.npy').astype(np.float32)
                    dons.append(x[:, 0] + x[:, 1] if style == 'far' else x[:, int(rng.integers(2))])
                if style == 'far':
                    bg = mixing.background_track(dons, len(a), sr, rng)
                else:
                    bg = mixing.concat_offset(dons, len(a), sr, rng)
                mask = np.repeat(act[:, c] > 0.5, 8)  # 12.5 Hz -> 100 Hz
                snr = float(rng.uniform(*snr_range))
                waves.append(((cid, c), mixing.mix(a[:, c], sr, bg, snr, mask)))
                meta.setdefault(cid, {})[c] = dict(snr=snr, donors=[str(d) for d in picks])
        n = (max(len(w) for _, w in waves) + ce.HOP - 1) // ce.HOP * ce.HOP
        x = np.zeros((len(waves), n), np.float32)
        for i, (_, w) in enumerate(waves):
            x[i, :len(w)] = w
        with torch.no_grad():
            out = enc.stream(torch.from_numpy(x).cuda(), 50, out_device='cpu', out_dtype=torch.float16)
        f = torch.cat([out['taps'].flatten(2), out['final']], -1).numpy()
        per = {}
        for i, ((cid, c), w) in enumerate(waves):
            per.setdefault(cid, {})[c] = f[i, :len(w) // ce.HOP][:, cols]
            done += len(w) / sr
        for cid, ch in per.items():
            T = min(len(ch[0]), len(ch[1]))
            np.save(f'{out_dir}/{cid}.npy', np.stack([ch[0][:T], ch[1][:T]], 1))
        work.commit()
        print(f'{g + len(group)}/{len(cids)}: {done / (time.time() - t0):.0f} channel-s/s', flush=True)
    return meta


# ---- background augmentation on streaming-ASR features (feats: asr) ----------------------
# Styles from bgspeech/bench.py, drawn per conversation channel. No near talker: r016 showed
# that teaching the head to ignore a voice like the user's costs clean accuracy.
ASR_AUG_STYLES = {'far': 0.5, 'tv': 0.2, 'babble': 0.2, 'music': 0.1}


def _asr_aug_image():
    if not modal.is_local():  # the container already runs on the built image
        return gpu_image
    from encode_asr_modal import asr_image
    bg = Path(__file__).resolve().parents[2] / 'bgspeech'
    return asr_image.add_local_dir(str(bg), '/root/bgspeech', ignore=['**/__pycache__', '*.pyc'])


@app.function(image=_asr_aug_image(), volumes=VOLUMES, gpu='L4', cpu=8, memory=49152, timeout=7200)
def encode_aug_asr(cids, donor_ids, snr_range=(-5.0, 20.0)):
    """Like encode_aug, for the streaming FastConformer: each channel of each conversation gets
    its own background (style drawn from ASR_AUG_STYLES, 4 random donor conversations, music
    from the training half of MUSAN fma, SNR ~ U(snr_range) against that channel's active
    speech) and is re-encoded. Writes /work/feats_asr_aug/oto/<cid>.npy, float16 [T, 2, 1024],
    frame-aligned with /work/feats_asr/oto. Labels are unchanged."""
    import hashlib
    import os
    import sys
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, '/root/bgspeech')
    import bench
    import encode_asr as ea
    import mixing
    sr = bench.SR
    enc = ea.load_model()
    mdir = f'{WORK}/musan/fma_train'
    music = []
    for f in sorted(os.listdir(mdir)):
        x, s = sf.read(f'{mdir}/{f}', dtype='float32', always_2d=True)
        music.append(mixing.resample_to(x.mean(1), s, sr))
    out_dir = f'{WORK}/feats_asr_aug/oto'
    os.makedirs(out_dir, exist_ok=True)
    styles, probs = list(ASR_AUG_STYLES), np.array(list(ASR_AUG_STYLES.values()))
    meta, t0, done = {}, time.time(), 0.0
    for cid in cids:
        a = np.load(f'{WORK}/audio/oto/{cid}.npy').astype(np.float32)
        act = np.load(f'{WORK}/labels/oto/{cid}.npz')['activity']  # [T, 2] on the 80 ms grid
        T = np.load(f'{WORK}/feats_asr/oto/{cid}.npy', mmap_mode='r').shape[0]
        rng = np.random.default_rng(int(hashlib.sha256(f'aug-asr:{cid}'.encode()).hexdigest()[:8], 16))
        feats = []
        for c in (0, 1):
            style = styles[int(rng.choice(len(styles), p=probs))]
            picks = rng.choice([d for d in donor_ids if d != cid], 4, replace=False)
            dons = [np.load(f'{WORK}/audio/oto/{d}.npy').astype(np.float32) for d in picks]
            bg = bench.backgrounds(f'train:{cid}:{c}', len(a), dons, music, styles=(style,))[style]
            mask = np.repeat(act[:, c] > 0.5, 8)  # 12.5 Hz -> 100 Hz
            snr = float(rng.uniform(*snr_range))
            w16 = ea.to16k(mixing.mix(a[:, c], sr, bg, snr, mask))
            feats.append(ea.encode_channel(enc, w16[:T * ea.FRAME], T))
            meta.setdefault(cid, {})[c] = dict(style=style, snr=snr, donors=[str(d) for d in picks])
            done += len(a) / sr
        np.save(f'{out_dir}/{cid}.tmp.npy', np.stack(feats, 1))
        os.replace(f'{out_dir}/{cid}.tmp.npy', f'{out_dir}/{cid}.npy')
        work.commit()
        print(f'{cid}: {meta[cid]}; {done / (time.time() - t0):.0f} channel-s/s', flush=True)
    return meta


@app.local_entrypoint()
def aug_asr(groups: int = 8, donors: int = 24):
    """modal run bgmix.py::aug_asr  (background-augmented FastConformer train features)"""
    print(aug_asr_plan.remote(groups, donors))


@app.function(image=image, volumes=VOLUMES, cpu=2, memory=4096, timeout=10800)
def aug_asr_plan(groups, n_donors):
    import os
    work.reload()
    split = json.load(open(f'{WORK}/split.json'))['splits']
    have = sorted(f[:-4] for f in os.listdir(f'{WORK}/audio/oto') if f.endswith('.npy'))
    root = f'{WORK}/feats_asr_aug'
    done = {f[:-4] for f in os.listdir(f'{root}/oto')} if os.path.isdir(f'{root}/oto') else set()
    train = [c for c in split['train'] if c in have and c not in done]
    donors = [c for c in have if c not in set(split['train']) | set(split['dev'])][:n_donors]
    assert len(donors) >= 8, donors
    meta = {}
    for m in encode_aug_asr.map([g for g in (train[i::groups] for i in range(groups)) if g],
                                kwargs=dict(donor_ids=donors)):
        meta.update(m)
    work.reload()
    os.makedirs(root, exist_ok=True)
    old = json.load(open(f'{root}/meta.json')) if os.path.exists(f'{root}/meta.json') else {}
    old.update(meta)
    json.dump(old, open(f'{root}/meta.json', 'w'))
    work.commit()
    return dict(encoded=len(meta), donors=len(donors), train=len(train))


ecapa_image = _code(modal.Image.debian_slim(python_version='3.12').pip_install(
    'torch==2.8.0', 'torchaudio==2.8.0', 'speechbrain==1.0.3', 'numpy', 'scipy', 'huggingface_hub<0.26', 'requests'))
K_ENROLL = 8


@app.function(image=ecapa_image, volumes=VOLUMES, gpu='L4', cpu=8, memory=32768, timeout=3600)
def enroll_ecapa():
    """Speaker-verification enrollment vectors (SpeechBrain ECAPA, 192-d) from clean audio.

    For every channel: otoSpeech train conversations get K_ENROLL vectors, each from a random
    20 s run of that speaker's active frames (labels activity); oto dev and TB dev get one,
    from the first 20 s of active speech (TB dev: bgspeech annotation activity).
    Writes /work/enroll_ecapa/{oto,tbdev}.npz: cid -> float32 [2, K, 192].
    """
    import os
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from speechbrain.inference.speaker import EncoderClassifier
    enc = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', run_opts={'device': 'cuda'},
                                         savedir='/tmp/ecapa')
    split = json.load(open('/work/split.json'))['splits']
    hop16 = 1280  # one 80 ms frame at 16 kHz
    rng = np.random.default_rng(0)

    def clip(a16, idx):
        return np.concatenate([a16[i * hop16:(i + 1) * hop16] for i in idx])

    def embed(clips):
        n = max(len(c) for c in clips)
        x = torch.zeros(len(clips), n)
        lens = torch.tensor([len(c) / n for c in clips])
        for i, c in enumerate(clips):
            x[i, :len(c)] = torch.from_numpy(c)
        with torch.no_grad():
            return enc.encode_batch(x.cuda(), lens.cuda())[:, 0].float().cpu().numpy()

    def one(audio24, act, k):
        a = audio24.astype(np.float32)
        out = np.zeros((2, k, 192), np.float32)
        for c in (0, 1):
            a16 = resample_poly(a[:, c], 2, 3).astype(np.float32)
            idx = np.flatnonzero(act[:len(a16) // hop16, c])
            if len(idx) < 25:
                continue
            clips = []
            for j in range(k):
                start = 0 if k == 1 else int(rng.integers(0, max(1, len(idx) - 250)))
                clips.append(clip(a16, idx[start:start + 250]))
            out[c] = embed(clips)
        return out

    os.makedirs('/work/enroll_ecapa', exist_ok=True)
    oto = {}
    for cid in split['train'] + split['dev']:
        if not os.path.exists(f'/work/labels/oto/{cid}.npz') or not os.path.exists(f'/work/audio/oto/{cid}.npy'):
            continue
        act = np.load(f'/work/labels/oto/{cid}.npz')['activity'] > 0.5
        oto[cid] = one(np.load(f'/work/audio/oto/{cid}.npy'), act, K_ENROLL if cid in split['train'] else 1)
    np.savez('/work/enroll_ecapa/oto.npz', **oto)
    tb = {}
    with np.load('/work/bg/tbdev_activity.npz') as z:
        for cid in z.files:
            tb[cid] = one(np.load(f'/work/audio/tbdev/{cid}.npy'), z[cid], 1)
    np.savez('/work/enroll_ecapa/tbdev.npz', **tb)
    work.commit()
    return dict(oto=len(oto), tbdev=len(tb))


@app.local_entrypoint()
def ecapa():
    print(enroll_ecapa.remote())


@app.local_entrypoint()
def aug(groups: int = 3, style: str = 'far', every: int = 1):
    """modal run bgmix.py::aug [--style near --every 2]  (encode background-augmented train features)"""
    print(aug_plan.remote(groups, style, every))


@app.function(image=image, volumes=VOLUMES, cpu=2, memory=4096, timeout=10800)
def aug_plan(groups, style='far', every=1):
    """Augment every `every`-th train conversation; donors are prepped conversations outside train/dev."""
    import os
    split = json.load(open('/work/split.json'))['splits']
    root = '/work/feats_aug' if style == 'far' else f'/work/feats_aug_{style}'
    have = sorted(f[:-4] for f in os.listdir('/work/audio/oto') if f.endswith('.npy'))
    done = {f[:-4] for f in os.listdir(f'{root}/oto')} if os.path.isdir(f'{root}/oto') else set()
    train = [c for c in split['train'] if c in have][::every]
    train = [c for c in train if c not in done]
    donors = [c for c in have if c not in set(split['train']) | set(split['dev'])]
    chunks = [train[i::groups] for i in range(groups)]
    meta = {}
    for m in encode_aug.map(chunks, kwargs=dict(donor_ids=donors, style=style,
                                                 snr_range=(-5.0, 20.0) if style == 'far' else (-5.0, 15.0))):
        meta.update(m)
    work.reload()
    os.makedirs(root, exist_ok=True)
    old = json.load(open(f'{root}/meta.json')) if os.path.exists(f'{root}/meta.json') else {}
    old.update(meta)
    json.dump(old, open(f'{root}/meta.json', 'w'))
    work.commit()
    return dict(encoded=len(meta), donors=len(donors), train=len(train))


@app.local_entrypoint()
def main(plan: str, masks: str, run: str = 'r012_fine', models: str = 'fine1_bal1_s1,fine1_bal1_s2',
         out_run: str = 'bg_r012', conds: str = 'snr10,snr5,snr0,gate5,gate0'):
    import numpy as np
    p = json.load(open(plan))
    with np.load(masks) as z:
        m = {k: z[k] for k in z.files}
    print(encode_infer.remote(p, m, run, models.split(','), out_run, conds.split(',')))
