"""Pause-warp augmentation: break the "time since voice activity" shortcut for EOT.

The floor head's EOT score climbs with silence length on both true ends and mid-turn holds
(semantics/README.md): long silence is evidence of a yield in otoSpeech, so the head leans on
it. Warped copies of each training conversation decorrelate the two:
  holds   mutual silences inside a turn (floor HELD by the speaker who resumes) are stretched,
          with probability P_HOLD, by 0.3-2.0 s
  yields  mutual silences after an EOT (floor OPEN or taken over) are shortened, with
          probability P_YIELD, to 0.16 s up to their own length
Edits happen on the 80 ms label grid, in the cached 24 kHz audio of both channels at once.
Inserted frames are copies of the gap's own interior audio (the room tone), joined with 5 ms
crossfades at every splice so no 12.5 Hz click pattern marks an edited pause. Labels follow
the same frame map (an inserted frame takes the label of the gap frame it copies), and floor
projections (future) are rebuilt from the warped floor. The warped audio is then encoded like
encode_asr.encode.

Outputs: /work/feats_asr_warp/oto/{cid}.npy, /work/labels_warp/oto/{cid}.npz (same keys as
labels/oto). train.py reads them for configs with warp_aug.

    modal run pausewarp.py [--groups 6]
"""
import json
import os

import modal

from common import VOLUMES, WORK, setup_path, work

app = modal.App('ssl-turn-pausewarp')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                      'scipy==1.15.3', 'soundfile==0.13.1')
         .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), '/root/ssl_turn',
                        ignore=['**/__pycache__', 'web/**', 'pipeline/**'])
         .add_local_python_source('common', 'encode_asr'))

SR, FRAME = 24000, 1920          # cached audio rate, samples per 80 ms frame
XFADE = 120                      # 5 ms crossfade at each splice
P_HOLD, P_YIELD = 0.5, 0.5
STRETCH = (4, 25)                # frames added to a stretched hold (0.32-2.0 s)
MIN_GAP = 2                      # gaps shorter than 0.16 s are left alone


def gaps(activity, floor):
    """Mutual-silence runs [a, b) of >= MIN_GAP frames between speech, with their kind:
    'hold' when the floor stays HELD by one speaker and that speaker talks next, else 'yield'."""
    import numpy as np
    quiet = activity.sum(1) == 0
    out, i, T = [], 0, len(quiet)
    while i < T:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < T and quiet[j]:
            j += 1
        if i > 0 and j < T and j - i >= MIN_GAP:
            hard = floor[i:j].argmax(1)
            holder = np.bincount(hard, minlength=4).argmax()
            nxt = activity[j]
            kind = 'hold' if holder < 2 and (hard == holder).mean() > 0.9 and nxt[holder] > 0 and nxt.sum() == 1 \
                else 'yield'
            out.append((i, j, kind))
        i = j
    return out


def frame_map(T, gap_list, rng):
    """New-frame -> old-frame index list, plus the number of holds stretched / yields cut."""
    src, last, n_h, n_y = [], 0, 0, 0
    for a, b, kind in gap_list:
        src.extend(range(last, a))
        if kind == 'hold' and rng.random() < P_HOLD:
            interior = list(range(a + 1, b - 1)) or list(range(a, b))
            extra = int(rng.integers(STRETCH[0], STRETCH[1] + 1))
            mid = (a + b) // 2
            src.extend(range(a, mid))
            off = int(rng.integers(len(interior)))
            src.extend(interior[(off + k) % len(interior)] for k in range(extra))
            src.extend(range(mid, b))
            n_h += 1
        elif kind == 'yield' and b - a > MIN_GAP and rng.random() < P_YIELD:
            keep = int(rng.integers(MIN_GAP, b - a))
            head = keep // 2
            src.extend(range(a, a + head))
            src.extend(range(b - (keep - head), b))
            n_y += 1
        else:
            src.extend(range(a, b))
        last = b
    src.extend(range(last, T))
    return src, n_h, n_y


def splice(audio, src):
    """audio [N, C] at 24 kHz; output frame i = old frame src[i], crossfaded at discontinuities."""
    import numpy as np
    T = len(src)
    out = np.empty((T * FRAME, audio.shape[1]), np.float32)
    ramp = np.linspace(0, 1, XFADE, dtype=np.float32)[:, None]
    for i, s in enumerate(src):
        out[i * FRAME:(i + 1) * FRAME] = audio[s * FRAME:(s + 1) * FRAME]
        if i and src[i - 1] + 1 != s and s * FRAME >= XFADE:
            # replace the samples before the splice with a fade from the previous source's
            # continuation into the new source's own lead-in
            lead = audio[s * FRAME - XFADE:s * FRAME]
            out[i * FRAME - XFADE:i * FRAME] = out[i * FRAME - XFADE:i * FRAME] * (1 - ramp) + lead * ramp
    return out


def warp_labels(z, src):
    import numpy as np
    setup_path()
    import labels as lb
    src = np.asarray(src)
    out = {k: z[k][src] for k in ('floor', 'floor_w', 'act', 'act_w', 'activity', 'fine') if k in z}
    out['future'], out['future_w'] = lb.floor_projection(out['floor'], out['floor_w'])
    out['times'] = (np.arange(len(src)) + 1) * lb.FRAME_S
    return out


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=8, memory=32768, timeout=7200)
def warp_group(cids, seed=0):
    import time
    import numpy as np
    import encode_asr
    model = encode_asr.load_model()
    stats, t0 = [], time.time()
    os.makedirs(f'{WORK}/feats_asr_warp/oto', exist_ok=True)
    os.makedirs(f'{WORK}/labels_warp/oto', exist_ok=True)
    for cid in cids:
        if os.path.exists(f'{WORK}/feats_asr_warp/oto/{cid}.npy'):
            continue
        rng = np.random.default_rng([seed, int(cid)])
        z = dict(np.load(f'{WORK}/labels/oto/{cid}.npz'))
        a24 = np.load(f'{WORK}/audio/oto/{cid}.npy').astype(np.float32)
        T = min(len(z['floor']), len(a24) // FRAME)
        z = {k: v[:T] for k, v in z.items()}
        g = gaps(z['activity'], z['floor'])
        src, n_h, n_y = frame_map(T, g, rng)
        a = splice(a24[:T * FRAME], src)
        chans = [encode_asr.to16k(a[:, c]) for c in (0, 1)]
        n = min(len(c) for c in chans)
        Tn = min(n // encode_asr.FRAME, len(src))
        feats = np.stack([encode_asr.encode_channel(model, chans[c][:Tn * encode_asr.FRAME], Tn) for c in (0, 1)], 1)
        lab = warp_labels(z, src[:Tn])
        np.save(f'{WORK}/feats_asr_warp/oto/{cid}.tmp.npy', feats)
        os.replace(f'{WORK}/feats_asr_warp/oto/{cid}.tmp.npy', f'{WORK}/feats_asr_warp/oto/{cid}.npy')
        np.savez_compressed(f'{WORK}/labels_warp/oto/{cid}.npz', **lab)
        kinds = [k for _, _, k in g]
        stats.append(dict(cid=cid, T=T, Tn=Tn, holds=kinds.count('hold'), yields=kinds.count('yield'),
                          stretched=n_h, cut=n_y))
        print(stats[-1], f'{time.time() - t0:.0f}s', flush=True)
    work.commit()
    return stats


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=1024, timeout=300)
def train_ids(n_train):
    have = {f[:-4] for f in os.listdir(f'{WORK}/feats_asr/oto') if not f.endswith('.tmp.npy')}
    split = json.load(open(f'{WORK}/split.json'))['splits']
    return [c for c in split['train'] if c in have][:n_train]


@app.local_entrypoint()
def main(groups: int = 6, n_train: int = 131):
    cids = train_ids.remote(n_train)
    total = []
    for r in warp_group.map([cids[i::groups] for i in range(groups)]):
        total += r
    print(json.dumps(dict(convs=len(total), holds=sum(s['holds'] for s in total),
                          stretched=sum(s['stretched'] for s in total), yields=sum(s['yields'] for s in total),
                          cut=sum(s['cut'] for s in total), frames=sum(s['T'] for s in total),
                          frames_warped=sum(s['Tn'] for s in total))))
