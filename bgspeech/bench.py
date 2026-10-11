"""bgbench: a fixed background-speech benchmark on TurnBench dev.

Deployment case: a voice agent hears the user through one microphone, and someone or
something else is audible in the room (a YouTube video, a TV, a person nearby). That sound
lands only in the user's channel. Every TurnBench dev conversation gets one "user" channel,
and each condition below adds a background to that channel only, at a fixed SNR against the
user's own active speech. The other channel stays clean.

Everything here is deterministic. The plan (user channel and donor conversations per
conversation) comes from SEED, and each background's random stream is seeded by
(VERSION, conversation, background style). The same background is used at every SNR of
a style; only its gain changes. Change VERSION whenever the recipe changes.

Backgrounds:
  far    another TB dev conversation (both speakers summed: a podcast or video playing),
         through a far-field room (RT60 0.3-0.7 s) and a small loudspeaker (150-6000 Hz)
  tv     far-field dialogue with a music bed 5 dB under it, through the loudspeaker
  music  music only through the loudspeaker (MUSAN fma, even-indexed files; the odd ones
         are left for training augmentation)
  near   one other talker in the room, close to the mic: a short, direct-dominated room
         response and no loudspeaker. Acoustically like the user; only identity differs
  babble four talkers at once (two donor conversations, all channels), far-field room,
         no loudspeaker: an office or cafe

Donors are other TB dev conversations, never the evaluated one. Training-time augmentation
must not use TB dev audio (bgmix.encode_aug uses otoSpeech conversations outside train/dev).

Scoring (bench_score.py) holds each model's clean operating point fixed, since a deployed
model doesn't know the TV came on.
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import signal

import mixing

VERSION = 'bgbench-v1'
SEED = 20261010
SR = 24000           # rendered audio rate (ssl_turn's cached-audio rate; runners resample)
MASK_RATE = 100.0    # activity masks for SNR references, in frames per second
N_DONORS = 4
# Labels that don't mean "this channel's speaker is talking" (prep.ACTIVITY_EXCLUDED):
# TB annotators mark bleed and noise on the silent channel too.
ACTIVITY_EXCLUDED = ('Awkward Silence', 'Non-Speech Noise', 'Channel Bleed')

# name: (style, SNR dB); 'clean' has no background.
CONDITIONS = {
    'clean': (None, None),
    'far10': ('far', 10.0), 'far5': ('far', 5.0), 'far0': ('far', 0.0), 'farm5': ('far', -5.0),
    'tv5': ('tv', 5.0), 'tv0': ('tv', 0.0),
    'music5': ('music', 5.0), 'music0': ('music', 0.0),
    'near10': ('near', 10.0), 'near5': ('near', 5.0), 'near0': ('near', 0.0),
    'babble5': ('babble', 5.0), 'babble0': ('babble', 0.0),
}
STYLES = ('far', 'tv', 'music', 'near', 'babble')
# Report-card groups (bench_score.py): the headline averages over these.
GROUPS = {'playback': ('far10', 'far5', 'far0', 'farm5', 'tv5', 'tv0'),
          'music': ('music5', 'music0'),
          'near talker': ('near10', 'near5', 'near0'),
          'babble': ('babble5', 'babble0')}


def make_plan(ids: list[str]) -> dict:
    """User channel and donor conversations for every conversation in `ids`."""
    ids = sorted(ids)
    rng = np.random.default_rng(SEED)
    plan = {'version': VERSION, 'seed': SEED, 'ids': ids, 'items': {}}
    for cid in ids:
        others = [c for c in ids if c != cid]
        plan['items'][cid] = {'user': int(rng.integers(1, 3)),
                              'donors': [str(d) for d in rng.choice(others, N_DONORS, replace=False)]}
    return plan


def music_files(names: list[str]) -> list[str]:
    """The benchmark's half of the MUSAN fma files (sorted, even indices)."""
    return sorted(names)[0::2]


def activity(annotations, speaker: int, n: int, rate: float = MASK_RATE, pad_s: float = 0.0) -> np.ndarray:
    """Union over annotators of `speaker`'s annotated speech (excluding ACTIVITY_EXCLUDED)."""
    m = np.zeros(n, bool)
    for a in mixing.ANNOTATORS:
        for start, end, label, *_ in annotations[(speaker, a)]:
            if label in ACTIVITY_EXCLUDED:
                continue
            m[max(0, int((start - pad_s) * rate)):min(n, int(np.ceil((end + pad_s) * rate)))] = True
    return m


def _rng(cid: str, style: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f'{VERSION}:{cid}:{style}'.encode()).hexdigest()[:8], 16))


def near_rir(sr: int, rng: np.random.Generator) -> np.ndarray:
    """Someone about a metre from the mic: short room tail, direct path dominant."""
    return mixing.far_field_rir(sr, rng, rt60=(0.2, 0.4), drr_db=(6.0, 12.0))


def backgrounds(cid: str, n: int, donors: list[np.ndarray], music: list[np.ndarray], sr: int = SR,
                styles=STYLES) -> dict[str, np.ndarray]:
    """One background track of n samples per style. `donors`: [samples, 2] arrays (both
    speakers of each donor conversation, in plan order); `music`: mono tracks."""
    out = {}
    dialog = [d[:, 0] + d[:, 1] for d in donors]
    for style in styles:
        rng = _rng(cid, style)
        if style == 'far':
            out[style] = mixing.playback(mixing.concat_offset(dialog, n, sr, rng), sr, rng)
        elif style == 'tv':
            talk = mixing.concat_offset(dialog, n, sr, rng)
            bed = mixing.concat_offset(music, n, sr, rng)
            bed = bed * (mixing.active_rms(talk, sr) / (mixing.active_rms(bed, sr) + 1e-9) * 10 ** (-5 / 20))
            out[style] = mixing.playback(talk + bed, sr, rng)
        elif style == 'music':
            out[style] = mixing.playback(mixing.concat_offset(music, n, sr, rng), sr, rng)
        elif style == 'near':
            # one talker: speaker 1 of each donor conversation, end to end
            x = mixing.concat_offset([d[:, 0] for d in donors], n, sr, rng)
            out[style] = signal.fftconvolve(x, near_rir(sr, rng))[:n].astype(np.float32)
        elif style == 'babble':
            # four simultaneous talkers: both speakers of two donors, each from its own offset
            talkers = [mixing.concat_offset([d[:, s] for d in donors[k::2]], n, sr, rng)
                       for k in (0, 1) for s in (0, 1)]
            x = sum(t / (mixing.active_rms(t, sr) + 1e-9) for t in talkers)
            out[style] = signal.fftconvolve(x, mixing.far_field_rir(sr, rng))[:n].astype(np.float32)
        else:
            raise ValueError(style)
    return out


def render(cid: str, user_wave: np.ndarray, user_mask: np.ndarray, bgs: dict[str, np.ndarray],
           cond: str, sr: int = SR) -> np.ndarray:
    """The user channel under `cond` (user_mask: user's activity at MASK_RATE)."""
    style, snr = CONDITIONS[cond]
    if style is None:
        return user_wave.astype(np.float32)
    return mixing.mix(user_wave, sr, bgs[style], snr, user_mask)  # mask at 100 Hz (MASK_RATE)
