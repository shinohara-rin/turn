"""Per-channel Silero VAD (ONNX, CPU) for the commit policy (../policy.py).

Input: the cached 24 kHz causal resample {WORK}/audio/{split}/{cid}.npy (prep.py), decimated
causally to 16 kHz (encode_asr.to16k). Silero runs streaming on 512-sample chunks (32 ms) with
its recurrent state and 64-sample context, so chunk i depends only on audio before (i + 1) * 32 ms.
Output: {WORK}/vad/{split}/{cid}.npy uint8 [T, 2] = round(255 * p(speech)).

    modal run vad.py --split tbdev
"""
import modal

from common import VOLUMES, WORK, _code, work

app = modal.App('ssl-turn-vad')
SILERO = 'silero-vad==6.2.0'
vad_image = _code(
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy', 'scipy', 'onnxruntime')
    .pip_install(SILERO, extra_options='--no-deps')
)


def silero(x16, sess):
    """x16 [n, B] float32 at 16 kHz -> [n // 512, B] speech probabilities."""
    import numpy as np
    B, N = x16.shape[1], len(x16) // 512
    x = np.ascontiguousarray(x16[:N * 512].T.reshape(B, N, 512))
    state = np.zeros((2, B, 128), np.float32)
    ctx = np.zeros((B, 64), np.float32)
    sr = np.array(16000, np.int64)
    out = np.empty((N, B), np.float32)
    for i in range(N):
        p, state = sess.run(None, {'input': np.concatenate([ctx, x[:, i]], 1), 'state': state, 'sr': sr})
        out[i] = p[:, 0]
        ctx = x[:, i, -64:]
    return out


@app.function(image=vad_image, volumes=VOLUMES, cpu=2, memory=8192, timeout=3600)
def vad_item(split, cid):
    import os
    from importlib.util import find_spec
    import numpy as np
    import sys
    import onnxruntime as ort
    sys.path.insert(0, '/root/ssl_turn/pipeline')
    from encode_asr import to16k
    out = f'{WORK}/vad/{split}/{cid}.npy'
    if os.path.exists(out):
        return cid, 'cached'
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    # the package's Python side imports torch; only its bundled ONNX file is used
    sess = ort.InferenceSession(os.path.join(os.path.dirname(find_spec('silero_vad').origin), 'data', 'silero_vad.onnx'), so)
    a = np.load(f'{WORK}/audio/{split}/{cid}.npy').astype(np.float32)
    p = silero(to16k(a), sess)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, np.round(p * 255).astype(np.uint8))
    work.commit()
    return cid, f'{len(p) * 0.032:.0f} s, speech {np.round((p > 0.5).mean(0), 3)}'


@app.local_entrypoint()
def main(split: str = 'tbdev'):
    cids = sorted(e.path.split('/')[-1][:-4] for e in work.listdir(f'audio/{split}') if e.path.endswith('.npy'))
    print(split, len(cids), 'conversations')
    for cid, msg in vad_item.starmap([(split, c) for c in cids]):
        print(cid, msg)
