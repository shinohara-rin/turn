"""Does the streaming FastConformer already carry the end-of-turn cue at pause onset?

For every silence span of LiveKit eot-bench (en, 400 user turns), dump causal features at a few
query points after the span starts (0.1 / 0.3 / 0.6 s of silence):
  enc    FastConformer features at the query frame (1024-d, mid layer + output)
  pre    mean FastConformer features over the last 0.4 s before the span (the turn's last words)
  hid    r019 head hidden state of the user channel at the query frame (192-d, after the final norm)
  eot_q  r019 score at the query frame
  pred   RNNT prediction-network output after the tokens emitted so far (640-d): a small LM state
  joint  RNNT joint hidden at the query frame (encoder frame + prediction state, 640-d)
  hyp    greedy RNNT transcript emitted up to the query frame
The greedy RNNT decode runs on the causal encoder output frame by frame, so it is causal too.
Frame convention as eotbench/to_harness.py: score at grid time t is frame floor(t / 0.08) - 1.
probe_fit.py fits cross-validated probes on the dump.

    modal run ssl_turn/semantics/probe_eotbench.py --out probe_en.npz
"""
import sys
from pathlib import Path

import modal

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / 'pipeline'))
from common import VOLUMES, WORK  # noqa: E402

DATASET = 'livekit/eot-bench-data'
REVISION = 'ca9d98a9686b920a2d8c9eb984224ba9be74e4dd'
QUERIES = (0.1, 0.3, 0.6)

app = modal.App('ssl-turn-semprobe')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libsndfile1', 'ffmpeg', 'build-essential')
         .pip_install('torch==2.8.0', 'torchaudio==2.8.0', 'nemo_toolkit[asr]==2.4.0', 'pyarrow==19.0.1',
                      'scipy==1.15.3', 'soundfile==0.13.1', 'huggingface_hub')
         .add_local_dir(str(SRC), '/root/ssl_turn', ignore=['**/__pycache__', 'pipeline/**', 'web/**'])
         .add_local_python_source('common', 'encode_asr'))


def build_head(cfg):
    import model as m
    assert cfg.get('feats') == 'asr' and not cfg.get('taps') and not cfg.get('enroll'), cfg
    return m.TurnModel(tap_layers=0, final_dim=1024, dim=cfg.get('dim', 256), heads=cfg.get('heads', 4),
                       layers=cfg.get('layers', 4), window_s=cfg.get('window_s', 20.0), dropout=0.0)


def greedy_states(model, enc_out):
    """Greedy RNNT decode over stored encoder frames (stored frame k + 1 <- encoder frame k, so frame
    0 is padding). Returns per stored frame f, after consuming frames <= f: prediction-net output
    [T, 640], joint hidden [T, 640] at frame f, and the transcript so far."""
    import numpy as np
    import torch
    dec, joint = model.decoder, model.joint
    blank = joint.num_classes_with_blank - 1
    T = len(enc_out)
    preds, joints, hyps = np.zeros((T, dec.pred_hidden), np.float32), np.zeros((T, 640), np.float32), [''] * T
    tokens = []
    with torch.no_grad():
        g, state = dec.predict(None, None, add_sos=False, batch_size=1)
        f_all = joint.project_encoder(torch.from_numpy(enc_out.astype(np.float32)).cuda()[None])  # [1, T, H]
        for t in range(1, T):
            f = f_all[:, t:t + 1]
            for _ in range(5):
                gp = joint.project_prednet(g)
                hidden = joint.joint_net[:-1](f + gp)
                k = int(joint.joint_net[-1](hidden).argmax(-1))
                if k == blank:
                    break
                tokens.append(k)
                g, state = dec.predict(torch.tensor([[k]], device='cuda'), state, add_sos=False, batch_size=1)
            preds[t] = g[0, -1].float().cpu().numpy()
            joints[t] = joint.joint_net[:-1](f + joint.project_prednet(g))[0, 0].float().cpu().numpy()
            hyps[t] = tokens and model.tokenizer.ids_to_text(tokens) or ''
    return preds, joints, hyps


@app.function(image=image, volumes=VOLUMES, gpu='L4', cpu=4, memory=16384, timeout=3600)
def dump(run='r019_asr_bgaug', name='bgaug_s2', language='en'):
    import io
    import json
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, '/root/ssl_turn')
    import encode_asr

    path = hf_hub_download(DATASET, f'data/{language}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=REVISION)
    table = pq.read_table(path).to_pylist()
    enc = encode_asr.load_model()
    ck = torch.load(f'{WORK}/runs/{run}/{name}.pt', map_location='cuda')
    net = build_head(ck['cfg']).cuda().eval()
    net.load_state_dict(ck['state'])
    hid = {}
    net.norm.register_forward_hook(lambda m, i, o: hid.__setitem__('x', o))
    rows = dict(enc=[], pre=[], hid=[], pred=[], joint=[], hyp=[], eot_q=[], label=[], turn=[], span=[], q=[], start=[], dur=[])
    meta = []
    for ti, r in enumerate(table):
        wav, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        wav = wav.mean(1) if wav.ndim > 1 else wav
        T = len(wav) // encode_asr.FRAME
        user = encode_asr.encode_channel(enc, wav[:T * encode_asr.FRAME], T)
        agent = encode_asr.encode_channel(enc, np.zeros(T * encode_asr.FRAME, np.float32), T)
        x = torch.from_numpy(np.stack([user, agent], 1))[None].cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(None, x)
        floor = o['floor'][0].float().softmax(-1).cpu().numpy()
        silent = o['act'][0].float().softmax(-1)[:, 0, 0].cpu().numpy()
        eot_q = (floor[:, 2] + floor[:, 1]) * silent
        h = hid['x'].float().view(1, 2, T, -1)[0, 0].cpu().numpy()   # [T, D] user channel
        pred, joint, hyps = greedy_states(enc, user[:, 512:])
        spans = r['silence_spans']
        for si, s in enumerate(spans):
            final = si == len(spans) - 1
            dur = s['end'] - s['start']
            pre = slice(max(0, int(s['start'] / 0.08) - 5), max(1, int(s['start'] / 0.08)))
            for q in QUERIES:
                if q > dur + 1e-6:
                    continue
                f = min(T - 1, max(0, int(np.floor(round((s['start'] + q) / 0.08, 6))) - 1))
                rows['enc'].append(user[f]); rows['pre'].append(user[pre].astype(np.float32).mean(0))
                rows['hid'].append(h[f]); rows['eot_q'].append(eot_q[f]); rows['label'].append(int(final))
                rows['pred'].append(pred[f]); rows['joint'].append(joint[f]); rows['hyp'].append(hyps[f])
                rows['turn'].append(ti); rows['span'].append(si); rows['q'].append(q)
                rows['start'].append(s['start']); rows['dur'].append(dur)
        meta.append(dict(id=r['id'], words=r['words'], messages=r['messages'], spans=spans))
    out = {k: np.asarray(v, np.float16 if k in ('enc', 'pre', 'hid', 'pred', 'joint') else None) for k, v in rows.items()}
    out['meta'] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    return buf.getvalue()


@app.local_entrypoint()
def main(run: str = 'r019_asr_bgaug', name: str = 'bgaug_s2', language: str = 'en', out: str = 'probe_en.npz'):
    data = dump.remote(run, name, language)
    open(out, 'wb').write(data)
    print('wrote', out)
