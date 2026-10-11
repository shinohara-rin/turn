"""Does X-ASR-zh-en (GilgameshWind/X-ASR-zh-en, streaming Zipformer transducer, ~1M h zh+en,
Apache-2.0) keep end-of-turn cues as well as our FastConformer? Linear-probe test on eot-bench,
the protocol of the multilingual thread's multilingual/encoder_probe.py (PR #13):

  every silence span of an eot-bench turn is a readout, READOUT_S into the span; the turn's final
  span is the true end (1), earlier ones are holds (0). Features: the encoder frames of the last
  WINDOW_S of audio ending at the readout (nothing after it is seen), last frame + mean of the last
  ~1 s. Within-language grouped 5-fold CV AUC of a PCA + logistic probe, plus en -> other transfer.

Encoders:
  fc_en   nvidia/stt_en_fastconformer_hybrid_large_streaming_multi (114M), att ctx [70, 0], taps
          after layers 9 (encode_asr.py's middle tap) and 17 (12.5 Hz)
  xasr    X-ASR-zh-en 160 ms streaming encoder (sherpa-onnx export, ~148M params): encoder_out
          (512-d, 25 Hz). The window is zero-padded by 0.32 s so every real frame is emitted: the
          model's chunk (160 ms) plus subsampling right context (130 ms) then sees zeros, never
          audio after the readout. Deployed, that lookahead is ~0.16-0.29 s of real delay.

    python xasr_probe.py extract fc_en --langs en,zh,es,ja      # GPU (NeMo), writes feats_fc_en.npz
    python xasr_probe.py extract xasr --langs en,zh,es,ja       # GPU (onnxruntime-gpu) or CPU
    python xasr_probe.py probe feats_fc_en.npz feats_xasr.npz
"""
import argparse
import io
import os
import sys

DATASET = 'livekit/eot-bench-data'
REVISION = 'ca9d98a9686b920a2d8c9eb984224ba9be74e4dd'
XASR = 'GilgameshWind/X-ASR-zh-en'
XASR_ENCODER = 'deployment/models/chunk-160ms-model/encoder-160ms.onnx'
READOUT_S = 0.2
WINDOW_S = 16.0
MEAN_S = 1.04
SR = 16000


def readouts(language):
    """-> list of (row id, wav window ending at the readout, label)."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, f'data/{language}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=REVISION)
    out = []
    for row in pq.read_table(path, columns=['id', 'audio', 'silence_spans']).to_pylist():
        wav, sr = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
        assert sr == SR, sr
        wav = wav.mean(1) if wav.ndim > 1 else wav
        spans = sorted(row['silence_spans'], key=lambda s: s['start'])
        for i, s in enumerate(spans):
            if s['end'] - s['start'] < READOUT_S:
                continue
            end = int(round((s['start'] + READOUT_S) * SR))
            start = max(0, end - int(WINDOW_S * SR))
            out.append((row['id'], np.ascontiguousarray(wav[start:end]), int(i == len(spans) - 1)))
    return out


def pool(h, frames_per_s):
    """h [T, D] -> [2D]: last frame, mean of the last MEAN_S."""
    import numpy as np
    n = max(1, int(round(MEAN_S * frames_per_s)))
    return np.concatenate([h[-1], h[-n:].mean(0)])


# ---- extraction -----------------------------------------------------------------------------------

def extract_fc(languages):
    import numpy as np
    import torch
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained('nvidia/stt_en_fastconformer_hybrid_large_streaming_multi',
                                                     map_location='cuda').eval()
    model.encoder.set_default_att_context_size([70, 0])
    model.preprocessor.featurizer.dither = 0.0
    model.preprocessor.featurizer.pad_to = 0
    taps, got = (8, 16), {}
    for i in taps:
        model.encoder.layers[i].register_forward_hook(
            lambda mod, a, o, i=i: got.__setitem__(i, (o[0] if isinstance(o, tuple) else o)))
    res = {}
    for lang in languages:
        rows = readouts(lang)
        feats = {i: [] for i in taps}
        for _, wav, _ in rows:
            x = torch.from_numpy(wav)[None].cuda()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                f, fl = model.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]], device='cuda'))
                model.encoder(audio_signal=f, length=fl)
            for i in taps:
                feats[i].append(pool(got[i][0].float().cpu().numpy(), 12.5))
        res.update(_pack(lang, rows, {f'L{i + 1}': np.stack(v) for i, v in feats.items()}))
        print(lang, len(rows), 'readouts', flush=True)
    return res


def _fbank(wav):
    import kaldi_native_fbank as knf
    import numpy as np
    o = knf.FbankOptions()
    o.frame_opts.dither = 0
    o.frame_opts.snip_edges = False
    o.frame_opts.samp_freq = SR
    o.mel_opts.num_bins = 80
    o.mel_opts.high_freq = -400
    fb = knf.OnlineFbank(o)
    fb.accept_waveform(SR, wav.tolist())
    fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])


def extract_xasr(languages, batch=64):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(XASR, XASR_ENCODER)
    sess = ort.InferenceSession(path, providers=[p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider')
                                                 if p in ort.get_available_providers()])
    meta = sess.get_modelmeta().custom_metadata_map
    T, step = int(meta['T']), int(meta['decode_chunk_len'])
    ins = sess.get_inputs()
    print('X-ASR providers', sess.get_providers(), 'chunk', step, 'frames + right context', T - step, flush=True)
    win = int(WINDOW_S * SR)
    pad = int(0.32 * SR)
    res = {}
    for lang in languages:
        rows = readouts(lang)
        out = []
        for b in range(0, len(rows), batch):
            chunk = rows[b:b + batch]
            x = np.zeros((len(chunk), win + pad), np.float32)
            for j, (_, wav, _) in enumerate(chunk):   # left-align every window to WINDOW_S
                x[j, win - len(wav):win] = wav
            fb = np.stack([_fbank(w) for w in x]).astype(np.float32)          # [B, F, 80]
            real = win // 160                                                # fbank frames of real audio
            states = [np.zeros([len(chunk) if not isinstance(d, int) else d for d in i.shape],
                               np.int64 if 'int64' in i.type else np.float32) for i in ins[1:]]
            frames = []
            for s in range(0, fb.shape[1] - T + 1, step):
                o = sess.run(None, {'x': fb[:, s:s + T], **{i.name: v for i, v in zip(ins[1:], states)}})
                frames.append(o[0])
                states = o[1:]
                if s + step >= real:
                    break
            h = np.concatenate(frames, 1)                                    # [B, T', 512] at 25 Hz
            keep = int(round(real / 4))                                      # 40 ms frames of real audio
            h = h[:, :keep]
            out += [pool(h[j], 25.0) for j in range(len(chunk))]
        res.update(_pack(lang, rows, {'L1': np.stack(out)}))
        print(lang, len(rows), 'readouts', flush=True)
    return res


def _pack(lang, rows, feats):
    import numpy as np
    r = {f'{lang}/ids': np.array([x[0] for x in rows]), f'{lang}/y': np.array([x[2] for x in rows], np.int8)}
    r.update({f'{lang}/{k}': v.astype(np.float16) for k, v in feats.items()})
    return r


# ---- probe ----------------------------------------------------------------------------------------

def _fit_score(Xtr, ytr, Xte, gtr):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def pipe(C):
        return make_pipeline(StandardScaler(), PCA(min(256, Xtr.shape[0] // 2, Xtr.shape[1]), random_state=0),
                             LogisticRegression(C=C, max_iter=3000, class_weight='balanced'))
    C = max((roc_auc_score(ytr, cross_val_predict(pipe(C), Xtr, ytr, groups=gtr, cv=GroupKFold(4),
                                                  method='decision_function')), C)
            for C in (1e-3, 1e-2, 1e-1))[1]
    return pipe(C).fit(Xtr, ytr).decision_function(Xte)


def cv_auc(X, y, g, folds=5):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    s = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        s[te] = _fit_score(X[tr], y[tr], X[te], g[tr])
    return roc_auc_score(y, s)


def probe(paths):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    for path in paths:
        z = np.load(path)
        d = {}
        for k in z.files:
            lang, key = k.split('/')
            d.setdefault(lang, {})[key] = z[k]
        name = os.path.basename(path).removeprefix('feats_').removesuffix('.npz')
        taps = sorted({k for v in d.values() for k in v if k.startswith('L')}, key=lambda k: int(k[1:]))
        for keys in [[t] for t in taps] + ([taps] if len(taps) > 1 else []):
            row = []
            for lang, v in d.items():
                X = np.concatenate([v[k].astype(np.float32) for k in keys], 1)
                row.append(f'{lang} {cv_auc(X, v["y"], v["ids"]):.3f}')
            if 'en' in d and len(d) > 1:
                ve = d['en']
                Xe = np.concatenate([ve[k].astype(np.float32) for k in keys], 1)
                for lang, v in d.items():
                    if lang != 'en':
                        X = np.concatenate([v[k].astype(np.float32) for k in keys], 1)
                        row.append(f'en->{lang} {roc_auc_score(v["y"], _fit_score(Xe, ve["y"], X, ve["ids"])):.3f}')
            print(f'{name:8s} {"+".join(keys):8s} ' + '  '.join(row), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('encoder', choices=['fc_en', 'xasr'])
    e.add_argument('--langs', default='en,zh,es,ja')
    e.add_argument('--out', default='')
    p = sub.add_parser('probe')
    p.add_argument('paths', nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'probe':
        return probe(a.paths)
    import numpy as np
    langs = a.langs.split(',')
    res = extract_fc(langs) if a.encoder == 'fc_en' else extract_xasr(langs)
    out = a.out or f'feats_{a.encoder}.npz'
    np.savez_compressed(out, **res)
    print('wrote', out)


if __name__ == '__main__':
    sys.exit(main())
