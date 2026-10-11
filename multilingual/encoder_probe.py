"""Which small streaming encoder keeps end-of-turn cues in es / ja / zh? Linear-probe test.

Data: LiveKit eot-bench (livekit/eot-bench-data, pinned revision), single user turns per language.
Every silence span is a readout; the clip's final span is the true end of turn (label 1), earlier
spans are holds (label 0). Each readout is taken READOUT_S into the span, from the last WINDOW_S of
audio ending there (the audio is cut at the readout, so nothing after it is seen whatever the
encoder's lookahead). Features per tap layer: the final frame plus the mean of the last ~1 s,
as in ssl_turn/pipeline/diagnose_asr.py.

Encoders:
  fc_en      nvidia/stt_en_fastconformer_hybrid_large_streaming_multi (114M, English), our r019
             backbone, attention context [70, 0]. Taps: layer 9 (the encode_asr.py middle tap) and 17.
  nemotron   nvidia/nemotron-3.5-asr-streaming-0.6b (600M, 35 languages), cache-aware FastConformer,
             attention context [56, 0]. Taps after layers 2, 4, 6, 8, 12, 24, so one pass also
             answers "what if we keep only the bottom k layers" (k = 4 is ~105M).
  moon_*     moonshine-ai/moonshine-streaming-{small-es, small-ja, tiny-zh, small} (one model per
             language; encoder only). Taps: middle and last encoder layer.

    modal run multilingual/encoder_probe.py              # extract (L4), writes feats_*.npz
    python multilingual/encoder_probe.py probe feats_*.npz
"""
import io
import sys

DATASET = 'livekit/eot-bench-data'
REVISION = 'ca9d98a9686b920a2d8c9eb984224ba9be74e4dd'
LANGS = ('en', 'es', 'ja', 'zh')
READOUT_S = 0.2
WINDOW_S = 16.0
MEAN_S = 1.04
SR = 16000

NEMO = {'fc_en': ('nvidia/stt_en_fastconformer_hybrid_large_streaming_multi', [70, 0], (8, 16)),
        'nemotron': ('nvidia/nemotron-3.5-asr-streaming-0.6b', [56, 0], (1, 3, 5, 7, 11, 23))}
MOON = {'moon_en': ('moonshine-ai/moonshine-streaming-small', 'en'),
        'moon_es': ('moonshine-ai/moonshine-streaming-small-es', 'es'),
        'moon_ja': ('moonshine-ai/moonshine-streaming-small-ja', 'ja'),
        'moon_zh': ('moonshine-ai/moonshine-streaming-tiny-zh', 'zh')}


def readouts(language):
    """-> list of (row id, wav prefix ending at the readout, label)."""
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(DATASET, f'data/{language}/validation-00000-of-00001.parquet', repo_type='dataset',
                           revision=REVISION)
    t = pq.read_table(path, columns=['id', 'audio', 'silence_spans']).to_pylist()
    out = []
    for row in t:
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


try:
    import modal
except ImportError:  # probe-only use
    modal = None

if modal is not None:
    app = modal.App('multilingual-encoder-probe')
    nemo_image = (modal.Image.debian_slim(python_version='3.11')
                  .apt_install('libsndfile1', 'ffmpeg', 'build-essential', 'git')
                  .pip_install('Cython', 'packaging')
                  .pip_install('nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main',
                               'pyarrow', 'soundfile', 'huggingface_hub'))
    moon_image = (modal.Image.debian_slim(python_version='3.11')
                  .apt_install('libsndfile1')
                  .pip_install('torch==2.8.0', 'transformers==5.19.0', 'pyarrow', 'soundfile', 'huggingface_hub'))

    @app.function(image=nemo_image, gpu='L4', cpu=4, memory=32768, timeout=3600)
    def extract_nemo(name, languages):
        import time
        import numpy as np
        import torch
        import nemo.collections.asr as nemo_asr
        model_id, ctx, taps = NEMO[name]
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id, map_location='cuda').eval()
        model.encoder.set_default_att_context_size(ctx)
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0
        layers = model.encoder.layers
        got = {}
        for i in taps:
            layers[i].register_forward_hook(
                lambda m, a, o, i=i: got.__setitem__(i, (o[0] if isinstance(o, tuple) else o)))
        print(name, len(layers), 'layers', sum(p.numel() for p in model.encoder.parameters()) / 1e6, 'M encoder')
        res = {}
        for lang in languages:
            t0 = time.time()
            rows = readouts(lang)
            feats = {i: [] for i in taps}
            for rid, wav, y in rows:
                x = torch.from_numpy(wav)[None].cuda()
                n = torch.tensor([x.shape[1]], device='cuda')
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    f, fl = model.preprocessor(input_signal=x, length=n)
                    model.encoder(audio_signal=f, length=fl)
                for i in taps:
                    h = got[i][0].float().cpu().numpy()  # [T, D] (NeMo layers are batch-first)
                    feats[i].append(pool(h, 12.5))
            res[f'{lang}/ids'] = np.array([r[0] for r in rows])
            res[f'{lang}/y'] = np.array([r[2] for r in rows], np.int8)
            for i in taps:
                res[f'{lang}/L{i + 1}'] = np.stack(feats[i]).astype(np.float16)
            print(lang, len(rows), 'readouts', int(res[f'{lang}/y'].sum()), 'ends', f'{time.time() - t0:.0f} s')
        buf = io.BytesIO()
        np.savez_compressed(buf, **res)
        return buf.getvalue()

    @app.function(image=moon_image, gpu='L4', cpu=4, memory=16384, timeout=3600)
    def extract_moon(name):
        import time
        import numpy as np
        import torch
        from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
        model_id, lang = MOON[name]
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(model_id).cuda().eval()
        proc = AutoProcessor.from_pretrained(model_id)
        enc = model.get_encoder()
        print(name, sum(p.numel() for p in enc.parameters()) / 1e6, 'M encoder')
        t0 = time.time()
        rows = readouts(lang)
        feats = None
        for rid, wav, y in rows:
            inp = proc(wav, return_tensors='pt', sampling_rate=SR)
            with torch.no_grad():
                o = enc(input_values=inp['input_values'].cuda(), attention_mask=inp['attention_mask'].cuda(),
                        output_hidden_states=True)
            hs = o.hidden_states
            taps = (len(hs) // 2, len(hs) - 1)
            feats = feats or {i: [] for i in taps}
            for i in taps:
                feats[i].append(pool(hs[i][0].float().cpu().numpy(), 50.0))
        res = {f'{lang}/ids': np.array([r[0] for r in rows]), f'{lang}/y': np.array([r[2] for r in rows], np.int8)}
        for i in feats:
            res[f'{lang}/L{i}'] = np.stack(feats[i]).astype(np.float16)
        print(lang, len(rows), 'readouts', f'{time.time() - t0:.0f} s')
        buf = io.BytesIO()
        np.savez_compressed(buf, **res)
        return buf.getvalue()

    @app.local_entrypoint()
    def main(encoders: str = 'fc_en,nemotron,moon_en,moon_es,moon_ja,moon_zh', languages: str = ','.join(LANGS)):
        calls = {}
        for name in encoders.split(','):
            if name in NEMO:
                calls[name] = extract_nemo.spawn(name, languages.split(','))
            else:
                calls[name] = extract_moon.spawn(name)
        for name, call in calls.items():
            try:
                data = call.get()
            except Exception as e:  # keep the other encoders' results
                print('FAILED', name, repr(e)[:2000])
                continue
            with open(f'feats_{name}.npz', 'wb') as f:
                f.write(data)
            print('wrote', f'feats_{name}.npz')


# ---------------------------------------------------------------- probe (local, CPU)

def load(path):
    import numpy as np
    z = np.load(path)
    d = {}
    for k in z.files:
        lang, key = k.split('/')
        d.setdefault(lang, {})[key] = z[k]
    return d


def feature_sets(d):
    """Named tap combinations: e.g. 'L4+L2' for a bottom-4-layer model with a middle tap."""
    keys = sorted({k for v in d.values() for k in v if k.startswith('L')}, key=lambda k: int(k[1:]))
    layers = [int(k[1:]) for k in keys]
    sets = {}
    for top in layers:
        mids = [l for l in layers if l <= top // 2 + 1 and l < top]
        mid = max(mids) if mids else None
        sets[f'L{top}' + (f'+L{mid}' if mid else '')] = [f'L{top}'] + ([f'L{mid}'] if mid else [])
    return sets


def cv_auc(X, y, g, folds=5):
    """Within-language: grouped (by turn) outer 5-fold, scores pooled over folds."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    s = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        s[te] = _fit_score(X[tr], y[tr], X[te], g[tr])
    return roc_auc_score(y, s)


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


def probe(paths):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rows = []
    for path in paths:
        name = path.split('feats_')[-1].removesuffix('.npz')
        d = load(path)
        for set_name, keys in feature_sets(d).items():
            if not all(k in v for v in d.values() for k in keys):
                continue
            res = {}
            for lang, v in d.items():
                X = np.concatenate([v[k].astype(np.float32) for k in keys], 1)
                res[lang] = cv_auc(X, v['y'], v['ids'])
            if 'en' in d and len(d) > 1:   # cross-lingual: train on English, test on each other language
                v = d['en']
                Xen = np.concatenate([v[k].astype(np.float32) for k in keys], 1)
                for lang, w in d.items():
                    if lang == 'en':
                        continue
                    X = np.concatenate([w[k].astype(np.float32) for k in keys], 1)
                    res[f'en>{lang}'] = roc_auc_score(w['y'], _fit_score(Xen, v['y'], X, v['ids']))
            rows.append((name, set_name, res))
            print(name, set_name, ' '.join(f'{k} {x:.3f}' for k, x in res.items()), flush=True)
    return rows


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'probe':
    probe(sys.argv[2:])
