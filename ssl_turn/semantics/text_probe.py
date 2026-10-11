"""LiveKit text turn detector (livekit/turn-detector v0.4.1-intl, via eot-bench's own adapter) on the
true words before each eot-bench pause, with the agent context. Text-only reference for probe_fit.py.

    python text_probe.py EOT_BENCH_REPO_DIR text_en.json
"""
import json
import sys
import warnings

sys.path.insert(0, sys.argv[1])
from huggingface_hub import HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
from eot_harness.livekit_text_adapter import LiveKitTextTurnDetectorAdapter
warnings.simplefilter("ignore")
fs = HfFileSystem()
p = 'datasets/livekit/eot-bench-data@ca9d98a9686b920a2d8c9eb984224ba9be74e4dd/data/en/validation-00000-of-00001.parquet'
with fs.open(p) as f:
    rows = pq.ParquetFile(f).read(columns=['id', 'silence_spans', 'words', 'messages']).to_pylist()
ad = LiveKitTextTurnDetectorAdapter()
print('last msg roles', [r['messages'][-1]['role'] for r in rows[:5]])
out = []
for ti, r in enumerate(rows):
    for si, s in enumerate(r['silence_spans']):
        words = ' '.join(w['word'] for w in r['words'] if w['end'] <= s['start'] + 1e-6)
        msgs = list(r['messages']) + [{'role': 'user', 'content': words}]
        out.append(dict(turn=ti, span=si, label=int(si == len(r['silence_spans']) - 1), prompt=ad._format_chat_input(msgs), words=words))
probs = []
B = 32
prompts = [o['prompt'] for o in out]
for i in range(0, len(prompts), B):
    probs += ad._batch_text_probs(prompts[i:i + B])
for o, pr in zip(out, probs):
    o['text_p'] = pr
    del o['prompt']
json.dump(out, open(sys.argv[2], 'w'))
from sklearn.metrics import roc_auc_score
print('text AUC', roc_auc_score([o['label'] for o in out], probs), len(out))
