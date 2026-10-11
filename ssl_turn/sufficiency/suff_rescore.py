"""Re-score the previous draft under the new prefix (Modal L4, one prefill per item).

For consecutive candidates j-1, j of a turn: mean per-token log-prob of draft_{j-1} given
(context, prefix_j) minus given (context, prefix_{j-1}), with a small instruct LM standing in for
the dialogue LLM. A drop means the new words made the cached draft less likely: a cheap,
causal staleness signal that needs no new generation.

    cd ssl_turn/sufficiency && modal run suff_rescore.py --ann ann.json --labels suff_dev.json --out rescore.json [--n 4000]
"""
import json
import random
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
LM = 'Qwen/Qwen2.5-1.5B-Instruct'
app = modal.App('ssl-turn-suff-rescore')
image = modal.Image.debian_slim(python_version='3.12').pip_install('torch==2.8.0', 'transformers==4.57.1', 'accelerate')

SYS = ('You are the listener in a casual two-person conversation. Write the one short reply (at most '
       '20 words) you would give if you answered right now.')


@app.function(image=image, gpu='L4', timeout=3600)
def score(items):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LM)
    model = AutoModelForCausalLM.from_pretrained(LM, torch_dtype=torch.bfloat16).cuda().eval()

    def lp(ctx, prefix, reply):
        msgs = [{'role': 'system', 'content': SYS},
                {'role': 'user', 'content': f'Other person said: "{ctx or "(nothing yet)"}"\nSpeaker so far: "{prefix}"'}]
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        a = tok(head, add_special_tokens=False).input_ids
        b = tok(reply, add_special_tokens=False).input_ids
        return a, b

    out = []
    B = 16
    flat = []
    for it in items:
        flat.append(lp(it['ctx'], it['prev_prefix'], it['prev_reply']))
        flat.append(lp(it['ctx'], it['prefix'], it['prev_reply']))
    res = []
    with torch.no_grad():
        for i in range(0, len(flat), B):
            chunk = flat[i:i + B]
            L = max(len(a) + len(b) for a, b in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id or 0)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for k, (a, b) in enumerate(chunk):
                x = a + b
                ids[k, :len(x)] = torch.tensor(x)
                att[k, :len(x)] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
            logp = torch.log_softmax(logits, -1)
            for k, (a, b) in enumerate(chunk):
                pos = torch.arange(len(a) - 1, len(a) + len(b) - 1)
                tgt = torch.tensor(b)
                res.append(logp[k, pos.cuda(), tgt.cuda()].mean().item())
    for n, it in enumerate(items):
        out.append(res[2 * n + 1] - res[2 * n])
    return out


@app.local_entrypoint()
def main(ann: str, labels: str, out: str = 'rescore.json', n: int = 4000):
    import suff_labels as L
    a, d = json.load(open(ann)), json.load(open(labels))
    data = {**{f'tb/{k}': v for k, v in a['tbdev'].items()}, **{f'oto/{k}': v for k, v in a['oto'].items()}}
    items = []
    for cid in sorted(data):
        prev = {1: '', 2: ''}
        lab = {(x[0], x[6], x[4]): x for x in d.get(cid, [])}
        for turn in L.turns(data[cid]):
            s, end = turn[0][0], turn[-1][3]
            words, cands = L.candidates(turn)
            pre = [(n_, lab.get((s, end, n_))) for n_, t, kind in cands if kind != 'end']
            pre = [(n_, x) for n_, x in pre if x and x[3] is not None]
            for j in range(1, len(pre)):
                (n0, x0), (n1, x1) = pre[j - 1], pre[j]
                items.append(dict(key=f'{cid}|{s}|{end}|{n1}', ctx=' '.join(prev[s].split()[-40:]),
                                  prev_prefix=' '.join(words[:n0][-80:]), prefix=' '.join(words[:n1][-80:]),
                                  prev_reply=x0[8] or ''))
            prev[3 - s] = ' '.join(words)
    random.Random(3).shuffle(items)
    items = items[:n]
    chunks = [items[i:i + 1000] for i in range(0, len(items), 1000)]
    vals = [v for part in score.map(chunks) for v in part]
    json.dump({it['key']: v for it, v in zip(items, vals)}, open(out, 'w'))
    print('wrote', out, len(vals))
