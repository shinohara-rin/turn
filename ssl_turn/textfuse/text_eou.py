"""Text-only end-of-turn probability at every annotator-'a' segment end on TurnBench dev.
Label-free context: own segments with words; other speaker's segments with >=3 words.
Outputs per segment end: P(<|im_end|>) from livekit/turn-detector and last-token hidden state."""
import json, re, sys, unicodedata, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
S = sys.argv[1]
A = json.load(open(f'{S}/data/tbdev_ann.json'))
MID = sys.argv[2] if len(sys.argv) > 2 else 'livekit/turn-detector'
tok = AutoTokenizer.from_pretrained(MID); model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.float32).eval()
torch.set_num_threads(4)
END = tok.convert_tokens_to_ids('<|im_end|>')

def clean(t):
    t = re.sub(r'\[[^\]]*\]', ' ', t or '')
    t = unicodedata.normalize('NFKC', t).lower()
    t = re.sub(r"[^\w\s'-]", ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

items = []
for cid, r in A.items():
    segs = {s: [(x['start_s'], x['end_s'], clean(x['text'])) for x in r[f'speaker_{s}_annotation_a']] for s in (1, 2)}
    for s in (1, 2):
        o = 3 - s
        for (a, b, txt) in segs[s]:
            if not txt:
                continue
            ev = [(x[0], 'user', x[2]) for x in segs[s] if x[1] <= b + 1e-6 and x[2]]
            ev += [(x[0], 'assistant', x[2]) for x in segs[o] if x[1] <= b and len(x[2].split()) >= 3]
            ev.sort()
            msgs = []
            for _, role, t in ev:
                if msgs and msgs[-1]['role'] == role:
                    msgs[-1]['content'] += ' ' + t
                else:
                    msgs.append(dict(role=role, content=t))
            msgs = msgs[-6:]
            items.append(dict(cid=cid, spk=s, start=a, end=b, msgs=msgs))
print(len(items), flush=True)
probs, hid = [], []
with torch.no_grad():
    for i, it in enumerate(items):
        text = tok.apply_chat_template(it['msgs'], tokenize=False, add_generation_prompt=False)
        text = text[:text.rfind('<|im_end|>')]
        ids = tok(text, return_tensors='pt').input_ids[:, -128:]
        out = model(ids, output_hidden_states=True)
        p = torch.softmax(out.logits[0, -1].float(), -1)[END].item()
        probs.append(p); hid.append(out.hidden_states[-1][0, -1].numpy().astype(np.float16))
        if i % 500 == 0: print(i, p, it['msgs'][-1]['content'][-60:], flush=True)
tag = MID.split('/')[-1]
json.dump([dict(cid=x['cid'], spk=x['spk'], start=x['start'], end=x['end'], p=p, last=x['msgs'][-1]['content'][-80:])
           for x, p in zip(items, probs)], open(f'{S}/sem/eou_{tag}.json', 'w'))
np.save(f'{S}/sem/hid_{tag}.npy', np.stack(hid))
