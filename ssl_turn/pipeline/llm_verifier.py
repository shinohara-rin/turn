"""LLM verifier on the audio model's own fires (leak-free text oracle; see llm_oracle.py).

For each audio commit (score.verifier_fires), the LLM sees a streaming-ASR-style transcript
visible at the fire time (annotator words revealed uniformly within segments, +0.3 s latency,
no continuation markers) and is asked a plain question, reasoning in a sentence or two first:
  eot: is A done with their turn, so B should respond now?
  int: is A taking the floor, rather than giving listener feedback?
The answer is the final 'ANSWER: yes|no' line (1.0 / 0.0); the probability of 'yes' at that
token is kept as a secondary score. Identical transcripts are asked once.

    python llm_verifier.py transcripts.json fires.json out.jsonl [--budgets 0.3,0.1,0.05]
"""
import argparse
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

LATENCY = 0.3
RUNS = 12
MODEL = 'nemotron-3-5-lightning-30b'
URL = 'https://api.runinfra.ai/v1/chat/completions'
HEAD = ('Live streaming-ASR transcript of a two-person spoken conversation (verbatim, fillers kept; '
        'the newest words may lag slightly behind the audio).\n\n')
ASK = {
    'eot': 'Speaker A may have just stopped talking. Is A done with their turn, so that B should start '
           'responding now? (No if A is likely to keep going, e.g. mid-sentence or trailing off to think.)',
    'int': 'B has been talking, and A has just started making sound (A\'s newest words may not be '
           'transcribed yet). Is A taking the floor (starting their own turn or interrupting B), rather '
           'than giving brief listener feedback like "yeah", "mhm", "right", "oh wow" or laughter while '
           'B keeps talking?',
}
TAIL = '\n\nReason in at most two short sentences, then end with a final line "ANSWER: yes" or "ANSWER: no".'


def words(conv):
    """[(word end time, speaker, segment start, word)] with uniform reveal inside each segment."""
    w = []
    for s, segs in conv.items():
        for a, b, text in segs:
            ws = re.findall(r'\[[^\]]*\]|\S+', text)  # keep [noise tags] whole
            w += [(a + (j + 1) / len(ws) * (b - a), int(s), a, x) for j, x in enumerate(ws)]
    return sorted(w)


def transcript(ws, t, target):
    """Visible words grouped per utterance (ordered by utterance start, as a per-channel streaming
    ASR would show them); consecutive utterances of one speaker are merged."""
    utt = {}
    for te, s, a, x in ws:
        if te + LATENCY > t:
            break
        utt.setdefault((a, s), []).append(x)
    runs = []
    for (a, s), xs in sorted(utt.items()):
        if runs and runs[-1][0] == s:
            runs[-1][1].extend(xs)
        else:
            runs.append((s, list(xs)))
    return '\n'.join(f"{'A' if s == target else 'B'}: {' '.join(xs)}" for s, xs in runs[-RUNS:]) or '(nothing yet)'


def ask(client, kind, text):
    body = dict(model=MODEL, messages=[{'role': 'user', 'content': HEAD + text + '\n\n' + ASK[kind] + TAIL}],
                max_tokens=400, temperature=0, logprobs=True, top_logprobs=5, reasoning_effort='none')
    for attempt in range(6):
        try:
            r = client.post(URL, json=body, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(r.status_code)
            r.raise_for_status()
            ch = r.json()['choices'][0]
            content = ch['message']['content'] or ''
            m = re.findall(r'ANSWER:\s*\**\s*(yes|no)', content, re.I)
            if not m:
                return dict(answer=None, raw=content[-200:])
            ans = 1.0 if m[-1].lower() == 'yes' else 0.0
            p = None
            toks = (ch.get('logprobs') or {}).get('content') or []
            for tk in reversed(toks):  # last yes/no token = the answer
                if tk['token'].strip().lower() in ('yes', 'no'):
                    yes = sum(math.exp(x['logprob']) for x in tk['top_logprobs'] if x['token'].strip().lower() == 'yes')
                    no = sum(math.exp(x['logprob']) for x in tk['top_logprobs'] if x['token'].strip().lower() == 'no')
                    p = yes / (yes + no) if yes + no else None
                    break
            return dict(answer=ans, p=p)
        except Exception:
            time.sleep(2 ** attempt)
    return dict(answer=None)


def main():
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument('transcripts')
    ap.add_argument('fires')
    ap.add_argument('out')
    ap.add_argument('--budgets', default='0.3,0.1,0.05')
    ap.add_argument('--workers', type=int, default=96)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    tr = json.load(open(args.transcripts))['tbdev']
    fires = json.load(open(args.fires))
    prompts = {}
    for kind, by in fires.items():
        for b in args.budgets.split(','):
            for cid, f in by[b]['fires'].items():
                ws = words(tr[cid])
                for s, times in f.items():
                    for t in times:
                        prompts.setdefault((kind, transcript(ws, t, int(s))), set()).add((cid, s, f'{t:.2f}'))
    done = set()
    if os.path.exists(args.out):
        done = {(r['kind'], r['text']) for r in map(json.loads, open(args.out)) if r.get('answer') is not None}
    todo = [k for k in prompts if k not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(prompts)} unique prompts, {len(done)} cached, {len(todo)} to ask', flush=True)
    lock, n = threading.Lock(), [0]
    client = httpx.Client(headers={'Content-Type': 'application/json'},  # the egress proxy injects auth
                          limits=httpx.Limits(max_connections=args.workers))
    with open(args.out, 'a') as f, ThreadPoolExecutor(args.workers) as pool:
        def run(k):
            rec = dict(kind=k[0], text=k[1], at=sorted(prompts[k]), **ask(client, *k))
            with lock:
                f.write(json.dumps(rec) + '\n')
                n[0] += 1
                if n[0] % 500 == 0:
                    f.flush()
                    print(f'{n[0]}/{len(todo)}', flush=True)
        list(pool.map(run, todo))
    print('done', flush=True)


if __name__ == '__main__':
    main()
