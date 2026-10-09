"""Text-oracle bound: causal transcript -> LLM yes/no judgements at decision points.

Annotator transcripts stand in for a perfect streaming ASR. Word j of an n-word segment
is visible at start + LATENCY + (j / n) * duration (no word timestamps exist; uniform
reveal). Only text and timing are used, never annotation labels. Queries:

  eot: at each non-empty segment end + LATENCY, "has this speaker finished their turn?"
  int: for segments starting while the other speaks (or <= 0.3 s after), at 1/2/4 words and
       the full segment, "is this speaker taking the floor (vs. listener feedback)?"

Answers come from a hosted LLM (RunInfra gateway, thinking disabled), scored as
P(yes) / (P(yes) + P(no)) from first-token logprobs. Results are cached per query in JSONL,
so runs resume. Transcripts stay local; nothing here is committed.

    python llm_oracle.py transcripts.json out.jsonl [--limit N]
"""
import argparse
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

LATENCY = 0.3
MODEL = 'nemotron-3-5-lightning-30b'
URL = 'https://api.runinfra.ai/v1/chat/completions'
CONTEXT = 10  # most recent visible lines
SYSTEM = ('You judge turn-taking in a live, verbatim transcript of a spoken two-person conversation '
          '(fillers and false starts are kept). Answer with a single word: yes or no.')


def visible(seg, t):
    """Words of segment [start, end, text] visible at time t (uniform reveal + latency)."""
    a, b, text = seg
    words = text.split()
    if not words or t < a + LATENCY:
        return []
    k = len(words) if t >= b + LATENCY else int(len(words) * (t - a - LATENCY) / max(b - a, 1e-3))
    return words[:max(0, min(len(words), k))]


def context(conv, t, target, exclude=None):
    """Transcript lines visible at t, ordered by segment start; target speaker shown as A."""
    rows = []
    for s, segs in conv.items():
        for seg in segs:
            if seg is exclude:
                continue
            w = visible(seg, t)
            if w:
                rows.append((seg[0], 'A' if int(s) == target else 'B', ' '.join(w) + ('' if t >= seg[1] + LATENCY else ' ...')))
    rows.sort()
    return '\n'.join(f'{who}: {txt}' for _, who, txt in rows[-CONTEXT:])


def queries(transcripts):
    for split, convs in transcripts.items():
        for cid, conv in convs.items():
            for s, segs in conv.items():
                s = int(s)
                other = conv[str(3 - s)] if str(3 - s) in conv else conv[3 - s]
                for i, seg in enumerate(segs):
                    a, b, text = seg
                    if not text.split():
                        continue
                    t = b + LATENCY
                    ctx = context(conv, t, s)
                    yield dict(id=f'{split}/{cid}/{s}/{i}/eot', split=split, cid=cid, speaker=s, kind='eot', t=t,
                               prompt=f'Transcript so far:\n{ctx}\n\nA has just paused. Has A finished their turn, so '
                                      f'that B is expected to respond now? (Answer no if A is likely to keep talking '
                                      f'after this pause.)')
                    overlapping = any(o[0] <= a <= o[1] + 0.3 for o in other if o[2].split())  # other is speaking
                    if not overlapping:
                        continue
                    n = len(text.split())
                    for k in sorted({min(n, 1), min(n, 2), min(n, 4), n}):
                        tk = a + LATENCY + (k / n) * (b - a)
                        prefix = ' '.join(text.split()[:k])
                        ctx = context(conv, tk, s, exclude=seg)
                        done = '' if k == n else ' ...'
                        yield dict(id=f'{split}/{cid}/{s}/{i}/int{k}', split=split, cid=cid, speaker=s, kind='int',
                                   t=tk, words=k, prompt=f'Transcript so far:\n{ctx}\n\nA now starts speaking while B '
                                   f'has the floor. A has said so far: "{prefix}{done}"\n\nIs A taking the floor (starting '
                                   f'their own turn or interrupting B), rather than giving brief listener feedback '
                                   f"(like 'yeah', 'mhm', 'right', 'oh wow') while B keeps talking?")


def ask(client, prompt):
    body = dict(model=MODEL, messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}],
                max_tokens=1, temperature=0, logprobs=True, top_logprobs=10, reasoning_effort='none')
    for attempt in range(5):
        try:
            r = client.post(URL, json=body, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(r.status_code)
            r.raise_for_status()
            top = r.json()['choices'][0]['logprobs']['content'][0]['top_logprobs']
            yes = sum(math.exp(x['logprob']) for x in top if x['token'].strip().lower() == 'yes')
            no = sum(math.exp(x['logprob']) for x in top if x['token'].strip().lower() == 'no')
            return yes / (yes + no) if yes + no > 0 else None
        except Exception:
            import time
            time.sleep(2 ** attempt)
    return None


def main():
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument('transcripts')
    ap.add_argument('out')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=32)
    args = ap.parse_args()
    done = set()
    if os.path.exists(args.out):
        done = {r['id'] for r in map(json.loads, open(args.out)) if r.get('p') is not None}  # retry failures
    todo = [q for q in queries(json.load(open(args.transcripts))) if q['id'] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(done)} cached, {len(todo)} to ask', flush=True)
    lock, n = threading.Lock(), [0]
    client = httpx.Client(headers={'Content-Type': 'application/json'})
    with open(args.out, 'a') as f, ThreadPoolExecutor(args.workers) as pool:
        def run(q):
            p = ask(client, q['prompt'])
            rec = {k: v for k, v in q.items() if k != 'prompt'}
            rec['p'] = p
            with lock:
                f.write(json.dumps(rec) + '\n')
                n[0] += 1
                if n[0] % 1000 == 0:
                    f.flush()
                    print(f'{n[0]}/{len(todo)}', flush=True)
        list(pool.map(run, todo))
    print('done', flush=True)


if __name__ == '__main__':
    main()
