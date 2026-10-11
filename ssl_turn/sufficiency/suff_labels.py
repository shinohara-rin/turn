"""Reply-sufficiency labels: has the speaker said enough for the listener to answer?

Rin's question (2026-10-11): instead of raw endpointing, predict when the speech holds enough
content to wake an LLM, including mid-utterance. Operational definition used here: a point in a
turn is *sufficient* when the reply drafted from the words so far is as good a response to the
whole turn as a reply drafted after hearing everything. That differs from "the speaker is done"
(complete_labels.py on the semantics branch): a speaker can keep talking after the point where
the listener's answer stopped changing ("can you send me the file ... the one from yesterday ...
whenever you get a chance"), and can be grammatically done while the reply still depends on
what comes next.

Candidates per turn (a maximal run of one speaker's floor-claiming segments):
  - 'pause': every segment end inside the turn, and the turn end ('end', sufficient by definition)
  - 'mid': clause punctuation inside a segment (the transcriber's , . ? ! roughly mark prosodic
    boundaries); its time is interpolated by character position, so it is approximate.
The LLM sees lower-cased words without punctuation and never the words after a prefix when
drafting. Two batched calls through the RunInfra gateway (the egress proxy injects auth):
  1. draft: a short listener reply from (context, prefix) for every candidate incl. the turn end
  2. judge: 0-10, is reply(prefix) as good a response to the full turn as reply(full)?

Output JSON: {cid: [[speaker, t, kind, score_or_None, words_before, words_total, turn_end_t,
                     prefix_tail, reply], ...]}
Input: the annotation dump from semantics/dump_annotations.py ({'tbdev': ..., 'oto': ...} or a
plain {cid: rows}). Rows are [speaker, ann, start, end, label, text]; only annotator 'a' is used.

    python suff_labels.py ann.json suff_dev.json [--max-conv N]
"""
import argparse
import json
import random
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import httpx

URL = 'https://api.runinfra.ai/v1/chat/completions'
MODEL = 'glm-5-3-flash'
CLAIMS = ('Normal Turn', 'Regular Turn', 'Strong Floor Hold', 'Filler', 'Bounded Response', 'Overlap',
          'Floor-taking Competitive Interruption', 'Floor-taking Cooperative Interruption')
MAX_CAND = 12      # per turn, evenly thinned, the turn end always kept
MIN_WORDS = 2
BATCH = 15

DRAFT = """Below are moments from casual two-person conversations. Each shows the other person's last
line (context) and what the speaker has said so far (lower case, no punctuation). The speaker may or
may not continue. You are the listener: write the one short reply (at most 20 words) you would give
if you answered right now. Always give your best reply, even if little has been said.
Reply with one line per item: "<number>: <reply>".

"""

JUDGE = """Below are moments from casual two-person conversations. For each you get the context, the
speaker's FULL turn, reply A (written by a listener who had only heard the turn up to some earlier
point) and reply B (written after hearing the full turn).
Rate 0-10 how well A works as the response to the FULL turn, compared with B.
10 = A is just as good: the rest of the turn added nothing that changes what the listener should say.
5 = A is on topic but misses something the speaker went on to say or ask.
0 = A is wrong for the full turn (answers the wrong question, ignores the main point, or would
interrupt an unfinished request).
Reply with one line per item: "<number>: <score>".

"""


def norm(t):
    t = unicodedata.normalize('NFKC', t.lower())
    t = re.sub(r'\[[^]]*\]', ' ', t)
    t = ''.join(ch for ch in t if not (unicodedata.category(ch).startswith('P') and ch not in "'-"))
    return re.sub(r'\s+', ' ', t).strip()


def turns(rows):
    """Maximal same-speaker runs of claim segments, ordered by start."""
    segs = sorted((r for r in rows if r[1] == 'a' and r[4] in CLAIMS and norm(r[5])), key=lambda r: r[2])
    out, cur = [], []
    for r in segs:
        if cur and r[0] != cur[-1][0]:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def candidates(turn):
    """(words, [(n_words, t, kind)]) for one turn."""
    words, cands = [], []
    for k, (_, _, a, b, _, text) in enumerate(turn):
        text = re.sub(r'\[[^]]*\]', ' ', text)
        L = max(1, len(text))
        for m in re.finditer(r'[^\s]+', text):
            w = norm(m.group())
            if not w:
                continue
            words.append(w)
            if re.search(r'[,.?!;]$', m.group()) and m.end() < len(text.rstrip()) - 1:
                cands.append((len(words), a + (b - a) * m.end() / L, 'mid'))
        cands.append((len(words), b, 'end' if k == len(turn) - 1 else 'pause'))
    seen, uniq = set(), []
    for c in cands:
        if c[0] >= MIN_WORDS and c[0] not in seen:
            seen.add(c[0])
            uniq.append(c)
    if len(uniq) > MAX_CAND:
        keep = sorted({round(i * (len(uniq) - 2) / (MAX_CAND - 2)) for i in range(MAX_CAND - 1)})
        uniq = [uniq[i] for i in keep] + [uniq[-1]]
    return words, uniq


def call(prompt, lines, parse):
    body = dict(model=MODEL, reasoning_effort='low', temperature=0, max_tokens=8000,
                messages=[{'role': 'user', 'content': prompt + '\n'.join(lines)}])
    for attempt in range(5):
        try:
            r = httpx.post(URL, json=body, timeout=180)
            r.raise_for_status()
            js = r.json()
            got = {int(m[1]): m[2].strip() for m in
                   re.finditer(r'(?m)^\W*(\d+)\W*[:.]\s*(.+)$', js['choices'][0]['message']['content'] or '')}
            if all(k in got for k in range(len(lines))):
                return [parse(got[k]) for k in range(len(lines))], js.get('usage', {}).get('cost', 0) or 0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 ** attempt)
    return [None] * len(lines), 0.0


def score_of(s):
    m = re.match(r'\d+', s)
    return min(10, int(m.group())) if m else None


def spread(items):
    """Order items so that one batch never holds two prefixes of the same turn: otherwise the
    model reads a later prefix in a neighbouring item and the draft leaks words it must not see."""
    rank, seen = [], {}
    for it in items:
        key = (it['cid'], it['s'], it['end'])
        rank.append(seen.get(key, 0))
        seen[key] = rank[-1] + 1
    rnd = random.Random(0)
    order = sorted(range(len(items)), key=lambda i: (rank[i], rnd.random()))
    return order


def run_batches(prompt, items, fmt, parse, workers=24):
    order = spread(items)
    chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    cost, out = 0.0, [None] * len(items)
    with ThreadPoolExecutor(workers) as pool:
        for n, (ch, (res, c)) in enumerate(zip(chunks, pool.map(
                lambda ch: call(prompt, [f'{k}. {fmt(items[i])}' for k, i in enumerate(ch)], parse), chunks))):
            for i, r in zip(ch, res):
                out[i] = r
            cost += c
            if n % 50 == 0:
                print(f'  {n}/{len(chunks)} ${cost:.3f}', flush=True)
    return out, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--max-conv', type=int, default=0)
    a = ap.parse_args()
    data = json.load(open(a.src))
    if 'tbdev' in data and 'oto' in data:
        data = {**{f'tb/{k}': v for k, v in data['tbdev'].items()}, **{f'oto/{k}': v for k, v in data['oto'].items()}}
    cids = sorted(data)[:a.max_conv or None]

    items = []   # one per candidate
    for cid in cids:
        prev_other = {1: '', 2: ''}
        for turn in turns(data[cid]):
            s = turn[0][0]
            words, cands = candidates(turn)
            ctx = prev_other[s]
            full = ' '.join(words)
            for n, t, kind in cands:
                items.append(dict(cid=cid, s=s, t=t, kind=kind, n=n, N=len(words), end=turn[-1][3], ctx=ctx,
                                  prefix=' '.join(words[:n]), full=full))
            prev_other[3 - s] = full
    print(len(items), 'candidates,', sum(i['kind'] != 'end' for i in items), 'before turn ends', flush=True)

    fmt_d = lambda it: f'context: "{it["ctx"][-200:] or "(none)"}"  speaker so far: "{" ".join(it["prefix"].split()[-80:])}"'  # noqa: E731
    replies, c1 = run_batches(DRAFT, items, fmt_d, lambda x: x.strip('" '))
    for it, r in zip(items, replies):
        it['reply'] = r
    full_reply = {(it['cid'], it['s'], it['end']): it['reply'] for it in items if it['kind'] == 'end'}

    todo = [it for it in items if it['kind'] != 'end' and it['reply']
            and full_reply.get((it['cid'], it['s'], it['end']))]
    fmt_j = lambda it: (f'context: "{it["ctx"][-200:] or "(none)"}"  full turn: "{" ".join(it["full"].split()[-150:])}"  '  # noqa: E731
                        f'A: "{it["reply"]}"  B: "{full_reply[(it["cid"], it["s"], it["end"])]}"')
    scores, c2 = run_batches(JUDGE, todo, fmt_j, score_of)
    for it, sc in zip(todo, scores):
        it['score'] = sc

    out = {}
    for it in items:
        sc = 10 if it['kind'] == 'end' else it.get('score')
        out.setdefault(it['cid'], []).append([it['s'], round(it['t'], 3), it['kind'], sc, it['n'], it['N'],
                                              it['end'], ' '.join(it['prefix'].split()[-12:]), it['reply']])
    json.dump(out, open(a.dst, 'w'))
    miss = sum(1 for it in todo if it.get('score') is None)
    print(f'done: {len(items)} candidates, {miss} unscored, ${c1 + c2:.3f}')


if __name__ == '__main__':
    main()
