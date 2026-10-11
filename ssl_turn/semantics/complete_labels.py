"""Semantic completion labels for otoSpeech pauses (Rin's "complete vs timeout" split).

Each pause point is the end of a floor-claiming segment of speaker c (annotator a) after which c
stays quiet for >= 0.16 s. An LLM sees the last few lines of the dialogue and what c has said in
the current turn, lower-cased and without punctuation (the transcriber's punctuation would give
the answer away), and rates whether c has finished what they meant to say, 0-10. Hesitations,
cut-off clauses and trailing fillers ("let me uh") are incomplete however long the silence that
follows; "that's all thanks" is complete. Words after the pause are never shown.

Output: {cid: [[speaker, end_s, score_0_10, next_speaker_or_0, gap_s], ...]} (JSON). Uses the
RunInfra gateway (the egress proxy injects auth).

    python complete_labels.py oto_train.json comp_train.json
"""
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import httpx

URL = 'https://api.runinfra.ai/v1/chat/completions'
MODEL = 'glm-5-3-flash'   # reasoning 'low'; picked over nemotron (no-reasoning scores were near random)
CLAIMS = ('Normal Turn', 'Regular Turn', 'Strong Floor Hold', 'Filler', 'Bounded Response', 'Overlap',
          'Floor-taking Competitive Interruption', 'Floor-taking Cooperative Interruption',
          'Non-floor Taking Competitive Interruption', 'Non-floor Taking Cooperative Interruption')
VOCAL = CLAIMS + ('Reaction Backchannel', 'Acknowledgement Backchannel', 'Continuer Backchannel', 'Laughter',
                  'Speech, Non-Linguistic')
MIN_PAUSE = 0.16
BATCH = 20
PROMPT = """Below are snippets from casual phone conversations. Each shows the other person's last line
(context) and what one speaker has said so far in their current turn (no punctuation). The speaker
then stopped talking; you do not know for how long or what comes next.
For each, rate 0-10 how likely the speaker is DONE with their turn (10 = finished thought, the listener
should respond now; 0 = obviously unfinished: cut off mid-sentence, trailing "and"/"the"/"um"/"uh",
hesitation, a list or story clearly continuing).
Reply with one line per item: "<number>: <score>".

"""


def norm(t):
    t = unicodedata.normalize('NFKC', t.lower())
    t = re.sub(r'\[[^]]*\]', ' ', t)
    t = ''.join(ch for ch in t if not (unicodedata.category(ch).startswith('P') and ch not in "'-"))
    return re.sub(r'\s+', ' ', t).strip()


def pauses(rows):
    """rows: [speaker, ann, start, end, label, text]. Returns pause items with context."""
    rows = sorted((r for r in rows if r[4] in VOCAL), key=lambda r: r[2])
    out = []
    for i, (s, _, a, b, label, text) in enumerate(rows):
        if label not in CLAIMS or not norm(text):
            continue
        own_next = min((r[2] for r in rows[i + 1:] if r[0] == s), default=None)
        if own_next is not None and own_next < b + MIN_PAUSE:
            continue  # no real pause
        # current turn: c's claim segments back to the last claim by the other speaker before it
        turn, earliest = [norm(text)], a
        for r in reversed(rows[:i]):
            if r[4] not in CLAIMS:
                continue
            if r[0] == s:
                turn.append(norm(r[5]))
                earliest = r[2]
            elif r[3] <= earliest + 0.01:
                break
        turn = ' '.join(reversed(turn))
        ctx = [r for r in rows[:i] if r[3] <= a and r[4] in CLAIMS and r[0] != s][-3:]
        nxt = [r for r in rows[i + 1:] if r[2] >= b - 0.5 and r[4] in CLAIMS]
        nxt_spk = nxt[0][0] if nxt else 0
        gap = (nxt[0][2] - b) if nxt else None
        out.append(dict(speaker=s, end=b, turn=' '.join(turn.split()[-60:]), ctx=[norm(r[5]) for r in ctx],
                        next=nxt_spk, gap=gap))
    return out


def ask(items):
    lines = []
    for k, it in enumerate(items):
        ctx = it['ctx'][-1] if it['ctx'] else '(none)'
        lines.append(f'{k}. context: "{ctx[-150:]}"  speaker so far: "{it["turn"][-250:]}"')
    body = dict(model=MODEL, reasoning_effort='low', temperature=0, max_tokens=8000,
                messages=[{'role': 'user', 'content': PROMPT + '\n'.join(lines)}])
    for attempt in range(5):
        try:
            r = httpx.post(URL, json=body, timeout=120)
            r.raise_for_status()
            txt = r.json()['choices'][0]['message']['content']
            got = {int(m[1]): int(m[2]) for m in re.finditer(r'(?m)^\W*(\d+)\W*[:.]\s*(\d+)', txt or '')}
            if all(k in got for k in range(len(items))):
                return [min(10, got[k]) for k in range(len(items))], r.json()['usage'].get('cost', 0)
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(2 ** attempt)
    return [None] * len(items), 0.0


def main(src, dst):
    data = json.load(open(src))
    if 'oto' in data and isinstance(data['oto'], dict) and 'tbdev' in data:
        data = data['oto']
    jobs = []
    for cid, rows in data.items():
        items = pauses(rows)
        for i in range(0, len(items), BATCH):
            jobs.append((cid, items[i:i + BATCH]))
    print(len(jobs), 'calls,', sum(len(j[1]) for j in jobs), 'pauses', flush=True)
    out, cost = {}, 0.0
    with ThreadPoolExecutor(24) as pool:
        for n, ((cid, items), (scores, c)) in enumerate(zip(jobs, pool.map(lambda j: ask(j[1]), jobs))):
            cost += c
            for it, sc in zip(items, scores):
                out.setdefault(cid, []).append([it['speaker'], it['end'], sc, it['next'], it['gap'], it['turn'][-80:]])
            if n % 100 == 0:
                print(n, f'${cost:.3f}', flush=True)
    json.dump(out, open(dst, 'w'))
    miss = sum(1 for v in out.values() for x in v if x[2] is None)
    print('done', sum(len(v) for v in out.values()), 'pauses,', miss, 'unscored,', f'${cost:.3f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
