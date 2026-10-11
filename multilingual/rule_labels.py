"""Turn-taking labels from per-speaker speech timing alone, in otoSpeech's own label convention.

CallHome (and most corpora in other languages) gives only who speaks when. otoSpeech's targets come
from per-segment TurnBench labels (Turn, Backchannel, Interruption, ...) passed through the pinned
TurnBench gold builder (`build_conversation_events`), which decides where the floor passes. This
module assigns each timed segment a canonical label with simple rules, then runs the *same* builder,
so EOT anchors, holds and floor targets follow oto's construction exactly:

  - Same-speaker segments closer than MERGE_S are merged.
  - Backchannel: shorter than BC_MAX_S, while the other speaker holds the floor (they spoke within
    BC_CONTEXT_S before it starts and speak again within BC_CONTEXT_S after it ends), and the same
    speaker does not go on talking within BC_CONTINUE_S.
  - Everything else is a Turn. No interruption labels: overlaps become hand-offs or co-talk
    (timing cannot tell interruptions apart; see the hindsight annotator results).

`validate` scores the rules on otoSpeech, where human labels exist: strip the labels, relabel the
speech segments by rule, and compare the builder's EOT events and backchannel calls with the human
ones.

    python multilingual/rule_labels.py validate SRT_DIR     # SRT_DIR/<cid>_<1|2>.srt
"""
import re
import sys

MERGE_S = 0.1
BC_MAX_S = 1.0
BC_CONTEXT_S = 1.0
BC_CONTINUE_S = 0.5
NONSPEECH = ('Awkward Silence', 'Non-Speech Noise', 'Channel Bleed')


def merge(segments, gap=MERGE_S):
    """segments: [(speaker, start, end)] -> merged per speaker, sorted by start."""
    out = []
    for s in (1, 2):
        cur = None
        for _, a, b in sorted(x for x in segments if x[0] == s):
            if cur and a - cur[2] < gap:
                cur[2] = max(cur[2], b)
            else:
                cur and out.append(tuple(cur))
                cur = [s, a, b]
        cur and out.append(tuple(cur))
    return sorted(out, key=lambda x: x[1])


def label(segments):
    """[(speaker, start, end)] -> [(speaker, start, end, canonical label)]."""
    segs = merge(segments)
    by = {s: [x for x in segs if x[0] == s] for s in (1, 2)}
    out = []
    for s, a, b in segs:
        other = by[3 - s]
        before = any(oa < a and ob > a - BC_CONTEXT_S for _, oa, ob in other)
        after = any(ob > b and oa < b + BC_CONTEXT_S for _, oa, ob in other)
        goes_on = any(a2 > b and a2 < b + BC_CONTINUE_S for _, a2, _ in by[s])
        bc = b - a < BC_MAX_S and before and after and not goes_on
        out.append((s, a, b, 'Backchannel' if bc else 'Turn'))
    return out


def events(labeled):
    """Run the pinned TurnBench gold builder on rule-labeled segments, as prep.oto_item does."""
    from dataclasses import asdict
    from turnbench.gold import ConsensusEvent, ConsensusViews, build_conversation_events
    fine = [ConsensusEvent(s, a, b, lab) for s, a, b, lab in labeled]
    turn = [ConsensusEvent(s, a, b, 'Turn') for s, a, b, lab in labeled if lab == 'Turn']
    return asdict(build_conversation_events(ConsensusViews(turn, [], fine, [])))


# ---------------------------------------------------------------- validation on otoSpeech

def parse_srt(path):
    import srt
    segs = []
    for e in srt.parse(open(path).read()):
        m = re.match(r'\[([^]]+)\]\s*(.*)', e.content, re.S)
        segs.append((e.start.total_seconds(), e.end.total_seconds(), m[1]))
    return segs


def human(srt_dir, cid):
    """-> (human canonical segments, speech segments without labels) for one oto conversation."""
    from turnbench.gold import CANONICAL
    segs, speech = [], []
    for s in (1, 2):
        for a, b, lab in parse_srt(f'{srt_dir}/{cid}_{s}.srt'):
            if lab in CANONICAL:
                segs.append((s, a, b, CANONICAL[lab]))
            if lab not in NONSPEECH and CANONICAL.get(lab) != 'NonContent':
                speech.append((s, a, b))
    return segs, speech


def match(pred, gold, tol):
    """Greedy one-to-one matching of (speaker, time) events within tol seconds -> hits."""
    used, hits = set(), 0
    for s, t in sorted(pred, key=lambda x: x[1]):
        best = None
        for j, (gs, gt) in enumerate(gold):
            if j not in used and gs == s and abs(gt - t) <= tol and (best is None or abs(gt - t) < abs(gold[best][1] - t)):
                best = j
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def validate(srt_dir, tol=0.3):
    import glob
    import os
    from turnbench.gold import ConsensusEvent, ConsensusViews, build_conversation_events
    from dataclasses import asdict
    cids = sorted({os.path.basename(p).rsplit('_', 1)[0] for p in glob.glob(f'{srt_dir}/*_1.srt')
                   if os.path.exists(p[:-6] + '_2.srt')})
    tot = dict(gold=0, pred=0, hit=0, bc_gold=0, bc_pred=0, bc_hit=0)
    for cid in cids:
        segs, speech = human(srt_dir, cid)
        turn = [ConsensusEvent(s, a, b, 'Turn') for s, a, b, lab in segs if lab in ('Turn', 'Interruption')]
        gold = asdict(build_conversation_events(ConsensusViews(turn, [], [ConsensusEvent(*x) for x in segs], [])))
        lab = label(speech)
        pred = events(lab)
        g = [(e['speaker'], e['time_s']) for e in gold['eot_positive_events']]
        p = [(e['speaker'], e['time_s']) for e in pred['eot_positive_events']]
        h = match(p, g, tol)
        # backchannels: a rule segment counts as found if it overlaps a human Backchannel of that speaker by half
        hb = [(s, a, b) for s, a, b, l in segs if l == 'Backchannel']
        pb = [(s, a, b) for s, a, b, l in lab if l == 'Backchannel']
        bh = sum(any(s == s2 and min(b, b2) - max(a, a2) >= 0.5 * (b - a) for s2, a2, b2 in hb) for s, a, b in pb)
        for k, v in dict(gold=len(g), pred=len(p), hit=h, bc_gold=len(hb), bc_pred=len(pb), bc_hit=bh).items():
            tot[k] += v
        print(f'{cid}: EOT gold {len(g)} rule {len(p)} hit {h}; BC gold {len(hb)} rule {len(pb)} hit {bh}')
    print(f'{len(cids)} conversations: EOT precision {tot["hit"] / max(tot["pred"], 1):.3f} '
          f'recall {tot["hit"] / max(tot["gold"], 1):.3f} (tol {tol} s); backchannel precision '
          f'{tot["bc_hit"] / max(tot["bc_pred"], 1):.3f}, human backchannels {tot["bc_gold"]} vs rule {tot["bc_pred"]}')
    return tot


if __name__ == '__main__' and len(sys.argv) > 2 and sys.argv[1] == 'validate':
    validate(sys.argv[2])
