"""Raw annotator segments (label + text) for TurnBench dev (annotators a/b/c) and otoSpeech dev
(annotator a), for backchannel analysis. Never committed: stays local / in the scratchpad.

    modal run ssl_turn/semantics/dump_annotations.py --out ann.json
"""
import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from common import OTO, TB_DEV, VOLUMES, WORK, cpu_image  # noqa: E402

app = modal.App('ssl-turn-anndump')


@app.function(image=cpu_image, volumes=VOLUMES, cpu=4, memory=32768, timeout=1800)
def dump():
    import re
    import srt
    from turnbench.data import conversation, conversation_ids, resolve_dataset
    out = {'tbdev': {}, 'oto': {}}
    ds = resolve_dataset(TB_DEV, skip_audio=True)
    for cid in conversation_ids(ds):
        ann = conversation(ds, cid).annotations
        out['tbdev'][cid] = [[s, a, x[0], x[1], x[2], x[3] if len(x) > 3 else '']
                             for (s, a), segs in ann.items() for x in segs]
    split = json.load(open(f'{WORK}/split.json'))['splits']
    for cid in split['dev']:
        rows = []
        for s in (1, 2):
            for e in srt.parse(open(f'{OTO}/{cid}/speaker_{s}_annotation_a.srt').read()):
                m = re.match(r'\[([^]]+)\]\s*(.*)', e.content, re.S)
                rows.append([s, 'a', e.start.total_seconds(), e.end.total_seconds(), m[1] if m else '',
                             (m[2] if m else e.content).strip()])
        out['oto'][cid] = rows
    return out


@app.local_entrypoint()
def main(out: str = 'ann.json'):
    json.dump(dump.remote(), open(out, 'w'))
    print('wrote', out)
