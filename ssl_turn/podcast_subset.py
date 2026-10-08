"""Deterministic, feed-disjoint subsets of the DuplexChat manifest.

DuplexChat (sarulab-speech/DuplexChat, arXiv 2607.04941) lists ~15.3M English
two-speaker podcast clips (282k h) as URLs + spans; audio is rebuilt with their
MIT pipeline (scripts/reconstruct_dataset.py). Podcast hosts recur across a
feed's episodes, so held-out data must be split by feed (rss_url), never by
clip. Per-feed and per-episode hour caps keep a few prolific shows from
dominating the speaker distribution.

Output rows keep the upstream schema, so the subset feeds the upstream
reconstruction script unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict

DUPLEXCHAT_REPO = 'sarulab-speech/DuplexChat'
DUPLEXCHAT_REVISION = 'ff5c4181a8c6dd7a771361d5165951242d0157ae'


def _unit(key, salt):
    return int(hashlib.sha256(f'{salt}:{key}'.encode()).hexdigest()[:15], 16) / 16 ** 15


def feed_partition(rss_url, val_fraction, salt='duplexchat-v1'):
    return 'val' if _unit(rss_url, salt + ':split') < val_fraction else 'train'


def select(rows, hours, partition='train', val_fraction=0.02, feed_cap_h=20.0, episode_cap_h=2.0,
           min_s=10.0, max_s=600.0, salt='duplexchat-v1'):
    """Pick clips in a stable pseudo-random order until `hours` are reached.

    Order is a hash of (audio_url, dialogue_idx), so growing `hours` yields a
    superset of a smaller selection, and scaling experiments stay nested.
    """
    scored = []
    for r in rows:
        if not (min_s <= r['duration_sec'] <= max_s):
            continue
        if feed_partition(r['rss_url'], val_fraction, salt) != partition:
            continue
        scored.append((_unit(f"{r['audio_url']}#{r['dialogue_idx']}", salt), r))
    scored.sort(key=lambda x: x[0])
    feed_s, episode_s, out, total = defaultdict(float), defaultdict(float), [], 0.0
    for _, r in scored:
        d = r['duration_sec']
        if feed_s[r['rss_url']] + d > feed_cap_h * 3600 or episode_s[r['audio_url']] + d > episode_cap_h * 3600:
            continue
        feed_s[r['rss_url']] += d
        episode_s[r['audio_url']] += d
        out.append(r)
        total += d
        if total >= hours * 3600:
            break
    return out


def read_manifest(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', help='duplexchat_manifest_en.jsonl.gz at the pinned revision')
    ap.add_argument('output')
    ap.add_argument('--hours', type=float, required=True)
    ap.add_argument('--partition', choices=('train', 'val'), default='train')
    ap.add_argument('--val-fraction', type=float, default=0.02)
    ap.add_argument('--feed-cap-h', type=float, default=20.0)
    ap.add_argument('--episode-cap-h', type=float, default=2.0)
    args = ap.parse_args()
    rows = select(read_manifest(args.manifest), args.hours, args.partition, args.val_fraction,
                  args.feed_cap_h, args.episode_cap_h)
    with gzip.open(args.output, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(json.dumps(dict(source=f'{DUPLEXCHAT_REPO}@{DUPLEXCHAT_REVISION}', partition=args.partition,
                          clips=len(rows), hours=sum(r['duration_sec'] for r in rows) / 3600,
                          feeds=len({r['rss_url'] for r in rows}),
                          episodes=len({r['audio_url'] for r in rows}))))


if __name__ == '__main__':
    main()
