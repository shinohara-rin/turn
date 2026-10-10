"""Corpus statistics of a synthetic split, through TurnBench's own gold code,
for comparison against TurnBench Table III and §IV-B (needs the `score` extra)."""

from collections import defaultdict
from statistics import median


def dataset_stats(source: str) -> dict:
    from turnbench.data import conversation, conversation_ids, resolve_dataset
    from turnbench.gold import consensus_for_conversation, events_for_conversation, TURN_CANONICAL

    import pyarrow.parquet as pq
    from pathlib import Path

    dataset = resolve_dataset(source)
    meta = {}
    for shard in sorted(Path(source).glob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "metadata"])
        meta.update(zip(table["conversation_id"].to_pylist(), table["metadata"].to_pylist()))
    per_type = defaultdict(lambda: defaultdict(float))
    ftos: list[float] = []
    for cid in conversation_ids(dataset):
        conv = conversation(dataset, cid)
        ev = events_for_conversation(conv)
        label_events, _ = consensus_for_conversation(conv)
        turn_events, _ = consensus_for_conversation(conv, canonical=TURN_CANONICAL)
        t = per_type[meta[cid]["conversation_type"]]
        t["conversations"] += 1
        t["minutes"] += conv.duration_s / 60
        t["eot_pos"] += len(ev.eot_positive_events)
        t["eot_neg"] += len(ev.eot_negative_spans)
        t["int_pos"] += len(ev.int_positive_events)
        t["int_neg"] += len(ev.int_negative_spans)
        t["excluded"] += len(ev.eot_excluded) + len(ev.int_excluded)
        t["backchannels"] += sum(e.label == "Backchannel" for e in label_events)
        t["agreement"] += meta[cid]["agreement"]
        # Floor-transfer offsets: next other-speaker turn start minus EOT anchor.
        starts = {s: sorted(e.start for e in turn_events if e.speaker == s) for s in (1, 2)}
        for anchor in ev.eot_positive_events:
            other = starts[3 - anchor.speaker]
            nxt = [s for s in other if s > anchor.time_s - 3.0]
            if nxt:
                ftos.append(min(nxt, key=lambda s: abs(s - anchor.time_s)) - anchor.time_s)
    out = {}
    for name, t in sorted(per_type.items()):
        m = t["minutes"]
        out[name] = {
            "conversations": int(t["conversations"]), "minutes": round(m, 1),
            "eot_pos": int(t["eot_pos"]), "eot_neg": int(t["eot_neg"]),
            "int_pos": int(t["int_pos"]), "int_neg": int(t["int_neg"]), "excluded": int(t["excluded"]),
            "bc_per_min": round(t["backchannels"] / m, 2), "int_per_min": round(t["int_pos"] / m, 2),
            "mean_agreement": round(t["agreement"] / t["conversations"], 3),
        }
    out["_fto_median_ms"] = round(1000 * median(ftos)) if ftos else None
    out["_fto_overlap_share"] = round(sum(f < 0 for f in ftos) / len(ftos), 3) if ftos else None
    return out
