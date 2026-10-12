from turnsynth.fish import map_segments


def seg(text, start, end):
    return {"text": text, "start": start, "end": end}


def test_map_segments_english_punctuation_and_contractions():
    words = ["Oh,", "you're", "back!", "How", "was", "it?"]
    segs = [seg("Oh", 0.0, 0.2), seg("you're", 0.3, 0.5), seg("back", 0.5, 0.8),
            seg("How", 1.0, 1.1), seg("was", 1.1, 1.3), seg("it", 1.3, 1.5)]
    out = map_segments(words, segs)
    assert [w.text for w in out] == words
    assert [(w.start, w.end) for w in out] == [(s["start"], s["end"]) for s in segs]


def test_map_segments_cjk_splits_a_segment_by_characters():
    words = ["昨日", "さ", "、", "駅", "前"]
    out = map_segments(words, [seg("昨日さ", 0.0, 0.6), seg("駅前", 0.8, 1.2)], "ja")
    assert abs(out[0].end - 0.4) < 1e-9 and abs(out[1].start - 0.4) < 1e-9
    assert out[2].start == out[2].end == out[1].end  # punctuation takes no time
    assert abs(out[4].start - 1.0) < 1e-9


def test_map_segments_gives_up_when_text_was_normalized_away():
    assert map_segments(["it", "costs", "15", "dollars"], [seg("it", 0, .1), seg("costs", .1, .3),
                        seg("fifteen", .3, .6), seg("dollars", .6, .9)]) is None


def test_map_segments_rejects_a_skipped_word():
    words = ["one"] * 11
    segs = [seg("one", 0.3 * i, 0.3 * i + 0.25) for i in range(10)]
    assert map_segments(words, segs) is None
