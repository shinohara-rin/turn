"""Per-language conventions: English, Japanese and Mandarin Chinese.

Scripts in every language are written with a single space between words,
because backchannels and interruptions are anchored by word count
(`after_word`). For Japanese and Chinese, which are written without spaces,
the generator LLM segments the text itself ("昨日 さ、 駅 で 友達 に 会っ た"),
and the spaces are taken out again before the text reaches the TTS
(`spoken`). ASR accuracy is then a character error rate (`units`; over the
hiragana reading for Japanese, so kanji/kana spelling choices don't count),
and forced alignment works on a romanization of each word (`romanize`:
pypinyin for Chinese, pykakasi for Japanese), since the MMS aligner's
character set is Latin.
"""

import re
import unicodedata

LANGUAGES = ("en", "ja", "zh")
CJK = ("ja", "zh")

COMMA = {"en": ",", "ja": "、", "zh": "，"}
ELLIPSIS = {"en": "...", "ja": "…", "zh": "……"}
SENTENCE_END = ".?!…。？！"
CLAUSE_END = ",;:-、，；："

_ASCII_WORD = re.compile(r"[A-Za-z0-9]")


def check(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"language {lang!r} not in {LANGUAGES}")
    return lang


def spoken(words: list[str], lang: str) -> str:
    """The text a TTS reads: words joined by spaces, or for ja/zh run together
    (keeping a space only between two Latin-script words, as in "iPhone 15")."""
    if lang not in CJK:
        return " ".join(words)
    out = ""
    for w in words:
        if out and _ASCII_WORD.match(out[-1]) and _ASCII_WORD.match(w[0]):
            out += " "
        out += w
    return out


def units(text: str, lang: str) -> list[str]:
    """What an error rate counts: lowercased words for English, characters for ja/zh
    (punctuation and spaces dropped; full-width forms folded by NFKC)."""
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "ja":
        # Compare readings: ASR and script often differ only in kanji vs kana spelling.
        return [c for c in hiragana(text) if c.isalnum()]
    if lang in CJK:
        return [c for c in text if c.isalnum()]
    return ["".join(c for c in w if c.isalnum()) for w in text.split() if any(c.isalnum() for c in w)]


def bare(text: str, lang: str) -> str:
    """Text with spaces and punctuation removed, for matching short tokens like backchannels."""
    return "".join(units(text, lang)) if lang in CJK else " ".join(units(text, lang))


_kks = None


def _kakasi():
    global _kks
    if _kks is None:
        import pykakasi

        _kks = pykakasi.kakasi()
    return _kks


def hiragana(text: str) -> str:
    """Japanese text as its hiragana reading (pykakasi), or unchanged if pykakasi is missing."""
    try:
        return "".join(p["hira"] for p in _kakasi().convert(text))
    except ImportError:
        return text


def romanize(words: list[str], lang: str) -> list[str]:
    """Latin-letter reading of each word, for the MMS forced aligner.

    Japanese is read as a whole sentence, since a reading depends on what
    follows ("行っ" alone reads "itsu", in "行った" it is "it"); each reading
    piece goes to the word holding its first character.
    """
    if lang == "zh":
        from pypinyin import lazy_pinyin

        return ["".join(lazy_pinyin(w, errors="ignore")) for w in words]
    if lang != "ja":
        return list(words)
    kks = _kakasi()
    owner = [i for i, w in enumerate(words) for _ in w]  # character -> word index
    out = [""] * len(words)
    pos = 0
    for piece in kks.convert("".join(words)):
        orig, roman = piece["orig"], piece["hepburn"]
        idx = owner[pos: pos + len(orig)]
        pos += len(orig)
        # A piece can run across words ("行った" over 行っ|た, "んだけど" over ん|だ|けど).
        # Later parts are kana, read on their own; the first part keeps the rest.
        parts = [(i, "".join(c for c, j in zip(orig, idx) if j == i)) for i in dict.fromkeys(idx)]
        for i, part in reversed(parts[1:]):
            r = "".join(x["hepburn"] for x in kks.convert(part)) if _KANA.fullmatch(part) else ""
            if r and roman.endswith(r) and len(r) < len(roman):
                out[i], roman = r, roman[: -len(r)]
            else:  # no clean split: share it by length
                k = round(len(roman) * len(part) / len(orig))
                out[i], roman = roman[len(roman) - k:], roman[: len(roman) - k]
            orig = orig[: -len(part)]
        if parts:
            out[parts[0][0]] += roman
    return out


_KANA = re.compile(r"[\u3040-\u30ffー]+")


# Listener tokens the offline rule judge treats as backchannels, after `bare`.
BACKCHANNELS = {
    "ja": {
        "continuer": {"うん", "うんうん", "ん", "んー", "ふん", "ふーん", "はい", "ええ", "ああ", "あー"},
        "reaction": {"へえ", "へー", "えー", "えっ", "ほんと", "ほんとに", "まじ", "まじで", "うそ", "すごい", "すごっ", "おお", "わあ", "あら", "まさか"},
        "acknowledgement": {"そう", "そうそう", "そうね", "そうだね", "そうですね", "そっか", "なるほど", "たしかに", "確かに", "だよね", "ですよね", "わかる", "了解", "はいはい"},
    },
    "zh": {
        "continuer": {"嗯", "嗯嗯", "嗯哼", "哦", "噢", "啊", "呃"},
        "reaction": {"哇", "真的", "真的吗", "天哪", "我的天", "不会吧", "是吗", "啊", "哈哈", "哎呀", "厉害"},
        "acknowledgement": {"对", "对对", "对对对", "是", "是的", "是啊", "好", "好的", "行", "没错", "明白", "知道了", "可以", "就是"},
    },
}
FILLERS = {
    "ja": {"えーと", "えっと", "あの", "あのー", "まあ", "なんか", "その", "えー", "うーん"},
    "zh": {"那个", "就是", "然后", "嗯", "呃", "这个", "那"},
}
