"""Find TTS glitches (a noise burst, a laugh run into words) inside a line.

TorchAudio's SQUIM estimates PESQ without a reference. Over a whole line a
short glitch averages away, so lines are scored on 2 s windows and the
worst window counts: on the casual example the two lowest windows were the
two spots a listener flagged (a noise under "I'm serious", a laugh pressed
against speech), at 1.5-2.6 against 2.8-3.9 for the rest.
"""

import numpy as np

WINDOW_S, HOP_S, MIN_S = 2.0, 0.5, 1.0
_model = None


def worst_window(audio: np.ndarray, sr: int) -> float | None:
    """Lowest windowed SQUIM PESQ of a clip, or None if it is too short to judge."""
    global _model
    import torch
    import torchaudio

    if len(audio) < MIN_S * sr:
        return None
    if _model is None:
        _model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().eval()
    x = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(audio, np.float32)), sr, 16000)
    w, h = int(WINDOW_S * 16000), int(HOP_S * 16000)
    starts = range(0, max(1, len(x) - w + 1), h)
    with torch.inference_mode():
        return min(_model(x[i: i + w][None])[1].item() for i in starts)
