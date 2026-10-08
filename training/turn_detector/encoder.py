"""Causal frozen Parakeet features, intended to run on the remote Colab VM.

ParakeetStreamingEncoder is the primary cache-streaming implementation; its
centered-STFT right context is charged to availability time. The alternative
ParakeetPrefixEncoder reruns preprocessing/encoder on past windows only and is
more expensive. Never backdate either feature sequence to encoder-frame time.

Primary API references (inspected 2026-10-07):
https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/models/rnnt_models.py
https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py
https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1
"""
from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np


@dataclass
class FeatureSequence:
    features: np.ndarray  # float32 [decisions, channel=2, encoder_dim]
    available_at_s: np.ndarray  # float64, exclusive audio endpoint
    metadata: dict[str, Any]


def causal_resample(audio: np.ndarray, sample_rate: int, target_rate: int = 16000):
    """One-sided FIR, no delay compensation and no flush into future time.

    Output k uses only source samples <= floor(k*sample_rate/target_rate).
    scipy.resample_poly is intentionally avoided: it compensates FIR delay.
    Returns waveform and the retained filter's signal delay (seconds).
    """
    from scipy.signal import firwin, upfirdn

    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("audio must be [samples, 2]; do not infer channel layout")
    if sample_rate <= 0 or int(sample_rate) != sample_rate:
        raise ValueError("sample_rate must be a positive integer")
    if not np.isfinite(x).all():
        raise ValueError("audio contains nonfinite values")
    if sample_rate == target_rate or len(x) == 0:
        return x.copy(), 0.0
    divisor = gcd(int(sample_rate), int(target_rate))
    up, down = target_rate // divisor, sample_rate // divisor
    half_len = 10 * max(up, down)
    taps = firwin(2 * half_len + 1, 1 / max(up, down), window=("kaiser", 5.0)) * up
    count = (len(x) * up + down - 1) // down
    y = upfirdn(taps, x, up=up, down=down, axis=0)[:count]
    return y.astype(np.float32), half_len / (sample_rate * up)


class ParakeetPrefixEncoder:
    """Frozen last-frame embeddings every 160 ms, with channels independent.

    Short prefixes are left-padded with zeros to a fixed context window before
    preprocessing. Batch padding therefore cannot leak later audio. No global
    loudness normalization is applied. NeMo's own normalization is per window.
    Last-frame embeddings have artificial right boundaries: measure their quality
    rather than claiming equivalence to Ooma or to cache-streaming activations.
    """

    def __init__(self, model=None, *, device="cuda", context_s=4.0,
                 decision_s=0.160, batch_size=16,
                 model_name="nvidia/parakeet_realtime_eou_120m-v1"):
        import torch

        if model is None:
            from nemo.collections.asr.models import ASRModel
            model = ASRModel.from_pretrained(model_name=model_name)
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = torch.device(device)
        self.context_samples = round(context_s * 16000)
        self.step_samples = round(decision_s * 16000)
        if self.context_samples < self.step_samples or self.step_samples <= 0:
            raise ValueError("context must cover at least one positive decision step")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.model_name = model_name
        featurizer = getattr(self.model.preprocessor, "featurizer", None)
        if featurizer is not None and hasattr(featurizer, "dither"):
            featurizer.dither = 0.0

    def extract(self, audio: np.ndarray, sample_rate: int = 16000,
                *, decision_samples=None) -> FeatureSequence:
        """Extract complete grid decisions only; no final partial/flush decision.

        Optional decision_samples are exclusive 16-kHz endpoints for sparse
        training. All results are CPU arrays; datasets/models stay on the VM.
        """
        import torch

        x, filter_delay = causal_resample(audio, sample_rate)
        # Floor the source duration: never issue a decision before its full
        # source prefix exists, even when resampling rounded the length upward.
        max_end = len(audio) * 16000 // sample_rate
        ends = (np.arange(self.step_samples, max_end + 1, self.step_samples)
                if decision_samples is None else np.asarray(decision_samples))
        if ends.ndim != 1 or np.any(ends != ends.astype(np.int64)):
            raise ValueError("decision_samples must be a one-dimensional integer sequence")
        ends = ends.astype(np.int64)
        if np.any(ends <= 0) or np.any(ends > max_end) or np.any(np.diff(ends) <= 0):
            raise ValueError("decisions must be increasing, positive and within observed audio")
        output = []
        with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
            for start in range(0, len(ends), self.batch_size):
                group = ends[start:start + self.batch_size]
                windows = np.zeros((len(group), 2, self.context_samples), dtype=np.float32)
                for i, end in enumerate(group):
                    window = x[max(0, end - self.context_samples):end]
                    windows[i, :, -len(window):] = window.T
                signal = torch.from_numpy(windows.reshape(-1, self.context_samples)).to(self.device)
                lengths = torch.full((len(signal),), self.context_samples,
                                     device=self.device, dtype=torch.long)
                processed, processed_len = self.model.preprocessor(input_signal=signal, length=lengths)
                encoded, encoded_len = self.model.encoder(audio_signal=processed, length=processed_len)
                # NeMo Conformer encoder output layout is [B,D,T].
                if encoded.ndim != 3 or torch.any(encoded_len <= 0):
                    raise RuntimeError("Unexpected NeMo encoder output")
                rows = torch.arange(len(signal), device=self.device)
                last = encoded[rows, :, encoded_len.long() - 1]
                output.append(last.float().cpu().numpy().reshape(len(group), 2, -1))
        dim = int(getattr(self.model.encoder, "d_model", 0))
        features = np.concatenate(output) if output else np.empty((0, 2, dim), np.float32)
        return FeatureSequence(features, ends.astype(np.float64) / 16000, {
            "model": self.model_name, "method": "causal_prefix_window_last_frame",
            "context_s": self.context_samples / 16000, "decision_s": self.step_samples / 16000,
            "sample_rate": 16000, "source_sample_rate": sample_rate,
            "resampler_signal_delay_s": filter_delay, "resampler_delay_compensated": False,
            "timestamps": "exclusive audio availability; compute latency not included",
            "cache_streaming": False,
        })


class ParakeetStreamingEncoder(ParakeetPrefixEncoder):
    """NeMo cache-aware chunks with centered-STFT lookahead explicitly charged.

    Raw log-Mel preprocessing has normalization disabled; normalization, if
    requested by the model, occurs only within each currently available chunk.
    Offline feature calculation is an optimization of a local STFT, not access
    to future audio: incomplete right-edge STFT frames are discarded and every
    chunk is timestamped by its last required sample. Run test_causality against
    the installed NeMo/model before extracting real data.
    """

    def extract(self, audio: np.ndarray, sample_rate: int = 16000,
                *, decision_samples=None) -> FeatureSequence:
        return self.extract_batch([audio], sample_rate=sample_rate,
                                  decision_samples=[decision_samples])[0]

    def extract_batch(self, audios: list[np.ndarray], sample_rate: int = 16000,
                      *, decision_samples=None) -> list[FeatureSequence]:
        """Batch independent conversations, never successive chunks in time.

        Audios share a source sample rate and may have unequal durations. Bucket
        similar lengths for efficiency. Each result ends at its own observed
        duration; incomplete terminal chunks are never reported. This uses FP32
        model arithmetic; no mixed-precision approximation is enabled here.
        """
        import torch
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        if not audios:
            return []
        if decision_samples is None:
            decision_samples = [None] * len(audios)
        if len(decision_samples) != len(audios):
            raise ValueError("Need one decision_samples entry per conversation")
        waveforms, delays, max_ends, all_ends = [], [], [], []
        for audio, requested in zip(audios, decision_samples):
            x, delay = causal_resample(audio, sample_rate)
            max_end = len(audio) * 16000 // sample_rate
            ends = (np.arange(self.step_samples, max_end + 1, self.step_samples)
                    if requested is None else np.asarray(requested))
            if ends.ndim != 1 or np.any(ends != ends.astype(np.int64)):
                raise ValueError("decision_samples must be a one-dimensional integer sequence")
            ends = ends.astype(np.int64)
            if np.any(ends <= 0) or np.any(ends > max_end) or np.any(np.diff(ends) <= 0):
                raise ValueError("decisions must be increasing, positive and within observed audio")
            waveforms.append(x)
            delays.append(delay)
            max_ends.append(max_end)
            all_ends.append(ends)
        encoder = self.model.encoder
        buffer = CacheAwareStreamingAudioBuffer(self.model, online_normalization=True)
        buffer.preprocessor.eval()
        featurizer = buffer.preprocessor.featurizer
        if getattr(featurizer, "frame_splicing", 1) != 1:
            raise ValueError("Streaming availability calculation requires frame_splicing=1")
        if getattr(featurizer, "stft_pad_amount", None) is not None:
            raise ValueError("Streaming availability calculation requires centered STFT")
        if str(getattr(featurizer, "normalize", None)).lower() not in ("none", "false", "na"):
            raise ValueError("Raw preprocessing must not normalize over the full utterance")
        hop, n_fft, dim = int(featurizer.hop_length), int(featurizer.n_fft), int(encoder.d_model)
        chunk_ends, chunk_features = [], []
        cfg = encoder.streaming_cfg
        batch = 2 * len(audios)

        def at_step(value, step):
            return int(value[min(step, 1)]) if isinstance(value, (list, tuple)) else int(value)

        with torch.inference_mode(), torch.backends.cudnn.flags(allow_tf32=False):
            if max(max_ends) > n_fft:
                signals = np.zeros((batch, max(map(len, waveforms))), dtype=np.float32)
                for i, x in enumerate(waveforms):
                    signals[2*i:2*i+2, :len(x)] = x.T
                lengths = torch.tensor([len(x) for x in waveforms for _ in range(2)],
                                       device=self.device, dtype=torch.long)
                raw, raw_len = buffer.preprocessor(
                    input_signal=torch.from_numpy(signals).to(self.device), length=lengths)
                complete_lengths = [min(int(raw_len[2*i]), max(0, (end-1-n_fft//2)//hop+1))
                                    for i, end in enumerate(max_ends)]
                for i, complete in enumerate(complete_lengths):
                    for channel in range(2):
                        # One masked frame avoids empty-stream normalization.
                        buffer.append_processed_signal(raw[2*i+channel:2*i+channel+1, :, :max(1, complete)])
                cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=batch)
                iterator = iter(buffer)
                step = 0
                while buffer.buffer_idx < max(complete_lengths):
                    chunk_end = buffer.buffer_idx + at_step(cfg.chunk_size, step)
                    if chunk_end > max(complete_lengths):
                        break
                    chunk, chunk_len = next(iterator)
                    encoded, encoded_len, cache_channel, cache_time, cache_len = encoder.cache_aware_stream_step(
                        processed_signal=chunk, processed_signal_length=chunk_len,
                        cache_last_channel=cache_channel, cache_last_time=cache_time,
                        cache_last_channel_len=cache_len, keep_all_outputs=False,
                        drop_extra_pre_encoded=0 if step == 0 else cfg.drop_extra_pre_encoded,
                    )
                    active = torch.tensor([chunk_end <= n for n in complete_lengths for _ in range(2)],
                                          device=self.device)
                    if torch.any(active & (encoded_len <= 0)):
                        raise RuntimeError("Complete streaming chunk produced no encoder frames")
                    last = encoded[torch.arange(batch, device=self.device), :, encoded_len.long().clamp(min=1)-1]
                    chunk_features.append(last.float().cpu().numpy().reshape(len(audios), 2, dim))
                    chunk_ends.append((chunk_end-1)*hop + n_fft//2 + 1)
                    step += 1
        results = []
        stacked = np.stack(chunk_features) if chunk_features else None
        for i, ends in enumerate(all_ends):
            result = np.zeros((len(ends), 2, dim), dtype=np.float32)
            if chunk_ends:
                indices = np.searchsorted(chunk_ends, ends, side="right")-1
                valid = indices >= 0
                result[valid] = stacked[indices[valid], i]
            else:
                valid = np.zeros(len(ends), dtype=bool)
            results.append(FeatureSequence(result, ends.astype(np.float64)/16000, {
                "model": self.model_name, "method": "cache_aware_streaming_last_available",
                "decision_s": self.step_samples/16000, "sample_rate": 16000,
                "source_sample_rate": sample_rate, "cache_streaming": True,
                "resampler_signal_delay_s": delays[i], "resampler_delay_compensated": False,
                "stft_right_context_s": (n_fft//2+1)/16000,
                "chunk_available_at_s": [v/16000 for v in chunk_ends if v <= max_ends[i]],
                "valid_decision_mask": valid.tolist(), "conversation_batch_size": len(audios),
                "timestamps": "decision audio availability; held-last complete chunk; compute latency excluded",
            }))
        return results


def test_batch_causality(encoder: ParakeetStreamingEncoder, *, atol=1e-3):
    """Synthetic ragged batch equivalence and per-member future independence."""
    rng = np.random.default_rng(810)
    waves = [rng.normal(0, .03, (round(s*16000), 2)).astype(np.float32)
             for s in [1.28, 1.92, .64, .01]]
    individual = [encoder.extract(a) for a in waves]
    batched = encoder.extract_batch(waves)
    deltas = []
    for a, b in zip(individual, batched):
        np.testing.assert_allclose(a.features, b.features, atol=atol, rtol=atol)
        deltas.append(float(np.max(np.abs(a.features-b.features))) if a.features.size else 0.)
    mutated = [a.copy() for a in waves]
    mutated[1][10240:] = rng.normal(0, .3, mutated[1][10240:].shape)
    changed = encoder.extract_batch(mutated)
    for i, (a, b) in enumerate(zip(batched, changed)):
        keep = a.available_at_s <= .64 if i == 1 else np.ones(len(a.features), dtype=bool)
        np.testing.assert_allclose(a.features[keep], b.features[keep], atol=1e-5, rtol=1e-5)
    return {"passed": True, "max_batch_deltas": deltas, "conversations": len(waves)}


def test_causality(encoder: ParakeetPrefixEncoder, *, sample_rate=16000,
                   duration_s=1.28, cutoff_s=0.64, atol=1e-5):
    """Remote synthetic test: suffix mutation, truncation, batch invariance.

    Pass an actual loaded encoder for the meaningful NeMo integration test.
    Returns evidence; raises on any prefix difference. No dataset is accessed.
    """
    rng = np.random.default_rng(733)
    audio = rng.normal(0, 0.03, (round(duration_s * sample_rate), 2)).astype(np.float32)
    cutoff = round(cutoff_s * sample_rate)
    changed = audio.copy()
    changed[cutoff:] = rng.normal(0, 0.3, changed[cutoff:].shape)
    original = encoder.extract(audio, sample_rate)
    mutated = encoder.extract(changed, sample_rate)
    truncated = encoder.extract(audio[:cutoff], sample_rate)
    keep = original.available_at_s <= cutoff / sample_rate
    np.testing.assert_allclose(original.features[keep], mutated.features[keep], atol=atol, rtol=atol)
    np.testing.assert_allclose(original.features[keep], truncated.features, atol=atol, rtol=atol)
    old_batch = encoder.batch_size
    try:
        encoder.batch_size = 1
        individual = encoder.extract(audio[:cutoff], sample_rate)
    finally:
        encoder.batch_size = old_batch
    np.testing.assert_allclose(truncated.features, individual.features, atol=atol, rtol=atol)
    if not np.any(keep):
        raise AssertionError("Causality test must include prefix decisions")
    return {"passed": True, "prefix_decisions": int(keep.sum()),
            "max_suffix_delta": float(np.max(np.abs(original.features[keep] - mutated.features[keep]))),
            "sample_rate": sample_rate, "shape": list(original.features.shape)}


if __name__ == "__main__":
    import json
    encoder = ParakeetStreamingEncoder()
    print(json.dumps(test_causality(encoder)))
    print(json.dumps(test_causality(encoder, sample_rate=24000)))
    print(json.dumps(test_batch_causality(encoder)))
