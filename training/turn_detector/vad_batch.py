"""Batched equivalent of cache_features.causal_vad; no decision-policy change.

Independent stereo conversations occupy fixed rows in Silero's recurrent batch.
Rows are never removed or reassigned mid-call. Padding occurs only after a row's
last complete 160ms decision, and its later outputs are discarded. Each call
resets all recurrent state. Input must already be causal-resampled 16kHz audio.
"""
import numpy as np


def causal_vad_batch(audios, *, detector=None, reduction="mean"):
    """Return one float32 [floor(N/2560),2] array per input [N,2] array.

    Identical protocol to the existing helper: five consecutive causal 512-sample
    (32ms) Silero probabilities are averaged for each 160ms decision. No threshold,
    audio normalization, resampling, lookahead, or model-weight changes are added.
    reduction="raw" returns all five 32ms scores per complete decision [5*T,2].
    reduction="last" instead takes the fifth (latest available) score per decision;
    this is an explicit serving-gate ablation, not the original mean protocol.
    A supplied detector is reset and used exclusively until this function returns.
    """
    if reduction not in ("mean", "last", "raw"):
        raise ValueError("reduction must be mean, last or raw")
    import torch
    if not audios:
        return []
    arrays = []
    for audio in audios:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError('Each conversation must be [samples,2] at 16000 Hz')
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite audio')
        arrays.append(x)
    decisions = [len(x)//2560 for x in arrays]
    longest = max(decisions)
    if not longest:
        return [np.empty((0, 2), dtype=np.float32) for _ in arrays]
    if detector is None:
        from silero_vad import load_silero_vad
        detector = load_silero_vad()
    detector.eval(); detector.reset_states()
    # CPU matches the original Silero gate. Keep the same fixed thread budget.
    torch.set_num_threads(2)
    batch = len(arrays)
    raw = np.empty((longest*5, batch, 2), dtype=np.float32)
    block = np.zeros((batch, 2, 512), dtype=np.float32)
    with torch.inference_mode():
        for frame in range(longest*5):
            start = frame*512
            block.fill(0.)
            for i, (audio, count) in enumerate(zip(arrays, decisions)):
                if frame < count*5:
                    block[i] = audio[start:start+512].T
            probability = detector(torch.from_numpy(block.reshape(2*batch, 512)), 16000)
            raw[frame] = probability.cpu().numpy().reshape(batch, 2)
    if reduction == "raw":
        return [raw[:count*5, i].copy() for i, count in enumerate(decisions)]
    windows = [raw[:count*5, i].reshape(count, 5, 2) for i, count in enumerate(decisions)]
    return [window.mean(axis=1) if reduction == "mean" else window[:, -1, :].copy()
            for window in windows]


def synthetic_vad_checks(*, atol=1e-6, rtol=1e-5):
    """Actual Silero parity, suffix, truncation, empty, and state-reset tests.

    Raises instead of relaxing tolerance. Remote only in the supplied smoke CLI.
    Synthetic waveforms check implementation semantics, not VAD accuracy.
    """
    from cache_features import causal_vad
    from silero_vad import load_silero_vad
    rng = np.random.default_rng(20261007)
    durations = [2.56, 1.31, .64, .01, 0.]
    audios = []
    for duration in durations:
        t = np.arange(round(duration*16000))/16000
        signal = .06*np.sin(2*np.pi*173*t)*((t%0.4)<.24)
        noise = rng.normal(0, .01, (len(t), 2))
        audios.append((noise+signal[:, None]).astype(np.float32))
    independent = [causal_vad(audio) for audio in audios]
    detector = load_silero_vad()
    batched = causal_vad_batch(audios, detector=detector)
    differences = []
    for original, candidate in zip(independent, batched):
        np.testing.assert_allclose(candidate, original, atol=atol, rtol=rtol)
        differences.append(float(np.max(np.abs(candidate-original))) if candidate.size else 0.)
    repeated = causal_vad_batch(audios, detector=detector)
    for original, candidate in zip(batched, repeated):
        np.testing.assert_array_equal(original, candidate)
    cutoff = 10240  # exactly 0.64s, four complete decision steps
    changed = [a.copy() for a in audios]
    changed[0][cutoff:] = rng.normal(0, .2, changed[0][cutoff:].shape)
    mutated = causal_vad_batch(changed, detector=detector)
    for i, (original, candidate) in enumerate(zip(batched, mutated)):
        keep = min(len(original), cutoff//2560) if i == 0 else len(original)
        np.testing.assert_array_equal(original[:keep], candidate[:keep])
    shortened = [audios[0][:cutoff]]+audios[1:]
    truncated = causal_vad_batch(shortened, detector=detector)
    for original, candidate in zip(batched, truncated):
        np.testing.assert_array_equal(original[:len(candidate)], candidate)
    # Empty peers must not alter nonempty outputs beyond documented numeric tolerance.
    alone = causal_vad_batch([audios[0]], detector=detector)[0]
    np.testing.assert_allclose(alone, batched[0], atol=atol, rtol=rtol)
    last_checks = synthetic_reduction_checks(load_silero_vad, atol=atol, rtol=rtol)
    return {'passed': True, 'reduction_checks': last_checks, 'sample_rate': 16000, 'durations_s': durations,
            'max_individual_batch_deltas': differences, 'atol': atol, 'rtol': rtol,
            'suffix_mutation': 'bitwise_equal', 'truncation': 'bitwise_equal',
            'state_reset': 'bitwise_equal', 'synthetic_only': True}


def synthetic_reduction_checks(detector_factory, *, atol=1e-6, rtol=1e-5):
    """Reference parity and prefix checks for both reductions; factory may be fake.

    Actual Silero coverage is invoked only by the remote synthetic_vad_checks.
    No real speech or labels are loaded here.
    """
    import torch
    rng = np.random.default_rng(206)
    audios = [rng.normal(0, .1, (n, 2)).astype(np.float32)
              for n in (12800, 7921, 2560, 256, 0)]
    cutoff = 5120
    results = {}
    for reduction in ("mean", "last", "raw"):
        reference = []
        for audio in audios:
            detector = detector_factory().eval()
            detector.reset_states()
            n = len(audio)//2560
            values = []
            with torch.inference_mode():
                for start in range(0, n*2560, 512):
                    values.append(detector(torch.from_numpy(audio[start:start+512].T.copy()),16000).cpu().numpy().reshape(2))
            windows = np.asarray(values,dtype=np.float32).reshape(n,5,2)
            reference.append(windows.mean(1) if reduction == "mean" else windows.reshape(-1,2) if reduction == "raw" else windows[:,-1,:])
        detector = detector_factory()
        original = causal_vad_batch(audios,detector=detector,reduction=reduction)
        deltas = []
        for a,b in zip(reference,original):
            np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
            deltas.append(float(np.max(abs(a-b))) if a.size else 0.)
        repeated = causal_vad_batch(audios,detector=detector,reduction=reduction)
        altered = [audio.copy() for audio in audios]
        altered[0][cutoff:] = rng.normal(0,.5,altered[0][cutoff:].shape)
        suffix = causal_vad_batch(altered,detector=detector,reduction=reduction)
        prefix = causal_vad_batch([audios[0][:cutoff]]+audios[1:],detector=detector,reduction=reduction)
        for i,a in enumerate(original):
            keep = min(len(a),cutoff//(512 if reduction == "raw" else 2560)) if i == 0 else len(a)
            np.testing.assert_array_equal(a,repeated[i])
            np.testing.assert_array_equal(a[:keep],suffix[i][:keep])
            np.testing.assert_array_equal(a[:len(prefix[i])],prefix[i])
        results[reduction] = dict(max_batch_delta=max(deltas),prefix_invariant=True,
                                  independent_speakers=True,state_reset=True)
    return dict(passed=True,synthetic_only=True,atol=atol,rtol=rtol,reductions=results)
