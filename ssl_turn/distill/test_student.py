import numpy as np
import torch

import student as st


def test_shapes_and_causality():
    torch.manual_seed(0)
    cfg = dict(preset='s1m', window_s=0.8, head_window_s=0.8)
    net = st.build(cfg).eval()
    T = 40
    wav = torch.randn(1, T * st.FRAME, 2) * 0.1
    with torch.no_grad():
        a = net(wav, with_feat=True)
        cut = 25
        wav2 = wav.clone()
        wav2[:, cut * st.FRAME:] = torch.randn_like(wav2[:, cut * st.FRAME:])
        b = net(wav2)
    assert a['floor'].shape == (1, T, 4) and a['feat'].shape == (1, T, 2, 1024) and a['yld'].shape == (1, T, 2)
    # frame t uses audio before (t + 1) * 80 ms only
    assert torch.allclose(a['floor'][:, :cut], b['floor'][:, :cut], atol=1e-5)
    assert not torch.allclose(a['floor'][:, cut], b['floor'][:, cut], atol=1e-5)


def test_chunked_equals_full():
    """Exact chunked inference: a window that starts context_frames before the chunk reproduces
    the full-sequence outputs."""
    torch.manual_seed(0)
    cfg = dict(preset='s1m', window_s=0.4, head_window_s=0.4, kernel=3, layers=2, head_layers=1)
    net = st.build(cfg).eval()
    ctx = st.context_frames(cfg)
    T = ctx + 30
    wav = torch.randn(1, T * st.FRAME, 2) * 0.1
    with torch.no_grad():
        full = net(wav)['floor']
        lo = 20
        part = net(wav[:, lo * st.FRAME:])['floor']
    assert torch.allclose(full[:, lo + ctx:], part[:, ctx:], atol=1e-4), (full[:, lo + ctx:] - part[:, ctx:]).abs().max()


def test_mel_matches_stft():
    wav = torch.randn(1, 16000)
    fe = st.MelFrontend()
    ours = fe(wav)
    # torch.stft centers the 400-sample window in the 512-sample frame: shift by the 56-sample offset
    x = torch.nn.functional.pad(wav, (st.WIN - st.HOP + (st.N_FFT - st.WIN) // 2, st.N_FFT))
    spec = torch.stft(x, st.N_FFT, st.HOP, st.WIN, torch.hann_window(st.WIN, periodic=True), center=False,
                      return_complex=True).abs() ** 2
    ref = torch.log(spec[0].T @ fe.fb + 1e-6)
    n = min(len(ref), ours.shape[1])
    assert torch.allclose(ours[0, :n], ref[:n], atol=1e-3)


def test_stream_step_matches_offline():
    import stream_student as ss
    torch.manual_seed(0)
    cfg = dict(preset='s1m', window_s=0.8, head_window_s=1.2, layers=2, head_layers=1)
    net = st.build(cfg).eval()
    T = 40
    wav = torch.randn(T * st.FRAME, 2) * 0.1
    with torch.no_grad():
        ref = net(wav[None])
    for K in (1, 2, 4):
        post, silent, fine, yld, eot, intr = ss.run_stream(ss.StudentStep(net, K).eval(), wav)
        assert torch.allclose(post[:, 0], ref['floor'][0].softmax(-1), atol=1e-4), K
        assert torch.allclose(yld, ref['yld'][0].sigmoid(), atol=1e-4), K
        assert torch.allclose(silent, ref['act'][0].softmax(-1)[..., 0], atol=1e-4), K
