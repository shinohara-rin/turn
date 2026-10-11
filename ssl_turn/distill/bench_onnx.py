"""Export StudentStep presets to ONNX and time one call per K on this CPU (onnxruntime).

    python bench_onnx.py [--ckpt d001/s3m.pt ...] --out /tmp/onnx
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import torch

import stream_student as ss
import student as st


def bench(path, K, threads, n=200):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    feeds = {i.name: np.zeros([d if isinstance(d, int) else 1 for d in i.shape],
                              np.int64 if 'int64' in i.type else np.float32) for i in sess.get_inputs()}
    feeds['audio'] = (np.random.randn(*feeds['audio'].shape) * 0.1).astype(np.float32)
    feeds['n_enc'][:] = 5
    feeds['n_head'][:] = 5
    for _ in range(10):
        sess.run(None, feeds)
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(None, feeds)
    dt = (time.perf_counter() - t0) / n
    return dt, dt / (K * 0.08)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/onnx')
    ap.add_argument('--presets', default='s1m,s3m,s8m')
    ap.add_argument('--threads', default='1,4')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    for p in a.presets.split(','):
        net = st.build(dict(preset=p)).eval()
        for K in (1, 2, 4):
            path = f'{a.out}/{p}_k{K}.onnx'
            ss.export(net, path, K)
            mb = os.path.getsize(path) / 2 ** 20
            for th in map(int, a.threads.split(',')):
                dt, rtf = bench(path, K, th)
                print(f'{p} K={K} threads={th}: {dt * 1000:.2f} ms/call, RTF {rtf:.3f}, {mb:.1f} MB fp32,'
                      f' {st.count_params(net) / 1e6:.2f}M params', flush=True)
