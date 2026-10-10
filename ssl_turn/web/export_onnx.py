"""Export the streaming turn model to ONNX for onnxruntime-web.

    python ssl_turn/web/export_onnx.py --head runs/r019_asr_bgaug/bgaug_s1.pt --out web_models/

Writes encoder_k{K}.onnx for each --frames K (StreamStep: 16 kHz audio window -> K 1024-d
features per channel, with caches) and head.onnx (HeadStep, 3M params, K dynamic), fp32, plus
for each K encoder_k{K}_fp16.onnx (fp16 conformer, fp32
mel front end and I/O), _w16 (fp16 weights stored, fp32 math: half the download and runs on
any backend) and _int8 (dynamic int8 weight matmuls, for WASM).
The head is small enough to ship fp32 everywhere. Then checks every
variant against the PyTorch step on random audio with onnxruntime (CPU).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_encoder as se  # noqa: E402
import stream_head as sh  # noqa: E402

OPSET = 17


def export_encoder(model, path, K=1):
    step = se.StreamStep(model, K).eval()
    names = step.state_names()
    args = (torch.zeros(2, se.win(K)), torch.tensor([5]), *step.init_state(2))
    torch.onnx.export(step, args, path, opset_version=OPSET, do_constant_folding=True,
                      input_names=['audio', 't'] + names, output_names=['feat'] + [n + '_out' for n in names])
    return step


def export_head(net, path):
    """Frames per call is a dynamic axis: K = 1 for frame 0, then the encoder's K."""
    step = sh.HeadStep(net).eval()
    kc, vc = step.init_state()
    args = (torch.zeros(2, 2, 1024), torch.tensor([3]), kc, vc)
    outs = ['post', 'silent', 'fine', 'eot', 'int']
    torch.onnx.export(step, args, path, opset_version=OPSET, do_constant_folding=True,
                      input_names=['feat', 'n', 'kcache', 'vcache'], output_names=outs + ['kcache_out', 'vcache_out'],
                      dynamic_axes={'feat': {1: 'K'}, **{o: {0: 'K'} for o in outs}})
    return step


def fold_constants(path):
    """ORT basic-level optimisation (constant folding, standard ONNX ops only), in place. Folds
    each layer's linear_pos(pos_emb): a constant [141, 512] x [512, 512] MatMul that would
    otherwise run every step and cost more than the rest of the layer."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = path + '.opt'
    ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    os.replace(path + '.opt', path)


def to_fp16(src, dst):
    """fp16 everywhere except the top-level graph (mel front end, masks, cache bookkeeping):
    the power spectrum overflows fp16."""
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(src)
    keep = [n.name for n in m.graph.node if n.name.count('/') < 2]  # top-level nodes: no module scope
    m = float16.convert_float_to_float16(m, keep_io_types=True, node_block_list=keep,
                                         op_block_list=float16.DEFAULT_OP_BLOCK_LIST + ['Where'])
    for out in m.graph.output:  # outputs fed by converted nodes come out fp16: cast back to fp32
        if out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            for n in m.graph.node:
                n.output[:] = [o + '_h' if o == out.name else o for o in n.output]
                n.input[:] = [i + '_h' if i == out.name else i for i in n.input]
            m.graph.node.append(onnx.helper.make_node('Cast', [out.name + '_h'], [out.name],
                                                      to=onnx.TensorProto.FLOAT, name=out.name + '_cast'))
            out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    onnx.save(m, dst)


def to_w16(src, dst, min_size=4096):
    """fp16 weight storage, fp32 math: large fp32 initializers become fp16 + Cast. ORT folds the
    Casts at session load, so one half-size download runs on WASM and on WebGPU adapters without
    shader-f16."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(src)
    casts = []
    for init in m.graph.initializer:
        if init.data_type == onnx.TensorProto.FLOAT and np.prod(init.dims) >= min_size:
            a = numpy_helper.to_array(init).astype(np.float16)
            name = init.name
            init.CopyFrom(numpy_helper.from_array(a, name + '_w16'))
            casts.append(onnx.helper.make_node('Cast', [name + '_w16'], [name], to=onnx.TensorProto.FLOAT,
                                               name=name + '_w16_cast'))
    nodes = casts + list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.save(m, dst)


def to_int8(src, dst):
    """Dynamic int8 for the conformer's weight MatMuls; the mel front end stays fp32."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    m = onnx.load(src, load_external_data=False)
    weights = {i.name for i in m.graph.initializer}
    keep = [n.name for n in m.graph.node if n.name.count('/') < 2 or
            (n.op_type in ('MatMul', 'Gemm') and not any(i in weights for i in n.input))]
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'],
                     nodes_to_exclude=keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True)
    ap.add_argument('--nemo', default=None, help='local .nemo file (default: download from HF)')
    ap.add_argument('--out', default='web_models')
    ap.add_argument('--frames', default='1,2,4', help='encoder variants: frames per call')
    ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = se.load_encoder(a.nemo)
    net, _ = sh.load(a.head)
    head = sh.HeadStep(net).eval()
    if not a.check_only:
        with torch.no_grad():
            export_head(net, f'{a.out}/head.onnx')
        fold_constants(f'{a.out}/head.onnx')
    for K in map(int, a.frames.split(',')):
        base = f'{a.out}/encoder_k{K}'
        if not a.check_only:
            with torch.no_grad():
                export_encoder(model, base + '.onnx', K)
            fold_constants(base + '.onnx')
            to_fp16(base + '.onnx', base + '_fp16.onnx')
            to_w16(base + '.onnx', base + '_w16.onnx')
            to_int8(base + '.onnx', base + '_int8.onnx')
        check(se.StreamStep(model, K).eval(), head, base, a.out)


def check(enc, head, base, out, steps=40):
    """Run torch and each ONNX variant over the same random-noise chunks; print max errors."""
    import onnxruntime as ort
    K = enc.K
    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal((1280 * (steps + 4), 2))).astype(np.float32)
    names = enc.state_names()
    ref = []
    with torch.no_grad():
        st, hs = enc.init_state(2), head.init_state()
        for t, w in se.windows(wav, steps, K):
            feat, *st = enc(torch.from_numpy(w), torch.tensor([t]), *st)
            post, *_, kc, vc = head(feat, torch.tensor([min(t, head.W - 1)]), *hs)
            hs = (kc, vc)
            ref.append((feat.numpy(), post.numpy()))
    h = ort.InferenceSession(f'{out}/head.onnx', providers=['CPUExecutionProvider'])
    for sfx in ('', '_fp16', '_w16', '_int8'):
        e = ort.InferenceSession(f'{base}{sfx}.onnx', providers=['CPUExecutionProvider'])
        st = [x.numpy() for x in enc.init_state(2)]
        hs = [x.numpy() for x in head.init_state()]
        ef, hp = 0.0, 0.0
        for (t, w), (rf, rp) in zip(se.windows(wav, steps, K), ref):
            feat, *st = e.run(None, dict(audio=w, t=np.array([t], np.int64), **dict(zip(names, st))))
            post, *_, kc, vc = h.run(None, dict(feat=feat, n=np.array([min(t, head.W - 1)], np.int64),
                                                kcache=hs[0], vcache=hs[1]))
            hs = [kc, vc]
            ef = max(ef, np.linalg.norm(feat - rf) / np.linalg.norm(rf))
            hp = max(hp, np.abs(post - rp).max())
        print(f'K={K} {sfx or "fp32"}: encoder feature max rel err {ef:.2e}, head posterior max abs err {hp:.2e}',
              flush=True)


if __name__ == '__main__':
    main()
