"""Remote-only storage screen; safetensors are weights, not executable ONNX."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--sha256', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--mirror', required=True, type=Path)
    a = p.parse_args()
    if platform.system() == 'Darwin' or not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Use a remote Colab with durable Drive mounted')
    if sha(a.checkpoint) != a.sha256:
        raise ValueError('Checkpoint hash mismatch')
    import torch
    from safetensors.torch import save_file, load_file
    import shutil
    torch.set_num_threads(2)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)['encoder']
    a.out.mkdir(parents=True, exist_ok=False)
    a.mirror.mkdir(parents=True, exist_ok=False)
    report = {'checkpoint_sha256': a.sha256, 'source_sha256': sha(__file__),
              'data_accessed': False, 'inference_tested': False, 'variants': {}}
    for name, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
        tensors = {k: (v.to(dtype) if v.is_floating_point() else v).contiguous().clone()
                   for k, v in state.items()}
        if any(not torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError('Nonfinite weights after conversion')
        path = a.out / f'encoder-{name}.safetensors'
        save_file(tensors, str(path))
        restored = load_file(str(path))
        if set(restored) != set(state) or any(not torch.equal(v, restored[k]) for k, v in tensors.items()):
            raise ValueError('Storage roundtrip failed')
        maximum = max(float((state[k].float() - v.float()).abs().max()) for k,v in restored.items() if v.numel())
        if name == 'fp32' and maximum != 0:
            raise ValueError('FP32 baseline altered')
        shutil.copyfile(path, a.mirror/path.name)
        digest = sha(path)
        if sha(a.mirror/path.name) != digest:
            raise ValueError('Drive readback mismatch')
        report['variants'][name] = {'bytes': path.stat().st_size, 'sha256': digest,
                                    'max_weight_delta': maximum, 'drive_verified': True}
    report['ratio_fp32_to_fp16'] = report['variants']['fp32']['bytes']/report['variants']['fp16']['bytes']
    report['limitations'] = 'Weights only; excludes graph, frontend, VAD, head and runtime. No inference, quality, memory or speed claim. FP16 rounding requires full recurrent and event qualification.'
    text = json.dumps(report, indent=2)
    (a.out/'packaging.json').write_text(text)
    shutil.copyfile(a.out/'packaging.json', a.mirror/'packaging.json')
    if sha(a.out/'packaging.json') != sha(a.mirror/'packaging.json'):
        raise ValueError('Report mirror failed')
    print(text)


if __name__ == '__main__':
    main()
