"""Download only selected tensors from a pinned Hugging Face safetensors shard.

Reads the shard header and the needed byte ranges over HTTP and writes a standalone
safetensors file, so an encoder can be taken from a large checkpoint without
downloading the decoder or LLM.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def fetch_tensors(repo, revision, shard, prefix, out):
    """Write tensors whose names start with `prefix` to `out`; returns an identity dict
    (source, tensor count, bytes, SHA256), also saved next to `out` as JSON."""
    import httpx
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo, shard, revision=revision)
    session = httpx.Client(headers=build_hf_headers(), follow_redirects=True, timeout=600)

    def get(start, end):
        r = session.get(url, headers={'Range': f'bytes={start}-{end}'})
        r.raise_for_status()
        if len(r.content) != end - start + 1:
            raise IOError('short range read')
        return r.content

    (header_len,) = struct.unpack('<Q', get(0, 7))
    header = json.loads(get(8, 8 + header_len - 1))
    base = 8 + header_len
    names = sorted(k for k in header if k.startswith(prefix))
    if not names:
        raise ValueError(f'No {prefix}* tensors in {shard}')
    spans = sorted((header[k]['data_offsets'][0], header[k]['data_offsets'][1], k) for k in names)
    # Selected tensors are usually contiguous; fetch as few large ranges as possible.
    new_header, offset, runs = {}, 0, []
    for start, end, name in spans:
        if runs and runs[-1][1] == start:
            runs[-1][1] = end
        else:
            runs.append([start, end])
        new_header[name] = dict(header[name], data_offsets=[offset, offset + end - start])
        offset += end - start
    new_header['__metadata__'] = {'format': 'pt', 'source': f'{repo}@{revision}/{shard}'}
    blob = json.dumps(new_header, separators=(',', ':')).encode()
    blob += b' ' * (-len(blob) % 8)
    tmp = out.with_suffix('.partial')
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<Q', len(blob)))
        f.write(blob)
        chunk = 256 << 20
        for start, end in runs:
            for lo in range(start, end, chunk):
                hi = min(end, lo + chunk)
                f.write(get(base + lo, base + hi - 1))
    tmp.rename(out)
    identity = dict(repo=repo, revision=revision, shard=shard, prefix=prefix, tensors=len(names),
                    bytes=offset, sha256=sha256(out))
    out.with_name(out.name + '.json').write_text(json.dumps(identity, indent=2))
    return identity
