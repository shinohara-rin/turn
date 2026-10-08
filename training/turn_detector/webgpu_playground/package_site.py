"""Package static source + pinned npm runtime; never package local model/data files."""
from pathlib import Path
import argparse,json,tarfile

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime-dir',type=Path,required=True,help='node_modules/onnxruntime-web (1.30.0)')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parent
    package=json.loads((args.runtime_dir/'package.json').read_text())
    if package['version']!='1.30.0':raise ValueError('Expected onnxruntime-web 1.30.0')
    files=[root/n for n in ('README.md','index.html','style.css','app.js','worker.js','capture.js','dsp.js','Notice.txt')]
    files+=sorted(p for p in (root/'licenses').rglob('*') if p.is_file())
    runtime=[args.runtime_dir/'dist'/'ort.webgpu.min.js']
    runtime+=sorted(p for p in (args.runtime_dir/'dist').glob('ort-wasm-*') if p.suffix in {'.wasm','.mjs'})
    with tarfile.open(args.out,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(root)))
        for p in runtime:tar.add(p,arcname='ort/'+p.name)
    print(f'Packed {len(files)+len(runtime)} allowlisted files into {args.out}')

if __name__=='__main__':main()
