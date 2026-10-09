"""CPU-only look at the staged datasets on Modal (no GPU, no data leaves Modal)."""
import modal

app = modal.App('ssl-turn-inspect')
datasets = modal.Volume.from_name('turnbench-datasets')
image = modal.Image.debian_slim(python_version='3.12').pip_install('soundfile', 'pyarrow', 'numpy')


@app.function(image=image, volumes={'/datasets': datasets}, cpu=2, memory=4096, timeout=600)
def inspect():
    import json, os, glob
    import soundfile as sf
    import pyarrow.parquet as pq
    oto = '/datasets/otoearth/otoSpeech-full-duplex-turn-104h'
    top = sorted(os.listdir(oto))
    print('OTO top-level count', len(top), 'non-dirs', [x for x in top if not os.path.isdir(f'{oto}/{x}')][:20])
    first = next(x for x in top if os.path.isdir(f'{oto}/{x}') and not x.startswith('.'))
    readme = open(f'{oto}/README.md').read()
    print('README', len(readme)); print(readme[readme.find('---', 4):][:3500])
    print('OTO example dir', first, os.listdir(f'{oto}/{first}'))
    for f in sorted(os.listdir(f'{oto}/{first}')):
        p = f'{oto}/{first}/{f}'
        if f.endswith('.wav'):
            i = sf.info(p); print(' ', f, i.samplerate, i.channels, i.subtype, round(i.duration, 1), os.path.getsize(p))
        elif f.endswith(('.srt', '.json', '.txt', '.csv')):
            print(' ', f, repr(open(p).read()[:400]))
    for f in [x for x in top if not os.path.isdir(f'{oto}/{x}')][:5]:
        print('FILE', f, repr(open(f'{oto}/{f}', errors='replace').read()[:600]))
    for split in ('turn-benchmark-dev', 'turn-benchmark-test'):
        d = f'/datasets/mundo-ai/{split}'
        files = sorted(glob.glob(f'{d}/**/*', recursive=True))
        print(split, [(os.path.relpath(f, d), os.path.getsize(f)) for f in files if os.path.isfile(f)][:12])
        pqs = [f for f in files if f.endswith('.parquet')]
        if pqs:
            s = pq.read_schema(pqs[0]); print(' schema', s.names)
            if split.endswith('dev'):
                t = pq.read_table(pqs[0], columns=['conversation_id']); print(' ids', t.column(0).to_pylist()[:10])
    print(open('/datasets/progress.json').read()[:300])
