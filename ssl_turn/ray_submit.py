#!/usr/bin/env python3
"""Submit a ray_run.py command to a Ray cluster through the dashboard's REST API.

Zips ssl_turn/ (plus any --add files) as the job's working_dir, uploads it with the same
package API the Ray SDK uses, and runs `python ray_run.py <args>` with the chosen pip env.

    python ray_submit.py --env gpu --gpus 1 -- encode_mtd --train 131
    python ray_submit.py --env turnbench --add /path/split.json:split.json -- prep
    python ray_submit.py --logs JOB_ID          # print a job's status and log tail

Needs RAY_AUTH_TOKEN; the cluster URL comes from --url or RAY_DASHBOARD_URL.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TURNBENCH = 'git+https://github.com/SesameAILabs/turnbench@38a6f874322430cb3ca71d8a52aa1e636e88bad8'
ENVS = {  # same packages as the Modal images in pipeline/common.py (torch unpinned for the node's CUDA)
    'turnbench': [TURNBENCH, 'srt', 'scipy', 'soundfile'],
    # No runtime-env pip for the GPU env: installing torch there can outlast Ray's 15 min
    # job-start timeout, so ray_run.py installs it into a persistent --target dir instead.
    'gpu': [],
}


def call(url, path, data=None, method=None, raw=False):
    body = data if isinstance(data, bytes) else (json.dumps(data).encode() if data is not None else None)
    req = urllib.request.Request(url + path, data=body, method=method or ('POST' if body is not None else 'GET'),
                                 headers={'Authorization': 'Bearer ' + os.environ['RAY_AUTH_TOKEN'],
                                          'Content-Type': 'application/octet-stream' if raw else 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = r.read()
    return out if raw else json.loads(out or b'null')


def package(extra):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(HERE):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for f in sorted(files):
                p = os.path.join(root, f)
                info = zipfile.ZipInfo(os.path.relpath(p, HERE), (1980, 1, 1, 0, 0, 0))  # stable hash
                z.writestr(info, open(p, 'rb').read())
        for src, arc in extra:
            z.writestr(zipfile.ZipInfo(arc, (1980, 1, 1, 0, 0, 0)), open(src, 'rb').read())
    data = buf.getvalue()
    return f'_ray_pkg_{hashlib.sha1(data).hexdigest()}.zip', data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=os.environ.get('RAY_DASHBOARD_URL',
                                                    'https://magic-defining-selection-advanced.trycloudflare.com'))
    ap.add_argument('--env', choices=sorted(ENVS), default='gpu')
    ap.add_argument('--gpus', type=float, default=0)
    ap.add_argument('--add', nargs='*', default=[], help='SRC:ARCNAME files to include in the working dir')
    ap.add_argument('--forward-env', nargs='*', default=[], help='env var names to pass to the job')
    ap.add_argument('--logs', help='print status and log tail of this job id, then exit')
    ap.add_argument('--tail', type=int, default=60)
    ap.add_argument('--stop', help='stop this job id, then exit')
    ap.add_argument('--wait', action='store_true')
    ap.add_argument('--python', default='python', help="interpreter on the node, e.g. a venv that already has torch")
    ap.add_argument('--setup', default='', help='shell command run on the node before the entrypoint')
    ap.add_argument('args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    url = a.url.rstrip('/')
    if a.stop:
        print(call(url, f'/api/jobs/{a.stop}/stop', {}))
        return
    if a.logs:
        st = call(url, f'/api/jobs/{a.logs}')
        logs = call(url, f'/api/jobs/{a.logs}/logs')['logs']
        print('\n'.join(logs.splitlines()[-a.tail:]))
        print('status', st['status'], st.get('message', '')[:300])
        return
    name, data = package([tuple(x.split(':', 1)) for x in a.add])
    call(url, f'/api/packages/gcs/{name}', data, method='PUT', raw=True)
    args = [x for x in a.args if x != '--']
    entry = f'{a.python} ray_run.py ' + ' '.join(args)
    job = {'entrypoint': f'{a.setup} && {entry}' if a.setup else entry, 'entrypoint_num_gpus': a.gpus,
           'runtime_env': {'working_dir': f'gcs://{name}', 'env_vars': {k: os.environ[k] for k in a.forward_env}}}
    if ENVS[a.env]:
        job['runtime_env']['pip'] = ENVS[a.env]
    jid = call(url, '/api/jobs/', job)['job_id']
    print('job', jid, flush=True)
    if a.wait:
        while (s := call(url, f'/api/jobs/{jid}')['status']) not in ('SUCCEEDED', 'FAILED', 'STOPPED'):
            time.sleep(10)
        print(call(url, f'/api/jobs/{jid}/logs')['logs'][-20000:])
        print('status', s)
        sys.exit(s != 'SUCCEEDED')


if __name__ == '__main__':
    main()
