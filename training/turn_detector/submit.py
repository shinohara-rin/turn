"""Spawn a pipeline command on the deployed Modal app (see modal_app.py); returns immediately."""
import argparse, os
import modal

ap = argparse.ArgumentParser()
ap.add_argument('--cmd', required=True)
ap.add_argument('--gpu', action='store_true')
args = ap.parse_args()
app = os.environ.get('TD_MODAL_APP', 'turn-detector-training')
call = modal.Function.from_name(app, 'run_gpu' if args.gpu else 'run_cpu').spawn(args.cmd)
print('SPAWNED', call.object_id, flush=True)
