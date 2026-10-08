"""Prepare any host (Colab, Modal container, plain GPU server) for the pipeline.

Idempotent. Creates the historical /content layout, checks the HF token, clones the
pinned TurnBench evaluator and, with --install-deps, installs requirements.txt.

  python setup_remote.py --install-deps            # Colab / fresh server
  TD_WORKSPACE=/data/td python setup_remote.py ... # non-Colab: /content is symlinked to it (needs write access to /)
"""
import argparse, os, subprocess, sys
from pathlib import Path

TURNBENCH_REV = '38a6f874322430cb3ca71d8a52aa1e636e88bad8'
HERE = Path(__file__).resolve().parent


def run(args):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout[-10000:], flush=True)
    result.check_returncode()


def ensure_content_root():
    content = Path('/content')
    workspace = os.environ.get('TD_WORKSPACE')
    if workspace and not content.exists():
        Path(workspace).mkdir(parents=True, exist_ok=True)
        content.symlink_to(workspace, target_is_directory=True)
    content.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--install-deps', action='store_true', help='pip install requirements.txt')
    ap.add_argument('--turnbench-dir', type=Path, default=Path('/content/turnbench'))
    args = ap.parse_args()
    ensure_content_root()
    os.environ.setdefault('HF_HOME', '/content/hf')
    Path('/content/turn-recreation').mkdir(exist_ok=True)
    from workspace import hf_token, TOKEN_FILE
    hf_token()
    if not TOKEN_FILE.is_file() and os.environ.get('HF_TOKEN'):
        # Scripts not yet using workspace.hf_token() still read the file.
        TOKEN_FILE.write_text(os.environ['HF_TOKEN'].strip()); TOKEN_FILE.chmod(0o600)
    if not (args.turnbench_dir / '.git').exists():
        run(['git', 'clone', 'https://github.com/SesameAILabs/turnbench.git', str(args.turnbench_dir)])
    run(['git', '-C', str(args.turnbench_dir), 'checkout', TURNBENCH_REV])
    if args.install_deps:
        run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(HERE / 'requirements.txt')])
    print('SETUP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
