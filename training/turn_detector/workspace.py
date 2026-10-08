"""Workspace/credential helpers shared by Colab, Modal and plain GPU machines.

The pipeline keeps the historical ``/content`` layout (``/content/turn-recreation``,
``/content/turnbench``, ``/content/hf``). Modal and generic hosts provide that
layout through ``setup_remote.py``/``modal_app.py``; the only thing scripts need
at run time is a way to find the Hugging Face token without assuming Colab.
"""
import os
from pathlib import Path

TOKEN_FILE = Path(os.environ.get('TD_HF_TOKEN_FILE', '/content/.hf_token'))


def hf_token(required=True):
    """HF_TOKEN env var first (Modal secret, CI, plain servers), then the Colab token file."""
    token = os.environ.get('HF_TOKEN', '').strip()
    if token:
        return token
    if TOKEN_FILE.is_file():
        return TOKEN_FILE.read_text().strip()
    if required:
        raise RuntimeError(f'No Hugging Face token: set HF_TOKEN or create {TOKEN_FILE} (mode 0600)')
    return None
