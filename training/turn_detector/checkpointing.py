"""Atomic epoch recovery for remote head training; no dataset or model policy."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import numpy as np

FORMAT_VERSION = 1


def _atomic_write(path, writer):
    """A final filename is visible only after its full contents are flushed."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(value, path):
    import torch
    _atomic_write(path, lambda file: torch.save(value, file))


def atomic_text(value, path):
    _atomic_write(path, lambda file: file.write(value.encode('utf-8')))


def capture_rng(generator=None):
    import torch
    return dict(torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy_global=np.random.get_state(),
                numpy_generator=copy.deepcopy(generator.bit_generator.state) if generator is not None else None)


def restore_rng(state, generator=None):
    import torch
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same CUDA device count as the saved run')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])
    np.random.set_state(state['numpy_global'])
    if state['numpy_generator'] is not None:
        if generator is None:
            raise ValueError('Checkpoint requires NumPy Generator state')
        generator.bit_generator.state = copy.deepcopy(state['numpy_generator'])


def _validate_config(saved, requested):
    # Paths may change after restoring persistent artifacts on a new VM. The
    # content hash, not path spelling, identifies the same frozen manifest.
    mutable = {'resume', 'out', 'manifest', 'turnbench_path', 'script_sha256',
               'torch_version', 'leakage_audit'}
    for key in sorted((set(saved) | set(requested)) - mutable):
        if saved.get(key) != requested.get(key):
            raise ValueError(f'Resume configuration changed: {key}: {saved.get(key)!r} -> {requested.get(key)!r}')


class EpochRecovery:
    """Self-contained epoch state plus repairable best/attempts side artifacts.

    An epoch file is first saved in `trained` phase before dev scoring, then
    replaced in `complete` phase after scoring. Both phases are resumable.
    Epoch/latest files hold optimizer and RNG state. `best.pt` remains a small
    inference-compatible artifact; its model is also embedded in every epoch
    recovery state, so a missed best.pt mirror can be reconstructed.
    """
    def __init__(self, out, config, resume=None):
        import torch
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=bool(resume))
        self.checkpoint = None
        self.attempts = []
        self.best = None
        self.best_inference = None
        self.config = config
        if resume:
            if resume == 'auto':
                paths = sorted(self.out.glob('epoch-[0-9]*.pt'),
                               key=lambda path: int(path.stem.split('-')[1]), reverse=True)
                candidates = paths[:1]
                if (self.out/'latest.pt').exists():
                    candidates.append(self.out/'latest.pt')
                if not candidates:
                    raise FileNotFoundError(f'No epoch checkpoint in {self.out}')
                # A mirror can copy latest.pt before the numbered file. Prefer
                # the actual newest state, including a completed scoring phase.
                checkpoint = None
                for path in candidates:
                    candidate = torch.load(path, map_location='cpu', weights_only=False)
                    def progress(value):
                        return (value['epoch'], value.get('training_state', {}).get('phase') == 'complete')
                    if checkpoint is None or progress(candidate) > progress(checkpoint):
                        checkpoint = candidate
            else:
                checkpoint = torch.load(Path(resume), map_location='cpu', weights_only=False)
            state = checkpoint.get('training_state')
            if state is None or state.get('version') != FORMAT_VERSION:
                raise ValueError('Legacy/inference checkpoint is evaluable but lacks exact-resume state; use epoch-NNN.pt or latest.pt from a resumable run')
            _validate_config(checkpoint['config'], config)
            if any(int(path.stem.split('-')[1]) > checkpoint['epoch'] for path in self.out.glob('epoch-[0-9]*.pt')):
                raise ValueError('Output contains newer epochs; resume latest or use a new output directory')
            self.checkpoint = checkpoint
            self.config = checkpoint['config']
            self.attempts = state['attempts']
            self.best = state['best']
            self.best_inference = state['best_inference']
            self._write_artifacts()
        atomic_text(json.dumps(self.config, indent=2), self.out/'config.json')

    @property
    def start_epoch(self):
        if self.checkpoint is None:
            return 1
        return self.checkpoint['epoch'] + (self.checkpoint['training_state']['phase'] == 'complete')

    def pending_evaluation(self, epoch):
        return (self.checkpoint is not None and self.checkpoint['epoch'] == epoch
                and self.checkpoint['training_state']['phase'] == 'trained')

    def restore(self, model, optimizer, generator=None):
        if self.checkpoint is None:
            return
        model.load_state_dict(self.checkpoint['model'])
        optimizer.load_state_dict(self.checkpoint['training_state']['optimizer'])
        # Restore last, after model construction/load has consumed RNG draws.
        restore_rng(self.checkpoint['training_state']['rng'], generator)

    def _save(self, checkpoint, optimizer, training_loss, phase, generator=None):
        checkpoint = {key: value for key, value in checkpoint.items() if key != 'training_state'}
        checkpoint['config'] = self.config
        checkpoint['training_state'] = dict(version=FORMAT_VERSION, phase=phase,
            optimizer=optimizer.state_dict(), rng=capture_rng(generator),
            attempts=copy.deepcopy(self.attempts), best=copy.deepcopy(self.best),
            best_inference=self.best_inference, training_loss=training_loss)
        atomic_torch_save(checkpoint, self.out/f"epoch-{checkpoint['epoch']:03}.pt")
        atomic_torch_save(checkpoint, self.out/'latest.pt')
        return checkpoint

    def save_trained(self, checkpoint, optimizer, training_loss, generator=None):
        return self._save(checkpoint, optimizer, training_loss, 'trained', generator)

    def complete(self, checkpoint, optimizer, training_loss, rows, selection_key, generator=None):
        if not rows or any(row['epoch'] != checkpoint['epoch'] for row in rows):
            raise ValueError('Expected nonempty dev attempts for exactly this epoch')
        candidate = copy.deepcopy(max(rows, key=selection_key))
        candidate['training_loss'] = training_loss
        # Replace this epoch transactionally; recovery never appends duplicates.
        self.attempts = [row for row in self.attempts if row['epoch'] != checkpoint['epoch']] + copy.deepcopy(rows)
        if self.best is None or selection_key(candidate) > selection_key(self.best):
            self.best = candidate
            self.best_inference = copy.deepcopy({key: value for key, value in checkpoint.items() if key != 'training_state'})
            self.best_inference['model'] = {key: value.detach().cpu().clone() for key, value in checkpoint['model'].items()}
            self.best_inference['operating_point'] = copy.deepcopy(self.best)
        # The checkpoint is the transaction record; side artifacts are rebuilt
        # from it after a crash at any point during the following writes.
        self._save(checkpoint, optimizer, training_loss, 'complete', generator)
        self._write_artifacts()
        return candidate

    def _write_artifacts(self):
        atomic_text(''.join(json.dumps(row)+'\n' for row in self.attempts), self.out/'attempts.jsonl')
        if self.best_inference is not None:
            atomic_torch_save(self.best_inference, self.out/'best.pt')
            atomic_text(json.dumps(self.best, indent=2), self.out/'best.json')
