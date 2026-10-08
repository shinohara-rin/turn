"""Mandatory synthetic runtime evidence for full phase128 benchmark inference.

Generate in the same immutable source directory/runtime as inference. This module
never loads benchmark data. Evidence binds bytes and runtime, not model quality.
"""
from benchmark_infer import digest, require_equal

FORMAT = 'phase128-runtime-qualification-v1'
ENCODER_CHECKS = {'missing_tail', 'prefix_reuse', 'suffix_mutation', 'truncation',
                  'exact_grid', 'unequal_length_batch_parity'}
SILERO_CHECKS = {'legacy_prefix_parity', 'suffix_mutation', 'truncation', 'no_padding',
                 'native_tail', 'reset'}
NUMERICS = ('batch_size', 'torch_threads', 'device', 'cuda', 'cudnn', 'tf32_matmul', 'tf32_cudnn')


def qualification_identity(freeze):
    """Exclude head/operating-point so one qualified encoder supports paired heads."""
    return {'base_model': freeze['base_model'],
            'continued_encoder': freeze['continued_encoder'],
            'silero_weights': freeze['silero_weights'],
            'sources': freeze['sources'],
            'dependencies_sha256': digest(freeze['dependencies']),
            'numerics': {key: freeze['protocol'][key] for key in NUMERICS},
            'phase_grid_contract': 'phase128-full-v1'}


def validate_qualification(report, identity):
    if report.get('format') != FORMAT or report.get('synthetic_only') is not True or report.get('passed') is not True:
        raise ValueError('Missing successful synthetic phase128 runtime qualification')
    require_equal(report.get('identity'), identity, 'Phase qualification artifact/runtime identity')
    for name, required in (('encoder', ENCODER_CHECKS), ('silero', SILERO_CHECKS)):
        result = report.get(name, {})
        if result.get('passed') is not True or result.get('synthetic_only') is not True:
            raise ValueError('Failed or nonsynthetic phase qualification: '+name)
        if result.get('phase_grid_contract') != 'phase128-full-v1' or not required.issubset(result.get('checks', [])):
            raise ValueError('Incomplete phase qualification checks: '+name)
    if report['encoder'].get('qualified_batch_size') != identity['numerics']['batch_size']:
        raise ValueError('Phase qualification batch size mismatch')
    return report


def run_qualification(identity, encoder, detector):
    """Caller supplies already hash-verified pinned models; no model downloads."""
    import torch
    from phase128_full_arrays import qualify_phase_encoder, qualify_native_silero
    threads = identity['numerics']['torch_threads']
    require_equal(torch.get_num_threads(), threads, 'Qualification thread setting')
    report = {'format': FORMAT, 'synthetic_only': True, 'passed': True, 'identity': identity,
              'encoder': qualify_phase_encoder(encoder, batch_size=identity['numerics']['batch_size'])}
    report['silero'] = qualify_native_silero(detector)
    require_equal(torch.get_num_threads(), threads, 'Qualification thread preservation')
    return validate_qualification(report, identity)
