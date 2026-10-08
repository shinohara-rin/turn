"""Synthetic evidence contracts; no models, data or network."""
import copy
import unittest
from phase128_qualification import FORMAT,ENCODER_CHECKS,SILERO_CHECKS,qualification_identity,validate_qualification

def fixture():
    freeze={'base_model':{'repo':'pinned','revision':'revision','sha256':'base'},'continued_encoder':None,
            'silero_weights':'silero','sources':{'encoder.py':'encoder','helper.py':'helper'},
            'dependencies':{'python':'version','packages':{},'import_path':[]},
            'protocol':dict(batch_size=4,torch_threads=2,device='GPU',cuda='version',cudnn=1,tf32_matmul=False,tf32_cudnn=True)}
    report={'format':FORMAT,'synthetic_only':True,'passed':True,'identity':qualification_identity(freeze),
            'encoder':dict(passed=True,synthetic_only=True,phase_grid_contract='phase128-full-v1',checks=sorted(ENCODER_CHECKS),qualified_batch_size=4),
            'silero':dict(passed=True,synthetic_only=True,phase_grid_contract='phase128-full-v1',checks=sorted(SILERO_CHECKS))}
    return freeze,report

class Qualification(unittest.TestCase):
    def test_context_and_artifact_changes_rejected(self):
        freeze,report=fixture();validate_qualification(report,qualification_identity(freeze))
        paths=[('base_model','sha256'),('base_model','revision'),('sources','encoder.py'),('sources','helper.py'),('dependencies','python')]
        paths += [('protocol',key) for key in freeze['protocol']]
        for outer,inner in paths:
            changed=copy.deepcopy(freeze);changed[outer][inner]='different'
            with self.subTest(outer=outer,inner=inner),self.assertRaises(ValueError):validate_qualification(report,qualification_identity(changed))
        for key in ('silero_weights','continued_encoder'):
            changed=copy.deepcopy(freeze);changed[key]='different'
            with self.assertRaises(ValueError):validate_qualification(report,qualification_identity(changed))

    def test_missing_failed_nonsynthetic_or_incomplete_checks_rejected(self):
        freeze,report=fixture();identity=qualification_identity(freeze)
        for field in ('format','passed','synthetic_only','identity'):
            changed=copy.deepcopy(report);changed.pop(field)
            with self.assertRaises(ValueError):validate_qualification(changed,identity)
        for field in ('passed','synthetic_only'):
            changed=copy.deepcopy(report);changed[field]=False
            with self.assertRaises(ValueError):validate_qualification(changed,identity)
        for section in ('encoder','silero'):
            for field in ('passed','synthetic_only'):
                changed=copy.deepcopy(report);changed[section][field]=False
                with self.assertRaises(ValueError):validate_qualification(changed,identity)
            for field in ('passed','synthetic_only','phase_grid_contract','checks'):
                changed=copy.deepcopy(report);changed[section].pop(field)
                with self.assertRaises(ValueError):validate_qualification(changed,identity)
            for check in report[section]['checks']:
                changed=copy.deepcopy(report);changed[section]['checks'].remove(check)
                with self.assertRaises(ValueError):validate_qualification(changed,identity)
        for size in (None,1,2,16):
            changed=copy.deepcopy(report);changed['encoder']['qualified_batch_size']=size
            with self.assertRaises(ValueError):validate_qualification(changed,identity)

    def test_same_representation_can_qualify_multiple_heads(self):
        freeze,report=fixture();freeze.update(checkpoint='anotherhead',feature_manifest='anothermanifest')
        freeze['protocol']['operating_point']={'threshold':.5,'recommit_s':None}
        validate_qualification(report,qualification_identity(freeze))

if __name__=='__main__':unittest.main()
