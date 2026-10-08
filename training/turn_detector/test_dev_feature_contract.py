"""Synthetic public-dev reuse contracts; no models, real data or network."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from benchmark_infer import sha,digest,validate_feature_export
from dev_feature_contract import (validate_features,validate_parent,validate_derived,validate_reuse_sources,
                                  checked_npz,replay_check,MODE,ATOL,RTOL)
from test_score_benchmark_dev import identity


def parent_identity():
    parent=identity();f=parent['freeze'];f['protocol']['dev_feature_cache']=True
    f.update({'base_model':{'sha256':'base'},'continued_encoder':None,'dependencies':{},
              'sources':{'experiment/'+name:'hash' for name in
               ('heads.py','temporal_head.py','ensemble_heads.py','encoder.py','vad_batch.py','benchmark_infer.py')}})
    return parent


def derived_identity():
    parent=parent_identity();value=copy.deepcopy(parent);f=value['freeze']
    f.update({'execution_mode':MODE,'feature_manifest':'new-manifest',
        'derivation':{'parent_identity':parent,'parent_identity_digest':digest(parent),
        'feature_inventory':{cid:dict(features_sha256='f',probabilities_sha256='p',transaction_sha256='t') for cid in parent['ids']},
        'head_audit':{'passed':True,'manifest_sha256':'new-manifest'},
        'baseline_replay':{'passed':True,'events_exact':True,'checkpoint':parent['freeze']['checkpoint'],
                          'atol':ATOL,'rtol':RTOL,'conversations':{cid:{} for cid in parent['ids']}}}})
    f['protocol']['dev_feature_cache']=False
    return value


class FeatureTests(unittest.TestCase):
    def test_feature_export_is_explicit_dev_only(self):
        validate_feature_export('dev',True,True)
        validate_feature_export('test',False,False)
        for split,probabilities in [('test',True),('test',False),('dev',False)]:
            with self.assertRaises(ValueError):validate_feature_export(split,probabilities,True)

    def test_public_dev_only(self):
        validate_parent(parent_identity())
        for key,value in [('split','test'),('dev_feature_cache',False),('revision','other')]:
            parent=parent_identity();parent['freeze']['protocol'][key]=value
            with self.assertRaises(ValueError):validate_parent(parent)
        with self.assertRaises(ValueError):validate_parent(derived_identity())

    def test_feature_shapes_grid_and_history(self):
        times=np.array([.16,.32]);vad=np.zeros((2,2),np.float32)
        feature=dict(times=times,vad=vad,features=np.zeros((2,2,512),np.float16),extras=np.zeros((2,2,1),np.float32))
        prob=dict(times=times,vad=vad,gate_vad=vad,probabilities=vad)
        validate_features(feature,prob,.4)
        for key,value in [('features',np.zeros((2,2,512),np.float32)),('extras',np.zeros((2,2))),
                          ('times',np.array([.16,.31])),('vad',np.ones((2,2))),('features',np.full((2,2,512),np.nan,np.float16))]:
            bad={**feature,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):validate_features(bad,prob,.4)

    def test_changed_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'synthetic.npz';np.savez(path,x=np.zeros(2));old=sha(path)
            checked_npz(path,old)
            np.savez(path,x=np.ones(2))
            with self.assertRaises(ValueError):checked_npz(path,old)

    def test_replay_events_exact_probabilities_tolerant(self):
        a=np.full((2,2),.3,np.float32);events={'speaker_1':[.32]}
        replay_check(a,a+1e-7,events,events)
        with self.assertRaises(AssertionError):replay_check(a,a+.01,events,events)
        with self.assertRaises(ValueError):replay_check(a,a,events,{'speaker_1':[.16]})

    def test_derived_lineage_replay_sources_and_coverage(self):
        validate_derived(derived_identity())
        for change in ('encoder','source','replay','coverage','inventory','dependency','manifest'):
            bad=derived_identity();f=bad['freeze'];d=f['derivation']
            if change=='encoder':f['continued_encoder']={'checkpoint':'wrong'}
            if change=='source':f['sources']['experiment/encoder.py']='changed'
            if change=='replay':d['baseline_replay']['events_exact']=False
            if change=='coverage':bad['ids']=['1']
            if change=='inventory':d['feature_inventory'].pop('2')
            if change=='dependency':f['dependencies']={'changed':True}
            if change=='manifest':f['feature_manifest']='wrong'
            with self.subTest(change=change),self.assertRaises(ValueError):validate_derived(bad)


if __name__=='__main__':unittest.main()
