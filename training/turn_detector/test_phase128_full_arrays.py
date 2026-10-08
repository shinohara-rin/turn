"""Remote synthetic NumPy/Torch tests only; no installed model or real data."""
import unittest
import numpy as np
from phase128_full_arrays import inputs_at_decisions,verify_replay_prefix,native_silero_all


class ArrayTests(unittest.TestCase):
    def test_full_tail_prefix_windows_and_suffix_invariance(self):
        n=2560*2+2048
        audio=np.arange(n*2,dtype=np.float32).reshape(n,2)/10000
        raw=np.arange((n//512)*2,dtype=np.float32).reshape(-1,2)/100
        ends=np.arange(2048,n+1,2560)
        vad,energy=inputs_at_decisions(audio,raw,ends)
        self.assertEqual(len(vad),3)
        np.testing.assert_array_equal(vad[0],raw[:4].mean(0))
        np.testing.assert_array_equal(vad[-1],raw[-5:].mean(0))
        changed=audio.copy();changed[4608:]=0
        mutated=raw.copy();mutated[9:]=0
        vv,ee=inputs_at_decisions(changed,mutated,ends)
        np.testing.assert_array_equal(vad[:2],vv[:2]);np.testing.assert_array_equal(energy[:2],ee[:2])
        with self.assertRaises(ValueError):inputs_at_decisions(audio,raw,ends[:-1])
        with self.assertRaises(ValueError):inputs_at_decisions(audio,raw[:-1],ends)

    def test_prefix_parity_rejects_drift(self):
        old=np.ones((2,2,512),np.float16)
        self.assertTrue(verify_replay_prefix(old,old.astype(np.float32))['passed'])
        with self.assertRaises(AssertionError):verify_replay_prefix(old,old.astype(np.float32)+.1)

    def test_native_tail_no_padding_and_reset(self):
        import torch
        class Fake:
            def eval(self):return self
            def reset_states(self):self.state=torch.zeros(2)
            def __call__(self,x,sr):
                if x.shape!=(2,512) or sr!=16000:raise AssertionError('Unexpected nativeframe')
                self.state=.5*self.state+x.mean(1)
                return self.state.sigmoid()
        fake=Fake();audio=np.ones((7168+31,2),np.float32)*.1
        raw=native_silero_all(audio,fake)
        self.assertEqual(raw.shape,(14,2))
        np.testing.assert_array_equal(raw,native_silero_all(audio[:-31],fake))
        changed=audio.copy();changed[4608:]=-.1
        np.testing.assert_array_equal(raw[:9],native_silero_all(changed,fake)[:9])


if __name__=='__main__':unittest.main()
