
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
import numpy as np
from vista import Session
from recording_quality import phase_labels


class ContactHoldChecks(unittest.TestCase):
    def session(self):
        s=Session.__new__(Session)
        s.e=SimpleNamespace(last_action=np.arange(6,dtype=float),dt=.05,
            data=SimpleNamespace(warning=[]),drive=Mock(),frame=0,
            meta={'harvest_mechanism':True,'tilted_stalk_prior':True},ik=Mock())
        s.tick=Mock();s.grasp_attempted=True
        return s

    def test_persistent_contact_holds_without_new_pose(self):
        s=self.session()
        with patch('vista.jaw_contact_state',return_value={'both':True}):r=s.check_contact_hold()
        self.assertTrue(r['stable']);self.assertEqual(r['observed_duration_s'],.5)
        self.assertEqual(s.tick.call_count,10)
        for call in s.tick.call_args_list:
            np.testing.assert_array_equal(call.args[0],s.e.last_action)
            self.assertEqual(call.args[1],'grasp_contact_check')
        self.assertEqual(phase_labels('grasp_contact_check'),('gripper',False,False))

    def test_intermittent_contact_stops_hold_and_blocks_pull(self):
        s=self.session()
        with patch('vista.jaw_contact_state',side_effect=[{'both':True},{'both':True},{'both':True},{'both':False}]):
            r=s.execute([{'action':'bend_pull','dx':-.02,'dy':0,'dz':0,'pitch_deg':-20,'seconds':1}])
        self.assertEqual(r[0]['status'],'rejected')
        self.assertFalse(r[0]['contact_stability']['stable'])
        s.e.ik.assert_not_called()
        self.assertEqual(s.tick.call_count,2)

    def test_no_contact_does_not_wait(self):
        s=self.session()
        with patch('vista.jaw_contact_state',return_value={'both':False}):r=s.check_contact_hold()
        self.assertFalse(r['stable']);s.tick.assert_not_called()

    def test_delivery_blocked_before_motion(self):
        s=self.session()
        with patch('vista.jaw_contact_state',side_effect=[{'both':True},{'both':True},{'both':False}]):
            with self.assertRaisesRegex(ValueError,'stable'):s.deliver_to_basket()
        s.e.ik.assert_not_called()
