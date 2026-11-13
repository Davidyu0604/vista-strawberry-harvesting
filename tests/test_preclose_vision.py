
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from preclose_vision import check_stalk_alignment
from vista import Session


class PrecloseVisionChecks(unittest.TestCase):
    def views(self, shifted=False, short=False, green=True):
        depth=np.full((240,320),.5)
        rgb=np.zeros((240,320,3),dtype=np.uint8)
        x=182 if shifted else 159
        rgb[117:122 if short else 141,x-1:x+2]=[30,130,20] if green else [180,20,20]
        view=(rgb,depth,{'position':np.zeros(3),'rotation':np.eye(3),'fovy':30.})
        return {'top':view,'wrist':view}

    def test_partial_visible_stalk_can_pass_without_whole_stalk_fit(self):
        result=check_stalk_alignment([0,0,-.5],[0,-1,0],self.views())
        self.assertTrue(result['passed'],result)
        self.assertLess(result['views']['top']['visible_span_m'],.025)

    def test_displacement_occlusion_and_non_green_evidence_rejected(self):
        for kwargs in ({'shifted':True},{'short':True},{'green':False}):
            self.assertFalse(check_stalk_alignment([0,0,-.5],[0,-1,0],self.views(**kwargs))['passed'])
        views=self.views();views.pop('wrist')
        self.assertFalse(check_stalk_alignment([0,0,-.5],[0,-1,0],views)['passed'])

    def test_empty_depth_and_invalid_axis_fail(self):
        views=self.views()
        for rgb,depth,cal in views.values():depth[:]=np.nan
        self.assertFalse(check_stalk_alignment([0,0,-.5],[0,-1,0],views)['passed'])
        with self.assertRaises(ValueError):check_stalk_alignment([0,0,-.5],[0,0,0],{})

    def session(self):
        s=Session.__new__(Session)
        e=SimpleNamespace(last_action=np.zeros(6),dt=.05,frame=0,tcp=0,
            data=SimpleNamespace(site_xpos=np.array([[.3,0,.3]]),warning=[]),drive=Mock(),
            meta={'harvest_mechanism':True,'tilted_stalk_prior':True})
        e.ik=Mock(return_value=(np.array([0,0,0,0,0,1.]),0.))
        s.e=e;s.calls=10;s.max_calls=32;s.grasp_attempted=False;s.target_stage='grasp'
        s.preclose_visual_check=True
        s.locked_target=np.array([.3,0,.3]);s.target_confirmed=True
        s.maturity_evidence={'passed':True,'target_m':[.3,0,.3],'axis_world':[0,0,1]}
        s.crops={'old':1};s.frames={'old':1};s.depth_frames={'old':1}
        def tick(q,phase):
            e.frame+=1;e.last_action=q.copy()
            if phase=='approach_retreat':e.data.site_xpos[0]=[.245,0,.3]
        s.tick=Mock(side_effect=tick)
        s.check_preclose_visual=Mock(return_value={'passed':False})
        return s

    def test_rejected_close_withdraws_open_clears_target_and_stops_batch(self):
        s=self.session()
        close={'action':'gripper','opening':-.15,'seconds':2.5}
        pull={'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}
        result=s.execute([close,pull])
        self.assertEqual(result[0]['status'],'rejected')
        self.assertEqual(result[1]['status'],'not_executed')
        self.assertFalse(s.grasp_attempted);self.assertIsNone(s.locked_target)
        self.assertIsNone(s.target_stage);self.assertFalse(s.frames or s.crops or s.depth_frames)
        self.assertEqual(s.preclose_recoveries,1)
        for call in s.tick.call_args_list:
            self.assertEqual(call.args[0][5],1);self.assertEqual(call.args[1],'approach_retreat')
        with self.assertRaisesRegex(ValueError,'neighbourhood'):
            s.lock_target(np.array([.301,0,.3]),{})

    def test_failed_retreat_latches_motion_stop(self):
        s=self.session();s.e.ik.return_value=(np.ones(6),.02)
        result=s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
        self.assertEqual(result[0]['status'],'rejected');s.tick.assert_not_called()
        self.assertTrue(s.preclose_recovery_failed)
        for name in ('play','approach_target','search_row','refine_parking','deliver_to_basket'):
            with self.assertRaisesRegex(ValueError,'recovery incomplete'):s._dispatch(name,{},'f',{},0)

    def test_recovery_budget_cannot_be_bypassed(self):
        s=self.session();s.preclose_recoveries=2
        for name in ('approach_target','search_row','refine_parking'):
            with self.assertRaisesRegex(ValueError,'exhausted'):s._dispatch(name,{},'f',{},0)
        result=s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
        self.assertEqual(result[0]['status'],'rejected');s.tick.assert_not_called()

    def test_fresh_check_allows_close_without_altering_command(self):
        s=self.session();s.check_preclose_visual.return_value={'passed':True}
        with patch('vista.robot_state',return_value={}),patch('vista.jaw_contact_state',return_value={'both':False}):
            result=s.execute([{'action':'gripper','opening':-.15,'seconds':.1}])
        self.assertEqual(result[0]['status'],'executed');self.assertTrue(s.grasp_attempted)
        s.check_preclose_visual.assert_called_once()
        self.assertEqual(s.e.last_action[5],-.15)


if __name__=='__main__':unittest.main()
