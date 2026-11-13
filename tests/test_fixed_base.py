
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from peduncle import PeduncleEnv, build_peduncle
from run import Recorder
from vista import Session


class FixedBaseChecks(unittest.TestCase):
    def test_seed_changes_fruit_not_initial_robot_pose(self):
        colors=[];positions=[]
        for seed in (0,1,3,7):
            _,old=build_peduncle(seed,single=False)
            _,fixed=build_peduncle(seed,single=False,fixed_base=True)
            self.assertEqual(fixed['initial_base_pose'],[0,0,0])
            self.assertEqual(fixed['initial_tcp_world'],[.19,-.15,.33])
            self.assertEqual(fixed['harvest_base_y'],0)
            self.assertEqual(fixed['maturity'],old['maturity'])
            self.assertEqual(fixed['fruit_positions'],old['fruit_positions'])
            self.assertGreater(fixed['interactive_fruit_count'],8)
            colors.append(fixed['maturity'][:8]);positions.append(fixed['fruit_positions'][:8])
        self.assertGreater(len(set(c[0] for c in colors)),1)
        self.assertGreater(len(set(map(str,colors))),1)
        self.assertGreater(len(set(map(str,positions))),1)

    def test_all_harvest_motion_entrypoints_rejected(self):
        s=Session.__new__(Session);s.fixed_base=True;s.e=Mock();s.tick=Mock()
        for name in ('search_row','refine_parking'):
            with self.assertRaisesRegex(ValueError,'Fixed harvest base'):s._dispatch(name,{},'f',{},0)
        with self.assertRaisesRegex(ValueError,'Fixed harvest base'):s.search_row(1)
        with self.assertRaisesRegex(ValueError,'Fixed harvest base'):s.refine_parking()
        with self.assertRaisesRegex(ValueError,'Fixed harvest base'):s.apply_parking_trial()
        s.e.drive.assert_not_called();s.tick.assert_not_called()

    def test_no_parking_candidates_or_search_hint(self):
        s=Session.__new__(Session);s.fixed_base=True;s.e=Mock();s.locked_target=np.array([.3,0,.3])
        with patch('vista.approach_reachability',return_value={'reachable':False}):
            plan=s.plan_parking()
        self.assertFalse(plan['candidates']);self.assertIsNone(plan['recommended_delta_y_m'])
        self.assertNotIn('search_row',plan['next_action'])
        s.target_reachability=plan;s.target_confirmed=False;s.station_attempts=12;s.station_failures=0
        status=s.station_status();self.assertTrue(status['station_limit_reached']);self.assertEqual(status['next_action'],'finish')

    def test_real_fixed_pose_brake_and_model_contract(self):
        e=PeduncleEnv(seed=7,single=False,fixed_base=True,render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                client=SimpleNamespace()
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),client,fixed_base=True)
                self.assertNotIn('search_row',[t['function']['name'] for t in client.tools])
                self.assertNotIn('refine_parking',[t['function']['name'] for t in client.tools])
                self.assertNotIn('search_row',s.messages[0]['content'])
                self.assertIn('deliver_to_basket',s.messages[0]['content'])
                frame=e.frame
                result=s.execute([{'action':'drive','linear':.04,'angular':0,'seconds':1}])
                self.assertEqual(result[0]['status'],'rejected');self.assertEqual(e.frame,frame)
                for _ in range(30):e.step(e.last_action)
                np.testing.assert_allclose(e.base_pose(),[0,0,0],atol=.001)
                self.assertFalse(any(w.number for w in e.data.warning))
                with self.assertRaisesRegex(ValueError,'cannot mix'):
                    Session(e,s.rec,None,fixed_base=True,target_aware_parking=True)
        finally:e.close()

if __name__=='__main__':unittest.main()
