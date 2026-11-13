
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import tempfile
import unittest
from pathlib import Path
import numpy as np
from mobile_scene import (sample_mobile_layout, search_limits, MODERATE_FORWARD_OPPORTUNITY_M, MODERATE_BACKWARD_OPPORTUNITY_M,
                          MODERATE_START_HALF_WIDTH_M, MODERATE_NEAR_FRUIT_INTERVAL_M)


class LayoutChecks(unittest.TestCase):
    def test_moderate_nearby_opportunities_without_fixed_count_or_ripe_id(self):
        counts=set();first_stages=set();modes=set()
        for seed in range(100):
            a=sample_mobile_layout(seed);b=sample_mobile_layout(seed)
            np.testing.assert_array_equal(a['positions'],b['positions'])
            self.assertEqual(a['maturity'],b['maturity'])
            points=np.asarray(a['positions']);along=abs(points[:,1]-a['initial_base_y'])
            ripe=np.array(a['maturity'])=='ripe'
            forward=points[:,1]-a['initial_base_y'];lo,hi=MODERATE_FORWARD_OPPORTUNITY_M
            self.assertTrue(((forward>=lo)&(forward<=hi)&ripe).any())
            lo,hi=MODERATE_BACKWARD_OPPORTUNITY_M
            self.assertTrue(((forward>=lo)&(forward<=hi)&ripe).any())
            self.assertEqual(a['backward_ripe_interval_m'],list(MODERATE_BACKWARD_OPPORTUNITY_M))
            self.assertNotEqual(a['start_region'],'gap')
            lo,hi=MODERATE_NEAR_FRUIT_INTERVAL_M
            self.assertTrue(((forward>=lo)&(forward<=hi)&~ripe).any())
            if a['start_region']=='unripe':self.assertFalse(((along<MODERATE_START_HALF_WIDTH_M)&ripe).any())
            distance=np.linalg.norm(points[:,None]-points[None,:],axis=2)
            np.fill_diagonal(distance,np.inf);self.assertGreater(distance.min(),.080)
            counts.add(len(points));first_stages.add(a['maturity'][0]);modes.add(a['start_region'])
        self.assertGreater(len(counts),10)
        self.assertEqual(first_stages,{'ripe','turning','green'})
        self.assertEqual(modes,{'unripe','mixed'})

    def test_invalid_profile_rejected(self):
        with self.assertRaisesRegex(ValueError,'level'):sample_mobile_layout(1,'easy')

    def test_repeatability_diversity_separation_and_start_regions(self):
        counts=set();modes=set();starts=[];first_stages=set()
        for seed in range(100):
            a=sample_mobile_layout(seed, "stress");b=sample_mobile_layout(seed, "stress")
            np.testing.assert_array_equal(a['positions'],b['positions'])
            self.assertEqual(a['maturity'],b['maturity'])
            points=np.asarray(a['positions']);distance=np.linalg.norm(points[:,None]-points[None,:],axis=2)
            np.fill_diagonal(distance,np.inf);self.assertGreater(distance.min(),.080)
            local=abs(points[:,1]-a['initial_base_y'])<.52
            if a['start_region']=='gap':self.assertFalse(local.any())
            if a['start_region']=='unripe':self.assertFalse(any(local[i] and m=='ripe' for i,m in enumerate(a['maturity'])))
            counts.add(len(points));modes.add(a['start_region']);starts.append(a['initial_base_y']);first_stages.add(a['maturity'][0])
        self.assertGreater(len(counts),10);self.assertEqual(modes,{'gap','unripe','mixed'})
        self.assertGreater(max(starts)-min(starts),1.4)
        self.assertEqual(first_stages,{'ripe','turning','green'})
        self.assertEqual(search_limits({}),(.10,.30))
        self.assertEqual(search_limits({'mobile_layout':True}),(.18,.72))

    def test_full_greenhouse_mobile_motion_and_envelope(self):
        from peduncle import PeduncleEnv
        from vista import Session
        # Mechanical fixture: empty queue supplied explicitly, not a visual success.
        e=PeduncleEnv(1,single=False,mobile_layout=True,mobile_layout_level='stress',render_quality='fast')
        class Recorder:
            def tick(self,q,phase):e.step(q)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                r=Recorder();r.out=Path(tmp)
                s=Session(e,r,None,centered_grasp=True,visual_proposals=True)
                self.assertGreater(e.meta['interactive_fruit_count'],100)
                self.assertEqual(e.meta['mobile_layout_audit']['start_region'],'gap')
                self.assertNotIn('start_region',s.messages[0]['content'])
                start=e.base_pose().copy()
                for _ in range(20):s.tick(e.last_action,'idle')
                for _ in range(4):
                    s.local_targets.register([],e.base_pose(),'Mechanical fixture only')
                    s.search_row(1)
                self.assertAlmostEqual(e.base_pose()[1]-start[1],.72,delta=.018)
                self.assertLess(abs(e.base_pose()[0]-start[0]),.005)
                self.assertFalse(s.local_targets.surveyed)
                with self.assertRaisesRegex(ValueError,'budget'):s.local_targets=None;s.search_row(1)
                self.assertTrue(e.attached.all());self.assertFalse(any(w.number for w in e.data.warning))
                np.testing.assert_array_equal(e.drive_command,[0,0])
        finally:e.close()


if __name__=='__main__':unittest.main()
