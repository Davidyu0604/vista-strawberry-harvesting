"""The observation posture must survive driving and return without scene changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
import tempfile
import unittest
import subprocess,json,hashlib
import numpy as np
from peduncle import PeduncleEnv
from mounted_basket import ready_point
from run import Recorder
from vista import Session


class StowedObservationTests(unittest.TestCase):
    def test_initial_search_and_return_keep_stow(self):
        e=PeduncleEnv(5,single=False,mobile_layout=True,render_quality='fast',initial_arm_pose='search_stow')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None,centered_grasp=True,visual_proposals=True)
                initial=e.last_action.copy()
                np.testing.assert_allclose(ready_point(e)-[0,e.base_pose()[1],0],[.19,-.04,.34],atol=1e-8)
                self.assertAlmostEqual(s.centered.observation_roll,initial[4])
                for _ in range(10):s.tick(e.last_action,'decision_settle')
                self.assertLess(np.linalg.norm(e.data.site_xpos[e.tcp]-ready_point(e)),.004)
                # A recorded new initial posture must replay exactly. Historical
                # episodes without the config field retain the legacy default.
                source_root=Path(__file__).resolve().parents[1]/'src'
                config={'seed':5,'quality':'fast','basket_mount':'chassis','mobile_layout':True,
                        'mobile_layout_level':'moderate','initial_arm_pose':'search_stow',
                        'source_sha256':{name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                                         for name in ('peduncle.py','env.py','scene.py','chassis.py')}}
                (s.rec.out/'run-config.json').write_text(json.dumps(config))
                code='''import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from replay_source import replay_environment
p=Path(sys.argv[2]);cfg=json.loads((p/'run-config.json').read_text())
e,source=replay_environment(p,cfg,{'single_fruit_fixture':False})
try:
 for line in (p/'trajectory.jsonl').read_text().splitlines():
  row=json.loads(line)
  assert np.max(abs(e.state()-row['state']))<1e-9
  assert np.max(abs(e.base_pose()-row['base_pose']))<1e-9
  e.drive(*row['base_command']);e.step(row['requested_action'])
finally:e.close()
'''
                subprocess.run([sys.executable,'-c',code,str(source_root),str(s.rec.out)],check=True,capture_output=True,text=True)
                s.local_targets.register([],e.base_pose(),'Mechanical fixture, no claimed fruit')
                result=s.search_row(1)
                self.assertAlmostEqual(result['actual_delta_y_m'],.18,delta=.005)
                self.assertFalse(any(r['phase'] in ('search_stow','search_observe_pose') for r in s.rec.rows))
                self.assertLess(np.max(abs(e.last_action-initial)),1e-8)
                # Execute a bounded arm adjustment, then verify return to the
                # original stowed position AND wrist angle rather than that adjustment.
                anchor={'tcp_world_m':e.data.site_xpos[e.tcp].copy(),'roll':float(e.last_action[4])}
                s.centered.adjust_observation(s.centered.observation_plan(anchor,0))
                self.assertLess(e.data.site_xpos[e.tcp][0],.17)
                restored=s.centered.restore_observation()
                self.assertTrue(restored['restored'])
                self.assertLess(np.linalg.norm(e.data.site_xpos[e.tcp]-ready_point(e)),.008)
                self.assertTrue(e.attached.all());self.assertFalse(any(w.number for w in e.data.warning))
        finally:e.close()

    def test_invalid_profile_rejected_before_scene_build(self):
        with self.assertRaisesRegex(ValueError,'initial arm pose'):
            PeduncleEnv(5,single=False,initial_arm_pose='unknown')
