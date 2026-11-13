
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image
from visual_maturity import assess_maturity
from vista import Session


class VisualMaturityChecks(unittest.TestCase):
    def assess(self, color, depth=None):
        if isinstance(color, tuple): color=Image.new('RGB',(160,120),color)
        return assess_maturity([0,.033,-.5],[0,1,0],
            np.full((120,160),.5) if depth is None else depth,
            {'fovy':30,'rotation':np.eye(3),'position':np.zeros(3)},color)[0]

    def test_red_passes_orange_green_dark_fail(self):
        self.assertTrue(self.assess((200,12,20))['passed'])
        for rgb in ((220,95,30),(90,150,40),(10,0,0),(150,150,150)):
            self.assertFalse(self.assess(rgb)['passed'],rgb)

    def test_mixed_fruit_and_adjacent_red_do_not_pass(self):
        image=Image.new('RGB',(160,120),(200,12,20))
        # All nearby fruit-body pixels are green; red background cannot count.
        for x in range(65,95):
            for y in range(40,80):image.putpixel((x,y),(90,150,40))
        self.assertFalse(self.assess(image)['passed'])
        image=Image.new('RGB',(160,120),(200,12,20))
        for x in range(80,160):
            for y in range(120):image.putpixel((x,y),(90,150,40))
        self.assertFalse(self.assess(image)['passed'])

    def test_missing_or_sparse_depth_fails(self):
        depth=np.full((120,160),np.nan)
        self.assertFalse(self.assess((200,12,20),depth)['passed'])
        depth[59:61,79:81]=.5
        self.assertFalse(self.assess((200,12,20),depth)['passed'])

    def test_approach_cannot_bypass_gate(self):
        session=Session.__new__(Session)
        session.e=SimpleNamespace(meta={'harvest_mechanism':True})
        session.locked_target=np.array([.3,0,.3]);session.target_confirmed=True
        session.grasp_attempted=False
        for evidence in (None,{'passed':False},
                         {'passed':True,'target_m':[.31,0,.3]}):
            session.maturity_evidence=evidence
            with self.assertRaisesRegex(ValueError,'maturity'):
                session.dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',{},0)

    def test_pregrasp_rechecks_and_clear_invalidates(self):
        session=Session.__new__(Session)
        session.e=SimpleNamespace(meta={'harvest_mechanism':True})
        session.locked_target=np.array([.3,0,.3]);session.target_confirmed=True
        session.grasp_attempted=False;session.target_stage=None
        session.maturity_evidence={'passed':True,'target_m':[.3,0,.3],'axis_world':[0,0,1]}
        session.check_maturity=lambda *args:({'passed':False},Image.new('RGB',(16,16)))
        result,_=session.dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',{},0)
        self.assertEqual(result['status'],'rejected')
        self.assertFalse(session.maturity_evidence['passed'])
        session.dispatch('clear_target',{},'f',{},0)
        self.assertIsNone(session.locked_target)
        self.assertIsNone(session.maturity_evidence)
