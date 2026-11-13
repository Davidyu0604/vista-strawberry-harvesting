
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from stalk_proposals import propose_stalk_points
from vista import Session


class ProposalGeometryTests(unittest.TestCase):
    def sample(self):
        depth=np.full((24,24),.2)
        rgb=np.zeros((24,24,3),dtype=np.uint8);rgb[:,:,1]=150
        calibration={'fovy':30.,'rotation':np.eye(3),'position':np.array([0,0,.45])}
        return rgb,depth,calibration,[0,0,.22]

    def test_bounded_fits_merge_same_axis(self):
        fitter=Mock(return_value=(np.array([0,0,.25]),{
            'axis_world':[0,0,1],'visible_length_m':.03,'samples':40}))
        result=propose_stalk_points(*self.sample(),fitter)
        self.assertEqual(fitter.call_count,32)
        self.assertEqual(len(result['proposals']),1)
        self.assertEqual(result['fit_passes'],32)
        self.assertFalse(result['coverage_certified'])

    def test_non_green_or_unmeasured_surface_never_fitted(self):
        rgb,depth,cal,body=self.sample();rgb[:]=[200,20,20]
        fitter=Mock()
        for invalid in (depth,np.full_like(depth,np.nan)):
            result=propose_stalk_points(rgb,invalid,cal,body,fitter)
            self.assertEqual(result['proposals'],[])
            self.assertFalse(result['coverage_certified'])
        fitter.assert_not_called()

    def test_fitter_failure_or_target_outside_active_body_not_proposed(self):
        for fitter in (Mock(side_effect=ValueError('ambiguous')),
                       Mock(return_value=(np.array([1,0,.25]),{}))):
            result=propose_stalk_points(*self.sample(),fitter)
            self.assertEqual(result['proposals'],[])
            self.assertFalse(result['coverage_certified'])


class NumberedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.state={'joints_rad':[0]*6,'tcp_world_m':[0,0,.3],'base_x_y_yaw':[0,0,0]}
        self.env=SimpleNamespace(meta={'tilted_stalk_prior':True,'harvest_mechanism':True,
                                      'basket_mount':'chassis','policy_cameras':['harvest_close','wrist']})
        self.session=Session(self.env,SimpleNamespace(out=Path(self.tmp.name)),None,visual_proposals=True)
        self.session.local_targets.register([{'surface_m':[.25,0,.22],
            'maturity':'ripe','visibility':'clear'}],[0,0,0],'visual survey')
        self.key=self.session.local_targets.active['candidate_key']
        self.session.stalk_proposals={'W1':{'proposal_id':'W1','camera':'wrist','frame_id':'f',
            'candidate_key':self.key,'pixel':[300,190],'state':self.state,
            'target_m':[999,999,999]}}  # Stored fit must never become a command.
        self.args={'frame_id':'f','candidate_key':self.key,'proposal_id':'W1',
                   'confirmed_short_stalk':True,'reason':'Visible connection to active ripe fruit'}

    def test_number_routes_saved_pixel_through_normal_localization_once(self):
        session=self.session;original=session._dispatch
        pixel=Mock(return_value=({'target_confirmed':True,
                                 'reachability':{'current':{'reachable':True}}},[]))
        def route(name,*args):
            return pixel(name,*args) if name=='locate_pixel' else original(name,*args)
        with patch.object(session,'_dispatch',side_effect=route),patch('vista.robot_state',return_value=self.state):
            result,_=session.dispatch('locate_stalk_proposal',self.args,'f',self.state,0)
        self.assertEqual(pixel.call_count,1)
        self.assertEqual(pixel.call_args.args[1],{'frame_id':'f','camera':'wrist','x':300,'y':190})
        self.assertEqual(session.station_attempts,1)
        self.assertEqual(session.local_targets.active['failures'],0)
        self.assertEqual(result['stalk_proposal_selection']['proposal_id'],'W1')

    def test_invalid_or_unconfirmed_number_spends_call_not_candidate_retry(self):
        for changes in ({'frame_id':'old'},{'candidate_key':'other'},{'proposal_id':'W99'},
                        {'confirmed_short_stalk':False},{'target_m':[.25,0,.25]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.session.dispatch('locate_stalk_proposal',{**self.args,**changes},'f',self.state,0)
        self.assertEqual(self.session.station_attempts,5)
        self.assertEqual(self.session.local_targets.active['failures'],0)
        self.assertIsNone(self.session.locked_target)

    def test_stored_old_frame_or_camera_drift_rejected(self):
        item=self.session.stalk_proposals['W1'];item['frame_id']='old'
        with self.assertRaisesRegex(ValueError,'stale'):
            self.session.dispatch('locate_stalk_proposal',self.args,'f',self.state,0)
        item['frame_id']='f'
        moved={**self.state,'tcp_world_m':[.1,0,.3]}
        with patch('vista.robot_state',return_value=moved),self.assertRaisesRegex(ValueError,'stale'):
            self.session.dispatch('locate_stalk_proposal',self.args,'f',self.state,0)
        self.assertEqual(self.session.local_targets.active['failures'],0)

    def test_actual_fit_failure_keeps_shared_candidate_limit(self):
        session=self.session;original=session._dispatch
        def route(name,*args):
            if name=='locate_pixel':
                session._local_depth_attempted=True
                raise ValueError('ambiguous stalk')
            return original(name,*args)
        with patch.object(session,'_dispatch',side_effect=route),patch('vista.robot_state',return_value=self.state):
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                session.dispatch('locate_stalk_proposal',self.args,'f',self.state,0)
        self.assertEqual(session.station_attempts,1)
        self.assertEqual(session.local_targets.active['failures'],1)

    def test_exhausted_budget_does_not_enter_localization(self):
        self.session.station_attempts=12
        with patch.object(self.session,'_dispatch') as dispatch:
            result,_=self.session.dispatch('locate_stalk_proposal',self.args,'f',self.state,0)
        dispatch.assert_not_called();self.assertEqual(result['status'],'rejected')

    def test_tool_only_advertised_when_enabled(self):
        for enabled in (False,True):
            client=SimpleNamespace()
            Session(self.env,SimpleNamespace(out=Path(self.tmp.name)),client,visual_proposals=enabled)
            names={t['function']['name'] for t in client.tools}
            self.assertEqual('locate_stalk_proposal' in names,enabled)


if __name__=='__main__':unittest.main()
