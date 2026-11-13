
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

from local_targets import LocalTargets, candidate_crop_region, body_pixel_evidence
from vista import Session, StalkFitError


def candidate(x,y=0,maturity='ripe',visibility='clear'):
    return {'surface_m':[x,y,.22],'maturity':maturity,'visibility':visibility,
            'x':200,'y':200,'reason':'visual red body','source_frame':'f'}


class LocalTargetChecks(unittest.TestCase):
    def queue(self):
        q=LocalTargets()
        q.register([candidate(.38),candidate(.25,visibility='partial'),candidate(.20,maturity='green')],[0,0,0],'all visible work area')
        return q

    def test_nearer_ripe_precedes_farther_clear_not_green(self):
        q=self.queue()
        self.assertEqual(q.active['surface_m'],[.25,0,.22])
        with self.assertRaisesRegex(ValueError,'nearest'):q.check_target([.38,0,.25])
        q.check_target([.25,0,.25])
        self.assertFalse(q.can_search)

    def test_two_failures_switch_fruit_not_station(self):
        q=self.queue();q.fail('bad pixel');self.assertEqual(q.active['surface_m'][0],.25)
        q.fail('bad fit');self.assertEqual(q.active['surface_m'][0],.38)
        self.assertFalse(q.can_search)
        q.fail('unreachable',terminal=True);self.assertTrue(q.can_search)

    def test_no_survey_is_not_evidence_for_moving(self):
        q=LocalTargets();self.assertFalse(q.can_search)
        with self.assertRaisesRegex(ValueError,'survey'):q.require_active()
        q.register([],[0,0,0],'work area contains only occluded plants')
        self.assertTrue(q.can_search)
        with self.assertRaisesRegex(ValueError,'already surveyed'):q.register([],[0,0,0],'retry')

    def test_deduplicates_body_pixels_and_snapshots_do_not_mutate(self):
        q=LocalTargets();q.register([candidate(.25),candidate(.252)],[0,0,0],'view')
        self.assertEqual(len(q.candidates),1)
        old=q.status();q.fail('bad pixel')
        self.assertEqual(old['active']['failures'],0)

    def session(self):
        s=Session.__new__(Session);s.local_targets=self.queue()
        s.e=SimpleNamespace(meta={'tilted_stalk_prior':True},drive=Mock(),frame=0)
        s.target_stage=None;s.grasp_attempted=False;s.target_confirmed=False
        s.locked_target=None;s.maturity_evidence=None;s.crops={};s.station_attempts=0;s.station_failures=0
        s.calls=4;s.max_calls=32;s.search_steps=0;s.fixed_base=False
        return s

    def test_controller_cannot_move_just_because_one_fruit_failed(self):
        s=self.session();s._dispatch=Mock(return_value=({'status':'rejected','reason':'bad point'},[]))
        s.dispatch('locate_pixel',{},'f',{},0);s.dispatch('locate_crop',{},'f',{},0)
        self.assertEqual(s.local_targets.active['surface_m'][0],.38)
        self.assertFalse(s.station_status()['station_limit_reached'])
        self.assertIsNone(s.locked_target)
        with self.assertRaisesRegex(ValueError,'Local-first'):s.search_row(1)
        with self.assertRaisesRegex(ValueError,'Local-first'):s.refine_parking()
        s.e.drive.assert_not_called()

    def test_budget_exhaustion_does_not_authorize_moving(self):
        s=self.session();s.station_attempts=12;s._dispatch=Mock()
        self.assertEqual(s.dispatch('inspect',{},'f',{},0)[0]['status'],'rejected')
        self.assertIn('finish',s.station_status()['next_action'])
        with self.assertRaisesRegex(ValueError,'Local-first'):s.search_row(1)
        s._dispatch.assert_not_called()

    def test_reachable_stays_even_if_optimizer_would_move(self):
        s=self.session();s.locked_target=np.array([.25,0,.25]);s.target_aware_parking=True
        with patch('vista.approach_reachability',return_value={'reachable':True}),patch('vista.optimized_parking_plan') as optimize:
            result=s.plan_parking()
        optimize.assert_not_called();self.assertIsNone(result['recommended_delta_y_m'])
        self.assertEqual(result['next_action'],'approach_target')

    def test_unreachable_advances_without_driving(self):
        s=self.session();s.target_confirmed=True;s.locked_target=np.array([.25,0,.25])
        s._dispatch=Mock(return_value=({'target_confirmed':True,'reachability':{'current':{'reachable':False}}},[]))
        result,_=s.dispatch('locate_pixel',{},'f',{},0)
        self.assertEqual(result['status'],'rejected');self.assertIsNone(s.locked_target)
        self.assertEqual(s.local_targets.active['surface_m'][0],.38)
        s.e.drive.assert_not_called()

    def test_single_failure_and_inspect_do_not_reset_candidate_count(self):
        s=self.session()
        def failed_fit(*args):
            s._local_depth_attempted=True
            raise ValueError('bad point')
        s._dispatch=Mock(side_effect=failed_fit)
        with self.assertRaises(ValueError):s.dispatch('locate_pixel',{},'f',{},0)
        s._dispatch=Mock(return_value=({'status':'ok'},[]));s.dispatch('inspect',{},'f',{},0)
        self.assertEqual(s.local_targets.active['failures'],1)

    def test_clearing_an_empty_lock_cannot_skip_near_candidate(self):
        s=self.session();active=s.local_targets.active
        for _ in range(3):s._dispatch('clear_target',{},'f',{},0)
        self.assertIs(s.local_targets.active,active)
        self.assertFalse(s.local_targets.can_search)

    def test_expired_crop_does_not_spend_next_fruit_retry(self):
        s=self.session();s.local_targets.fail('first');s.local_targets.fail('second')
        active=s.local_targets.active
        with self.assertRaisesRegex(ValueError,'expired crop'):
            s.dispatch('locate_crop',{'frame_id':'f','crop_id':'old','x':10,'y':20},'f',{},0)
        self.assertIs(s.local_targets.active,active)
        self.assertEqual(active['failures'],0)
        self.assertEqual(s.station_failures,0)
        self.assertEqual(s.station_attempts,1)  # Request still costs a model/perception call.

    def test_background_or_other_fruit_depth_does_not_charge_active(self):
        s=self.session();active=s.local_targets.active
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'depth.npz'
            # Central pixel unprojects onto the farther candidate, not active.
            np.savez(path,depth=np.array([[.4]]),position=[.38,0,.65],rotation=np.eye(3),fovy=60.)
            s.depth_frames={('f','harvest_close'):path}
            rgb=Path(tmp)/'rgb.png';Image.new('RGB',(1,1),(20,21,22)).save(rgb)
            s.frames={'f':{'harvest_close':rgb}}
            with patch('vista.tilted_stalk_grasp_point') as fit:
                result,images=s.dispatch('locate_pixel',{'frame_id':'f','camera':'harvest_close','x':0,'y':0},'f',{},0)
            fit.assert_not_called()
        self.assertEqual(result['selected_pixel'],[0,0])
        self.assertEqual(result['selected_rgb'],[20,21,22])
        self.assertEqual(len(images),1)
        self.assertFalse(result['candidate_failure_counted'])
        self.assertEqual(result['attempted_candidate_key'],active['candidate_key'])
        self.assertEqual(active['failures'],0)
        self.assertEqual(s.station_failures,0)
        self.assertEqual(s.station_attempts,1)
        self.assertFalse(s.local_targets.can_search)

    def test_crop_candidate_binding_rejects_without_charging(self):
        s=self.session();active=s.local_targets.active
        s.crops['old']={'call':s.calls,'candidate_key':'another-candidate'}
        result,_=s.dispatch('locate_crop',{'frame_id':'f','crop_id':'old','x':100,'y':100},'f',{},0)
        self.assertFalse(result['candidate_failure_counted'])
        self.assertEqual(active['failures'],0)

    def test_perception_limit_does_not_block_confirmed_motion(self):
        s=self.session();s.station_attempts=12;s.target_confirmed=True
        s.local_targets.ready();s._dispatch=Mock(return_value=({'status':'executed'},[]))
        status=s.station_status()
        self.assertEqual(status['remaining_perception_calls'],0)
        self.assertIn('approach_target',status['next_action'])
        self.assertTrue(status['ordinary_motion_uses_only_model_call_budget'])
        self.assertTrue(status['observation_adjustment_also_uses_perception_budget'])
        s.dispatch('approach_target',{},'f',{},0)
        s._dispatch.assert_called_once()

    def test_context_crop_contains_measured_body_and_stalk_zone(self):
        calibration={'position':np.array([0,0,1.]),'rotation':np.eye(3),'fovy':60.}
        box,uv=candidate_crop_region([.1,0,.3],calibration,(480,360))
        x,y,w,h=box
        self.assertTrue(0<=x<=uv[0]<x+w<=480)
        self.assertTrue(0<=y<=uv[1]<y+h<=360)
        self.assertGreater(w,30);self.assertGreater(h,30)
        self.assertIsNone(candidate_crop_region([0,0,2],calibration,(480,360)))

    def test_invalid_ripe_body_cannot_authorize_driving_or_farther_target(self):
        q=LocalTargets();bad={**candidate(.20),'body_pixel_verified':False}
        q.register([bad,candidate(.30)],[0,0,0],'model claimed background was ripe body')
        self.assertIsNone(q.active);self.assertFalse(q.can_search)
        self.assertEqual(q.status()['unverified_measurements'],1)
        self.assertEqual(q.status()['blocked_by_unverified_candidate'],'view-candidate-1')
        with self.assertRaisesRegex(ValueError,'candidates are NOT exhausted.*correct_candidate_body'):
            q.require_active()
        q.candidates[0]['correction_attempts']=2
        with self.assertRaisesRegex(ValueError,'corrections exhausted; finish'):q.require_active()
        self.assertEqual(q.candidates[1]['status'],'pending')
        s=self.session();s.local_targets=q
        self.assertIn('unverified',s.station_status()['next_action'])
        with self.assertRaisesRegex(ValueError,'Local-first'):s.search_row(1)
        s.e.drive.assert_not_called()

    def test_farther_unverified_claim_does_not_block_valid_near_fruit(self):
        q=LocalTargets();q.register([candidate(.20),{**candidate(.40),'body_pixel_verified':False}],
                                   [0,0,0],'near ripe body and farther uncertain depth')
        self.assertEqual(q.active['surface_m'][0],.20)
        self.assertIsNone(q.status()['blocked_by_unverified_candidate'])
        q.fail('fit failed');q.fail('fit failed')
        self.assertIsNone(q.active);self.assertFalse(q.can_search)

    def test_body_colour_guard_does_not_treat_background_as_absence(self):
        for colour in [(22,29,25),(40,100,25),(255,140,40)]:
            self.assertFalse(body_pixel_evidence(Image.new('RGB',(9,9),colour),4,4)['passed'])
        self.assertTrue(body_pixel_evidence(Image.new('RGB',(9,9),(220,25,30)),4,4)['passed'])

    def test_wrist_fit_uses_selected_camera_and_shared_candidate_budget(self):
        s=self.session();s.e.meta['harvest_mechanism']=True
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s.frames={'f':{}};s.depth_frames={}
            for camera,z,color in [('wrist',.4,(20,130,30)),('harvest_close',.7,(80,20,20))]:
                depth=root/(camera+'.npz');rgb=root/(camera+'.png')
                np.savez(depth,depth=np.array([[z]]),position=[.25,0,.65],rotation=np.eye(3),fovy=60.)
                Image.new('RGB',(1,1),color).save(rgb)
                s.frames['f'][camera]=rgb;s.depth_frames[('f',camera)]=depth
            def failed_fit(point,depth,calibration,image):
                self.assertAlmostEqual(float(depth[0,0]),.4)
                self.assertEqual(image.getpixel((0,0)),(20,130,30))
                raise StalkFitError('Occluded stalk',{'samples':0})
            with patch('vista.tilted_stalk_grasp_point',side_effect=failed_fit) as fit:
                result,_=s.dispatch('locate_pixel',{'frame_id':'f','camera':'wrist','x':0,'y':0},'f',{},0)
            fit.assert_called_once()
        self.assertEqual(result['stalk_fit_camera'],'wrist')
        self.assertEqual(result['suggested_camera'],'harvest_close')
        self.assertEqual(s.local_targets.active['failures'],1)
        self.assertEqual(s.station_attempts,1);self.assertFalse(s.local_targets.can_search)

    def test_active_crops_keep_distinct_camera_bindings(self):
        s=self.session();s.log=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'observations').mkdir();s.rec=SimpleNamespace(out=root)
            s.frames={'f':{}};s.depth_frames={}
            for camera in ('harvest_close','wrist'):
                path=root/(camera+'.npz');rgb=root/(camera+'.png')
                np.savez(path,depth=np.ones((360,480)),position=[.25,0,1],rotation=np.eye(3),fovy=60.)
                Image.new('RGB',(480,360),'green').save(rgb)
                s.frames['f'][camera]=rgb;s.depth_frames[('f',camera)]=path
            close,_=s.active_candidate_crop('f',{},camera='harvest_close')
            wrist,_=s.active_candidate_crop('f',{},camera='wrist')
            self.assertNotEqual(close['crop_id'],wrist['crop_id'])
            self.assertEqual(s.crops[wrist['crop_id']]['camera'],'wrist')
            self.assertEqual(s.crops[close['crop_id']]['camera'],'harvest_close')
            self.assertEqual(close['candidate_key'],wrist['candidate_key'])
            self.assertEqual(s.local_targets.active['failures'],0)

    def test_unverified_depth_cannot_deduplicate_away_another_claim(self):
        q=LocalTargets();q.register([{**candidate(.20),'body_pixel_verified':False},
            candidate(.201),{**candidate(.202),'body_pixel_verified':False}],[0,0,0],'view')
        self.assertEqual(len(q.candidates),3)
        self.assertFalse(q.can_search)

    def test_body_correction_preserves_identity_retries_and_nearest_order(self):
        q=self.queue();q.fail('one actual failed fit')
        bad=q.candidates[-1];bad.update(status='unverified_measurement',body_pixel_verified=False)
        key=bad['candidate_key']
        item=q.begin_body_correction(key,205,198)
        ok,_=q.correct_body(item,{**candidate(.20,.07),'x':205,'y':198,
            'body_pixel_verified':True,'body_pixel_check':{'passed':True}},[0,0,0])
        self.assertTrue(ok);self.assertEqual(q.active['candidate_key'],key)
        self.assertEqual(q.active['original_measurement']['surface_m'],[.38,0,.22])
        self.assertEqual(next(c for c in q.candidates if c['surface_m'][0]==.25)['failures'],1)
        with self.assertRaisesRegex(ValueError,'Only an unverified'):q.begin_body_correction(key,210,198)

    def test_failed_correction_cannot_walk_anchor_or_clear_unknown(self):
        q=LocalTargets();q.register([{**candidate(.20),'body_pixel_verified':False}],[0,0,0],'view')
        item=q.candidates[0];key=item['candidate_key']
        q.begin_body_correction(key,225,200)
        ok,_=q.correct_body(item,{**candidate(.25),'x':225,'body_pixel_verified':False,
            'body_pixel_check':{'passed':False}},[0,0,0])
        self.assertFalse(ok);self.assertEqual(item['x'],200)
        with self.assertRaisesRegex(ValueError,'ORIGINAL'):q.begin_body_correction(key,245,200)
        with self.assertRaisesRegex(ValueError,'exhausted'):q.begin_body_correction(key,201,200)
        self.assertFalse(q.can_search);self.assertEqual(item['failures'],0)

    def test_correction_cannot_clear_unknown_with_far_or_duplicate_body(self):
        for point,reason in [(.7,'working envelope'),(.30,'overlaps')]:
            q=LocalTargets();q.register([{**candidate(.20),'body_pixel_verified':False},candidate(.30)],
                                      [0,0,0],'view')
            item=q.begin_body_correction(q.candidates[0]['candidate_key'],205,200)
            ok,message=q.correct_body(item,{**candidate(point),'x':205,'body_pixel_verified':True,
                'body_pixel_check':{'passed':True}},[0,0,0])
            self.assertFalse(ok);self.assertIn(reason,message);self.assertFalse(q.can_search)

    def test_body_corrections_share_perception_budget_and_do_not_charge_stalk(self):
        s=self.session();s._dispatch=Mock(return_value=({'status':'rejected','reason':'no red body'},[]))
        s.dispatch('correct_candidate_body',{},'f',{},0)
        self.assertEqual(s.station_attempts,1);self.assertEqual(s.station_failures,0)
        s.station_attempts=12;s._dispatch.reset_mock()
        self.assertEqual(s.dispatch('correct_candidate_body',{},'f',{},0)[0]['status'],'rejected')
        s._dispatch.assert_not_called()

    def test_body_correction_dispatch_measures_current_rgbd_and_rejects_stale_camera(self):
        for moved in (False,True):
            s=self.session();s.max_wait=90
            s.local_targets=LocalTargets()
            s.local_targets.register([{**candidate(.3),'x':239,'y':179,'source_frame':'old',
                                      'body_pixel_verified':False}], [0,0,0],'view')
            key=s.local_targets.candidates[0]['candidate_key']
            state={'joints_rad':[0]*6,'tcp_world_m':[0,0,0],'base_x_y_yaw':[0,0,0]}
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);old=root/'old.npz';current=root/'current.npz';rgb=root/'rgb.png'
                np.savez(old,depth=np.full((360,480),.4),position=[.25,0,.62],rotation=np.eye(3),fovy=60.)
                np.savez(current,depth=np.full((360,480),.4),position=[.26 if moved else .25,0,.62],rotation=np.eye(3),fovy=60.)
                Image.new('RGB',(480,360),(220,25,30)).save(rgb)
                s.depth_frames={('old','harvest_close'):old,('f','harvest_close'):current}
                s.frames={'f':{'harvest_close':rgb}}
                args={'frame_id':'f','candidate_key':key,'x':239,'y':179,'reason':'same nearby red body'}
                with patch('vista.robot_state',return_value=state):
                    if moved:
                        with self.assertRaisesRegex(ValueError,'camera moved'):s.dispatch('correct_candidate_body',args,'f',state,0)
                        self.assertFalse(s.local_targets.can_search)
                    else:
                        result,images=s.dispatch('correct_candidate_body',args,'f',state,0)
                        self.assertEqual(result['status'],'corrected');self.assertEqual(len(images),1)
                        self.assertEqual(s.local_targets.active['candidate_key'],key)
                        self.assertAlmostEqual(s.local_targets.active['surface_m'][2],.22)
                self.assertEqual(s.station_attempts,1);self.assertEqual(s.station_failures,0)


if __name__=='__main__':unittest.main()
