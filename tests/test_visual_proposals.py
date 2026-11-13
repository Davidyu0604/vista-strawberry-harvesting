
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import tempfile
import numpy as np
from PIL import Image,ImageDraw

from visual_proposals import propose_bodies,reviewed_candidates,draw_proposals,normalize_review_frame
from local_targets import LocalTargets,body_pixel_evidence
from vista import Session,depth_point


class ProposalChecks(unittest.TestCase):
    def test_current_frame_alias_does_not_relax_review_fields_or_staleness(self):
        rgb,depth,cal,b=self.fixture();original=self.review(b)
        for extra in (False,True):
            args={**original,'basis_frame_id':'f'}
            if not extra:args.pop('frame_id')
            normalized=normalize_review_frame(args,'f')
            self.assertEqual(normalized,original)
            self.assertIn('basis_frame_id',args)
            reviewed_candidates(normalized,b,'f',rgb)
        for fields in ({'basis_frame_id':'old'},{'basis_frame_id':'f','frame_id':'old'}):
            with self.assertRaisesRegex(ValueError,'current frame_id'):
                normalize_review_frame({**original,**fields},'f')
        with self.assertRaisesRegex(ValueError,'extra='):
            reviewed_candidates(normalize_review_frame({**original,'basis_frame_id':'f','unknown':True},'f'),b,'f',rgb)
        with self.assertRaisesRegex(ValueError,'missing='):
            reviewed_candidates(normalize_review_frame({'basis_frame_id':'f'},'f'),b,'f',rgb)

    def fixture(self):
        rgb=Image.new('RGB',(100,80),'black');d=ImageDraw.Draw(rgb)
        d.ellipse((15,20,35,45),fill=(210,10,20))
        d.ellipse((60,30,85,60),fill=(220,10,20))
        d.rectangle((22,28,24,32),fill='white')  # Seeds must not become body samples.
        depth=np.full((80,100),.4)
        cal={'position':[0,0,.65],'rotation':np.eye(3),'fovy':60.}
        return rgb,depth,cal,propose_bodies(rgb,depth,cal,[0,0,0],depth_point)

    def review(self,batch):
        return {'frame_id':'f','coverage':'entire work area checked','missing_visible_ripe':False,
                'candidates':[{'proposal_id':p['proposal_id'],'maturity':'ripe','visibility':'clear',
                               'reason':'red body visible'} for p in batch['proposals']]}

    def test_interior_red_depth_and_order_without_scene_access(self):
        rgb,depth,cal,b=self.fixture();self.assertEqual(len(b['proposals']),2)
        self.assertFalse(b['coverage_certified'])
        for p in b['proposals']:
            self.assertTrue(body_pixel_evidence(rgb,*p['pixel'])['passed'])
            self.assertTrue(p['depth_verified'])
        ds=[p['distance_xy_m'] for p in b['proposals']];self.assertEqual(ds,sorted(ds))
        self.assertEqual(draw_proposals(rgb,b).size,(200,160))

    def test_depth_hole_chooses_another_interior_sample(self):
        rgb,depth,cal,b=self.fixture();x,y=b['proposals'][0]['pixel'];depth[y,x]=np.nan
        new=propose_bodies(rgb,depth,cal,[0,0,0],depth_point)
        self.assertEqual(len(new['proposals']),2)
        self.assertTrue(all(p['depth_verified'] for p in new['proposals']))
        self.assertNotIn([x,y],[p['pixel'] for p in new['proposals']])

    def test_empty_image_is_not_certified_coverage(self):
        rgb,depth,cal,_=self.fixture()
        b=propose_bodies(Image.new('RGB',rgb.size,'green'),depth,cal,[0,0,0],depth_point)
        self.assertEqual(b['proposals'],[]);self.assertFalse(b['coverage_certified'])

    def test_orange_context_keeps_red_flecks_on_one_measured_body(self):
        rgb,depth,cal,_=self.fixture();rgb=Image.new('RGB',rgb.size,'black');d=ImageDraw.Draw(rgb)
        d.ellipse((20,20,55,60),fill=(230,90,25))
        for y in (30,40,50):d.line((29,y,40,y),fill=(220,10,20))
        batch=propose_bodies(rgb,depth,cal,[0,0,0],depth_point)
        self.assertEqual(len(batch['proposals']),1)
        p=batch['proposals'][0];self.assertTrue(p['depth_verified'])
        self.assertEqual(p['bbox_xyxy'],[20,20,55,60])
        self.assertEqual(p['red_area_pixels'],36)
        self.assertGreater(p['body_area_pixels'],500)
        # A measured orange body is not approved ripe-body evidence.
        self.assertFalse(body_pixel_evidence(rgb,*p['pixel'])['passed'])
        measured,blocked=reviewed_candidates(self.review(batch),batch,'f',rgb)
        q=LocalTargets();q.register(measured,[0,0,0],'whole image')
        self.assertIsNone(q.active);self.assertFalse(q.can_search)
        review=self.review(batch);review['candidates'][0]['maturity']='turning'
        measured,blocked=reviewed_candidates(review,batch,'f',rgb)
        q=LocalTargets();q.register(measured,[0,0,0],'whole image')
        self.assertIsNone(blocked);self.assertTrue(q.can_search)

    def test_orange_expansion_does_not_propose_unseeded_background(self):
        rgb,depth,cal,_=self.fixture();rgb=Image.new('RGB',rgb.size,(180,75,30))
        b=propose_bodies(rgb,depth,cal,[0,0,0],depth_point)
        self.assertEqual(b['proposals'],[]);self.assertFalse(b['coverage_certified'])

    def test_orange_context_refresh_preserves_region_and_depth_holes(self):
        from visual_proposals import refresh_proposals
        rgb,depth,cal,_=self.fixture();rgb=Image.new('RGB',rgb.size,'black');d=ImageDraw.Draw(rgb)
        d.ellipse((20,20,55,60),fill=(230,90,25));d.line((29,30,40,30),fill='red')
        hole=np.full_like(depth,np.nan)
        b=propose_bodies(rgb,hole,cal,[0,0,0],depth_point)
        self.assertIsNone(b['proposals'][0]['surface_m'])
        still=refresh_proposals(b,rgb,hole,cal,[0,0,0],depth_point)
        self.assertIsNone(still['proposals'][0]['surface_m'])
        new=refresh_proposals(b,rgb,depth,cal,[0,0,0],depth_point)
        self.assertTrue(new['proposals'][0]['depth_verified'])
        self.assertEqual(new['proposals'][0]['bbox_xyxy'],b['proposals'][0]['bbox_xyxy'])
        self.assertEqual(new['proposals'][0]['proposal_id'],b['proposals'][0]['proposal_id'])

    def test_two_bodies_joined_by_red_bridge_are_separate_proposals(self):
        rgb,depth,cal,_=self.fixture();rgb=Image.new('RGB',rgb.size,'black');d=ImageDraw.Draw(rgb)
        d.ellipse((20,20,45,50),fill='red');d.ellipse((48,23,73,53),fill='red')
        d.rectangle((40,33,53,38),fill='red')
        b=propose_bodies(rgb,depth,cal,[0,0,0],depth_point)
        self.assertEqual(len(b['proposals']),2)
        self.assertTrue(any(p['pixel'][0]<45 for p in b['proposals']))
        self.assertTrue(any(p['pixel'][0]>48 for p in b['proposals']))

    def test_thin_far_fragment_excluded_but_unknown_depth_preserved(self):
        rgb,depth,cal,_=self.fixture();rgb=Image.new('RGB',rgb.size,'black');d=ImageDraw.Draw(rgb)
        d.rectangle((15,20,30,20),fill='red');depth[:]=1.9
        self.assertEqual(propose_bodies(rgb,depth,cal,[0,0,0],depth_point)['proposals'],[])
        depth[20,20]=np.nan
        b=propose_bodies(rgb,depth,cal,[0,0,0],depth_point)
        self.assertEqual(len(b['proposals']),1);self.assertFalse(b['proposals'][0]['depth_verified'])

    def test_cannot_invent_omit_duplicate_or_use_stale_labels(self):
        rgb,_,_,b=self.fixture()
        for mutate in (lambda a:a.update(frame_id='old'),lambda a:a['candidates'].pop(),
                       lambda a:a['candidates'].append(a['candidates'][0]),
                       lambda a:a['candidates'][0].update(proposal_id='P999'),
                       lambda a:a['candidates'][0].update(x=208,y=155)):
            a=self.review(b);mutate(a)
            with self.assertRaises(ValueError):reviewed_candidates(a,b,'f',rgb)

    def test_uncertain_near_proposal_blocks_far_and_driving(self):
        rgb,_,_,b=self.fixture();a=self.review(b);a['candidates'][0]['maturity']='uncertain'
        measured,blocked=reviewed_candidates(a,b,'f',rgb);q=LocalTargets();q.register(measured,[0,0,0],a['coverage'])
        q.selection_blocked_reason=blocked
        self.assertIsNone(q.active);self.assertFalse(q.can_search)
        with self.assertRaisesRegex(ValueError,'uncertain'):q.require_active()
        s=Session.__new__(Session);s.local_targets=q;s.target_confirmed=False
        self.assertIn('uncertain',s.station_status()['next_action'])

    def test_missing_depth_missing_fruit_and_truncation_cannot_authorize_motion(self):
        for kind in ('depth','missing','limit'):
            rgb,depth,cal,b=self.fixture();a=self.review(b)
            if kind=='depth':
                depth[:]=np.nan;b=propose_bodies(rgb,depth,cal,[0,0,0],depth_point);a=self.review(b)
            if kind=='missing':a['missing_visible_ripe']=True
            if kind=='limit':b['omitted_by_limit']=1
            measured,blocked=reviewed_candidates(a,b,'f',rgb)
            self.assertIsNotNone(blocked)
            q=LocalTargets();q.register(measured,[0,0,0],a['coverage']);q.selection_blocked_reason=blocked
            self.assertIsNone(q.active);self.assertFalse(q.can_search)
            with self.assertRaises(ValueError):q.require_active()

    def test_session_registration_uses_measured_pixels_and_existing_budget(self):
        rgb,_,_,b=self.fixture();a=self.review(b)
        s=Session.__new__(Session);s.local_targets=LocalTargets();s.visual_proposals=True
        s.proposal_frames={'f':b};s.target_stage=None;s.grasp_attempted=False;s.target_confirmed=False
        s.e=SimpleNamespace(meta={'tilted_stalk_prior':True});s.max_wait=90
        s.station_attempts=0;s.station_failures=0
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rgb.png';rgb.save(path);s.frames={'f':{'harvest_close':path}}
            with patch('vista.robot_state'),patch('vista.check_observation'):
                result,_=s.dispatch('survey_candidates',a,'f',{'base_x_y_yaw':[0,0,0]},0)
                self.assertEqual(result['status'],'registered')
                self.assertEqual(s.station_attempts,1)
                self.assertEqual(s.local_targets.active['x'],b['proposals'][0]['pixel'][0])
                with self.assertRaisesRegex(ValueError,'already surveyed'):
                    s.dispatch('survey_candidates',a,'f',{'base_x_y_yaw':[0,0,0]},0)

    def test_opt_in_schema_and_original_schema_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=SimpleNamespace(meta={'tilted_stalk_prior':True})
            client=SimpleNamespace();rec=SimpleNamespace(out=Path(tmp))
            Session(env,rec,client,visual_proposals=True)
            props=next(t for t in client.tools if t['function']['name']=='survey_candidates')['function']['parameters']['properties']
            self.assertIn('missing_visible_ripe',props)
            self.assertIn('proposal_id',props['candidates']['items']['properties'])
            legacy=SimpleNamespace();Session(env,rec,legacy)
            props=next(t for t in legacy.tools if t['function']['name']=='survey_candidates')['function']['parameters']['properties']
            self.assertNotIn('missing_visible_ripe',props)
            self.assertIn('x',props['candidates']['items']['properties'])

    def test_observation_to_review_uses_current_frame_and_both_cameras(self):
        rgb,_,cal,_=self.fixture();rgb=rgb.resize((480,360))
        state={'base_x_y_yaw':[0,0,0],'joints_rad':[0]*6,'tcp_world_m':[0,0,0]}
        with tempfile.TemporaryDirectory() as tmp:
            env=SimpleNamespace(meta={'tilted_stalk_prior':True,'policy_cameras':['harvest_close','wrist']},
                frame=10,data=SimpleNamespace(time=.5),render=lambda *a:np.asarray(rgb),
                depth_view=lambda *a:(np.full((360,480),.4),cal))
            session=Session(env,SimpleNamespace(out=Path(tmp)),SimpleNamespace(),visual_proposals=True)
            with patch('vista.robot_state',return_value=state):
                frame,observed,message=session.observe()
                batch=session.proposal_frames[frame]
                self.assertEqual(set(session.frames[frame]),{'harvest_close','wrist'})
                # Original close, numbered close, wrist: still two sensors.
                self.assertEqual(sum(c['type']=='image_url' for c in message['content']),3)
                review=self.review(batch);review['frame_id']=frame
                result,_=session.dispatch('survey_candidates',review,frame,observed,0)
            self.assertEqual(result['status'],'registered')
            self.assertIsNotNone(session.local_targets.active)
            self.assertEqual(len(list((Path(tmp)/'observations').glob('*'))),4)


if __name__=='__main__':unittest.main()
