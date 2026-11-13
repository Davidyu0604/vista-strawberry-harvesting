"""Unsuccessful finish and bounded exhaustion must not discard remaining work."""
import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from vista import Session
from collect_fresh50 import inspect_episode


class CollectionContinuationTests(unittest.TestCase):
    def session(self):
        s=Session.__new__(Session)
        s.require_delivery_before_finish=True;s.delivery_completed=False;s.finished=False
        s.calls=15;s.max_calls=32;s.visual_proposals=True;s.centered=Mock()
        s.target_stage=None;s.grasp_attempted=False;s.target_confirmed=False
        s.local_targets=None;s.motion_stop_trace=None;s.preclose_recovery_failed=False
        s.station_status=Mock(return_value={'next_action':'inspect fresh views'})
        return s

    def test_failed_finish_repeatedly_continues_even_near_budget(self):
        s=self.session()
        for count in (15,16,30,31,32):
            s.calls=count
            result,_=s._dispatch('finish',{'reason':'Cannot confirm stalk'},'f1093',{},0)
            self.assertFalse(result['episode_ended']);self.assertFalse(s.finished)
            self.assertEqual(result['remaining_model_calls'],32-count)
        s.centered.assert_not_called()

    def test_completed_delivery_permits_finish(self):
        s=self.session();s.delivery_completed=True;s.target_stage='grasp';s.grasp_attempted=True
        result,_=s._dispatch('finish',{'reason':'Delivery complete'},'f10',{},0)
        self.assertTrue(result['episode_ended']);self.assertTrue(s.finished)

    def test_readonly_inspect_still_works_after_station_budget(self):
        s=self.session();s.e=SimpleNamespace(meta={'tilted_stalk_prior':True})
        s.station_attempts=12;s._dispatch=Mock(return_value=({'inspected':True},[]))
        s.dispatch('inspect',{},'f10',{},0)
        s._dispatch.assert_called_once();self.assertEqual(s.station_attempts,12)

    def test_budget_exhausted_episode_is_archived_and_batch_can_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'source').mkdir();(p/'source/x.py').write_text('x')
            import hashlib
            frozen={'x.py':hashlib.sha256(b'x').hexdigest()}
            report=dict(success=False,model_calls=32,error=None,stop_reason='call_budget_exhausted')
            config=dict(seed=26,max_calls=32,initial_arm_pose='search_stow',basket_mount='chassis',
                        policy_cameras=['harvest_close','wrist'],mobile_layout=True,centered_grasp=True,
                        visual_proposals=True,mobile_layout_level='moderate',fixed_base=False,source_sha256=frozen)
            (p/'report.json').write_text(json.dumps(report));(p/'run-config.json').write_text(json.dumps(config))
            (p/'vista-events.jsonl').write_text((' {"kind":"request"}\n')*32)
            quality=dict(integrity_passed=True,recording_finalized=True,episode_completed=False,behaviour_review_blocks_training=False)
            with patch('collect_fresh50.audit_recording',return_value=quality),patch('collect_fresh50.source_hashes',return_value=frozen):
                result=inspect_episode(p,26,0,frozen)
                self.assertFalse(result['harvest_success']);self.assertFalse(result['episode_completed'])
                self.assertEqual(result['model_calls'],32)
                report.update(stop_reason='motion_stopped',error='TrackingError')
                (p/'report.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'Runtime error'):inspect_episode(p,26,0,frozen)


if __name__=='__main__':unittest.main()
