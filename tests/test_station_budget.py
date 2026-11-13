
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from vista import Session


class StationBudgetChecks(unittest.TestCase):
    def session(self):
        s=Session.__new__(Session)
        s.e=SimpleNamespace(meta={'tilted_stalk_prior':True})
        s.target_stage=None;s.grasp_attempted=False;s.target_confirmed=False
        s.station_attempts=0;s.station_failures=0
        s._dispatch=Mock(side_effect=lambda *args:({'status':'ok'},[]))
        return s

    def test_six_inspects_then_search_required(self):
        s=self.session()
        for _ in range(6):s.dispatch('inspect',{},'f',{},0)
        result,_=s.dispatch('inspect',{},'f',{},0)
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(s._dispatch.call_count,6)
        self.assertEqual(result['station_search']['next_action'],'search_row or finish')
        s.dispatch('clear_target',{},'f',{},0)
        self.assertTrue(s.station_status()['station_limit_reached'])
        s.dispatch('search_row',{},'f',{},0)
        self.assertEqual(s._dispatch.call_args.args[0],'search_row')

    def test_two_failures_not_reset_by_inspection(self):
        s=self.session()
        s._dispatch.side_effect=lambda name,*args:({'status':'rejected' if name=='locate_crop' else 'ok'},[])
        s.dispatch('locate_crop',{},'f',{},0)
        s.dispatch('inspect',{},'f',{},0)
        s.dispatch('locate_crop',{},'f',{},0)
        self.assertEqual(s.station_failures,2)
        self.assertEqual(s.station_attempts,3)
        self.assertEqual(s.dispatch('locate_pixel',{},'f',{},0)[0]['status'],'rejected')
        self.assertEqual(s._dispatch.call_count,3)

    def test_successful_target_can_approach_at_limit(self):
        s=self.session();s.station_attempts=6;s.target_confirmed=True
        self.assertEqual(s.station_status()['next_action'],'approach_target or finish')
        s.dispatch('approach_target',{},'f',{},0)
        self.assertEqual(s._dispatch.call_count,1)

    def test_exception_counts_failure(self):
        s=self.session();s._dispatch.side_effect=ValueError('bad point')
        with self.assertRaises(ValueError):s.dispatch('locate_pixel',{},'f',{},0)
        self.assertEqual(s.station_failures,1)
