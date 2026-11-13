
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import copy
import unittest
from validate_centered_cycle import audit_cycle


class AcceptanceChecks(unittest.TestCase):
    def fixture(self):
        return {'offline_associated_fruit':2,'stable_target_stalk_hold':True,'failure':None,
            'continuation':{'error':None,'warnings':[0],
                'evaluation':{'success':True,'single_fruit_delivered':True,'ripe_fruit_in_basket':[2]},
                'trace':[{'phase':'bend_pull','stalk_bilateral_ids_offline':[2]},
                         {'phase':'delivery_lower','stalk_bilateral_ids_offline':[2]},
                         {'phase':'delivery_release','stalk_bilateral_ids_offline':[]}]}}

    def test_released_fruit_need_not_remain_grasped(self):
        self.assertTrue(audit_cycle(self.fixture())['full_cycle_passed'])

    def test_other_ripe_fruit_or_contact_loss_is_not_accepted(self):
        original=self.fixture()
        wrong=copy.deepcopy(original);wrong['continuation']['evaluation']['ripe_fruit_in_basket']=[1]
        self.assertFalse(audit_cycle(wrong)['full_cycle_passed'])
        lost=copy.deepcopy(original);lost['continuation']['trace'][0]['stalk_bilateral_ids_offline']=[1]
        self.assertFalse(audit_cycle(lost)['full_cycle_passed'])

    def test_completed_motion_does_not_override_failure(self):
        d=self.fixture();d['continuation']['error']='tracking error'
        self.assertFalse(audit_cycle(d)['full_cycle_passed'])
        d=self.fixture();d['continuation']['evaluation']['success']=False
        self.assertFalse(audit_cycle(d)['full_cycle_passed'])


if __name__=='__main__':unittest.main()
