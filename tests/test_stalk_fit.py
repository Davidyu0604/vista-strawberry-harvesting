"""Regression: nearby green leaves need not contain any thin-stalk samples."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from vista import StalkFitError, tilted_stalk_grasp_point


class StalkFitChecks(unittest.TestCase):
    def test_empty_and_sparse_tube_rejected_before_quantiles(self):
        depth=np.full((120,160),.5)
        calibration={'fovy':42,'rotation':np.eye(3),'position':np.zeros(3)}
        for center_sample in (False,True):
            rgb=np.zeros((120,160,3),dtype=np.uint8)
            rgb[57:63,85:92]=[30,150,30]
            if center_sample:rgb[60,80]=[30,150,30]
            with self.subTest(center_sample=center_sample):
                with patch('vista.np.quantile',side_effect=AssertionError('quantile must not run')):
                    with self.assertRaises(StalkFitError) as caught:
                        tilted_stalk_grasp_point(np.array([0,0,-.5]),depth,calibration,Image.fromarray(rgb))
                self.assertEqual(caught.exception.diagnostics['samples'],int(center_sample))
                self.assertIsNone(caught.exception.diagnostics['visible_length_m'])
