"""Compiler sleep compatibility must preserve physics and unrelated errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import unittest
import xml.etree.ElementTree as ET
import mujoco
from env import compile_scene


class InitialSleepChecks(unittest.TestCase):
    def test_contacting_sleep_roots_start_awake_and_reset(self):
        xml = '''<mujoco><option><flag sleep="enable"/></option><worldbody>
          <body name="awake"><freejoint/><geom size=".1"/></body>
          <body name="sleep_a" pos=".15 0 0" sleep="init"><freejoint/><geom size=".1"/></body>
          <body name="sleep_b" pos=".30 0 0" sleep="init"><freejoint/><geom size=".1"/></body>
          <body name="far" pos="2 0 0" sleep="init"><freejoint/><geom size=".1"/></body>
        </worldbody></mujoco>'''
        model, fixed, names = compile_scene(xml)
        self.assertEqual(set(names), {'sleep_a', 'sleep_b'})
        before, after = ET.fromstring(xml), ET.fromstring(fixed)
        for name in names:
            before.find(f".//body[@name='{name}']").set('sleep', 'auto')
        self.assertEqual(ET.tostring(before), ET.tostring(after))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        self.assertGreaterEqual(data.tree_asleep[model.body_treeid[model.body('far').id]], 0)

    def test_valid_scene_is_byte_identical(self):
        xml = '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        _, fixed, names = compile_scene(xml)
        self.assertEqual(fixed, xml)
        self.assertEqual(names, [])

    def test_unrelated_compile_error_propagates(self):
        with self.assertRaises(ValueError):
            compile_scene('<mujoco><worldbody><invalid/></worldbody></mujoco>')


if __name__ == '__main__':
    unittest.main()
