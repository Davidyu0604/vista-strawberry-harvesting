"""Offline protocol and real MuJoCo integration tests; no paid API calls."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import httpx
import numpy as np
from vista import Client, Session, validate_actions, evaluate, robot_state, depth_point, check_observation, CallBudgetExhausted
from copy import deepcopy
from env import StrawberryEnv
from run import Recorder


def reply(name, args, reason='tool_calls'):
    return {'choices': [{'finish_reason': reason, 'message': {'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'test-call', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}


class Checks(unittest.TestCase):
    def test_inspect_rulers_preserve_source_pixels(self):
        from PIL import Image
        from vista import inspect_crop
        pixels = np.zeros((360,480,3), dtype=np.uint8)
        pixels[185,237] = [250, 40, 90]
        image, mapping = inspect_crop(Image.fromarray(pixels), [215,145,40,100])
        scale = mapping['scale']; ox,oy = mapping['display_origin_xy']
        # The known source landmark must remain at its advertised display position.
        np.testing.assert_array_equal(np.asarray(image)[oy+40*scale:oy+41*scale,ox+22*scale:ox+23*scale],
                                      np.broadcast_to([250,40,90],(scale,scale,3)))
        self.assertEqual(image.size,(40*scale+ox,100*scale+oy))

    def test_crop_selection_edges_and_margins(self):
        from PIL import Image
        from vista import inspect_crop, crop_source_pixel
        for region in ([0,0,1,1], [215,145,40,100], [0,0,480,360], [479,359,1,1]):
            image,mapping=inspect_crop(Image.new('RGB',(480,360)),region,local_coordinates=True)
            x,y,w,h=region; scale=mapping['scale']; ox,oy=mapping['display_origin_xy']
            self.assertEqual(crop_source_pixel(mapping,ox,oy),(x,y))
            self.assertEqual(crop_source_pixel(mapping,image.width-1,image.height-1),(x+w-1,y+h-1))
            for bad in ((ox-1,oy),(ox,oy-1),(image.width,oy),(ox,image.height),(True,oy),(ox,32.5)):
                with self.assertRaises(ValueError):crop_source_pixel(mapping,*bad)

    def test_stalk_rejection_diagnostics(self):
        from vista import stalk_grasp_point, StalkFitError
        depth=np.ones((10,10))
        calibration={'fovy':60,'rotation':np.eye(3),'position':np.zeros(3)}
        with self.assertRaises(StalkFitError) as caught:
            stalk_grasp_point(np.zeros(3),depth,calibration,np.zeros((10,10,3)))
        self.assertEqual(caught.exception.diagnostics['samples'],0)
        self.assertIsNone(caught.exception.diagnostics['visible_height_m'])

    def test_tilted_stalk_fit_from_rendered_depth(self):
        import mujoco
        import xml.etree.ElementTree as ET
        from scene import camera
        from vista import tilted_stalk_grasp_point, verify_projected_point, StalkFitError
        for degrees,length in ((0,.036),(15,.036),(15,.012)):
            theta=np.deg2rad(degrees); axis=np.array([np.sin(theta),0,np.cos(theta)])
            a=-axis*length/2;b=axis*length/2
            tree=ET.fromstring('<mujoco><visual><global offwidth="480" offheight="360"/></visual><worldbody><light pos="0 -.2 .3"/><geom type="capsule" size=".0015" rgba=".2 .65 .1 1"/></worldbody></mujoco>')
            world=tree.find('worldbody');world.find('geom').set('fromto',' '.join(map(str,[*a,*b])))
            camera(world,'test',[0,-.25,.08],[0,0,0],40)
            model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            renderer=mujoco.Renderer(model,360,480)
            try:
                renderer.update_scene(data,camera='test');rgb=renderer.render().copy()
                renderer.enable_depth_rendering();depth=renderer.render().copy()
                calibration={'rotation':data.cam_xmat[0].reshape(3,3),'position':data.cam_xpos[0],'fovy':40.}
                point=depth_point(depth,calibration,240,180)
                if length<.025:
                    with self.assertRaises(StalkFitError):tilted_stalk_grasp_point(point,depth,calibration,rgb)
                else:
                    target,fit=tilted_stalk_grasp_point(point,depth,calibration,rgb)
                    self.assertAlmostEqual(fit['visible_length_m'],length,delta=.004)
                    self.assertLess(np.linalg.norm(target-axis*(length/2-.024)),.004)
                    self.assertGreater(np.dot(fit['axis_world'],axis),np.cos(np.deg2rad(8)))
            finally:renderer.close()

    def test_contact_finger_drift_does_not_disable_pull(self):
        before = {'joints_rad':[.537,-.02,1.1,-1.095,-1.123,.4],
                  'tcp_world_m':[.263,-.141,.220], 'base_x_y_yaw':[0,0,0]}
        current = deepcopy(before)
        current['joints_rad'][5] = .1
        current['joints_rad'][0] += .0082
        current['tcp_world_m'][1] += .002
        check_observation(before,current,2,90)
        for field,index,delta in [('joints_rad',0,.05), ('tcp_world_m',2,.02),
                                  ('base_x_y_yaw',0,.02), ('base_x_y_yaw',2,.02)]:
            shifted = deepcopy(before); shifted[field][index] += delta
            with self.assertRaisesRegex(ValueError,'Observation stale'):
                check_observation(before,shifted,2,90)
        with self.assertRaises(ValueError):check_observation(before,before,91,90)

    def test_depth_unprojection_against_rendered_geometry(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="2 2 .01"/>
          <geom type="sphere" pos=".1 .05 .03" size=".03"/>
          <camera name="test" pos="0 0 1" fovy="60"/>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model); mujoco.mj_forward(model,data)
        renderer = mujoco.Renderer(model,100,200)
        try:
            renderer.update_scene(data,camera='test'); renderer.enable_depth_rendering()
            depth = renderer.render()
            calibration = {'position':data.cam_xpos[0], 'rotation':data.cam_xmat[0].reshape(3,3), 'fovy':60}
            plane = depth_point(depth,calibration,20,20)
            self.assertAlmostEqual(plane[2],0,places=4)
            self.assertLess(plane[0],0); self.assertGreater(plane[1],0)
            sphere = depth_point(depth,calibration,109,45)
            self.assertAlmostEqual(np.linalg.norm(sphere-[.1,.05,.03]), .03, delta=.003)
            with self.assertRaises(ValueError):depth_point(depth,calibration,200,20)
        finally:renderer.close()

    def test_batch_rejects_all_before_execution(self):
        for bad in [float('nan'), True, .081]:
            args = {'basis_frame_id':'f0', 'expectation':'move', 'actions':[
                {'action':'observe','seconds':.1}, {'action':'move_tcp','seconds':1,'dx':bad,'dy':0,'dz':0}]}
            with self.assertRaises(ValueError): validate_actions(args, 'f0')
        with self.assertRaises(ValueError): validate_actions(args, 'old')

    def make_client(self, root, handler):
        config = root/'fixture.env'
        config.write_text('LLM_BASE_URL=https://fixture.invalid/v1\nLLM_MODEL=fixture\nLLM_API_KEY=fixture-secret\n')
        with patch.dict('os.environ', {}, clear=True): client = Client(config)
        client.http.close(); client.http = httpx.Client(transport=httpx.MockTransport(handler))
        return client

    def test_incomplete_response_does_not_execute_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            requests = []
            def handler(request):
                requests.append(request)
                return httpx.Response(200, json=reply('finish', {'reason':'truncated'}, 'length'))
            client = self.make_client(Path(tmp), handler)
            try:
                with self.assertRaises(RuntimeError): client.call([])
                self.assertEqual(len(requests), 1)
            finally: client.close()

    def test_actual_simulation_visual_tool_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); requests = []
            def handler(request):
                body = json.loads(request.content); requests.append(body)
                observation = body['messages'][-1]['content']
                info = json.loads(observation[0]['text'])
                self.assertEqual(set(info), {'frame_id','simulation_time','camera_image_size','joints_rad','tcp_world_m','base_x_y_yaw'})
                self.assertEqual(sum(c['type']=='image_url' for c in observation), 2)
                self.assertNotIn('fruit_positions', json.dumps(body))
                time.sleep(.25)
                if len(requests) == 1:
                    payload = reply('play', {'basis_frame_id':info['frame_id'],'expectation':'must not execute',
                        'actions':[{'action':'drive','linear':.06,'angular':0,'seconds':1}]})
                    other = reply('read_guide', {})['choices'][0]['message']['tool_calls'][0]
                    other['id'] = 'second-test-call'
                    payload['choices'][0]['message']['tool_calls'].append(other)
                    return httpx.Response(200,json=payload)
                if len(requests) == 2:
                    return httpx.Response(200,json=reply('play',{'basis_frame_id':info['frame_id'],'expectation':'small upward motion',
                        'actions':[{'action':'move_tcp','dx':0,'dy':0,'dz':.005,'seconds':1}]}))
                return httpx.Response(200,json=reply('finish',{'reason':'offline interface check, not a harvest'}))
            client = self.make_client(root, handler); e = StrawberryEnv(0,render_quality='fast')
            try:
                r = Recorder(e,root/'episode',False,False); session = Session(e,r,client,max_calls=3)
                session.run()
                self.assertTrue(session.finished)
                self.assertGreater(session.wait_ticks,0)
                self.assertTrue(any(row['phase']=='move_tcp' for row in r.rows))
                self.assertFalse(any(row['phase']=='drive' for row in r.rows))
                self.assertGreater(r.rows[-1]['tcp_position'][2]-r.rows[0]['tcp_position'][2], .002)
                self.assertFalse(evaluate(e,r.rows)['success'])
                frame = next(iter(session.frames)); state = robot_state(e)
                result,_ = session.dispatch('locate_pixel',{'frame_id':frame,'camera':'top','x':50,'y':300},frame,state,0)
                self.assertEqual(len(result['surface_world_m']),3)
                with self.assertRaises(ValueError):
                    session.dispatch('locate_pixel',{'frame_id':'old','camera':'top','x':50,'y':300},frame,state,0)
                args = {'basis_frame_id':frame,'expectation':'stale motion','actions':[{'action':'observe','seconds':.1}]}
                with self.assertRaises(ValueError): session.dispatch('play',args,frame,state,100)
                self.assertNotIn('fixture-secret',session.events.read_text())
                self.assertEqual(len(requests),3)
                print(json.dumps({'offline_calls':3,'wait_ticks':session.wait_ticks,'frames':len(r.rows),'false_success_prevented':True,'multiple_calls_rejected_without_motion':True}))
                requests.clear()
                limited = Session(e,r,client,max_calls=1)
                with self.assertRaises(CallBudgetExhausted):limited.run()
                self.assertFalse(limited.finished)
                np.testing.assert_array_equal(e.drive_command,[0,0])
            finally: e.close(); client.close()


if __name__ == '__main__': unittest.main()
