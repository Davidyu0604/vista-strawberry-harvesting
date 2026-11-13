"""VISTA-style visual/tool loop. Network worker never accesses MuJoCo.

Robot observations are separated from evaluator/Recorder ground truth. No hardware imports.
"""
import argparse
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import sys
from pathlib import Path
import time
from urllib.parse import urlsplit

# Select before env imports MuJoCo/PyOpenGL. The native viewer uses GLFW/GLX;
# an EGL offscreen context cannot safely share that thread's current GLX state.
if __name__ == '__main__' and '--viewer' in sys.argv:
    os.environ['MUJOCO_GL'] = 'glfw'
    if sys.platform.startswith('linux'):
        os.environ['PYOPENGL_PLATFORM'] = 'glx'

from visual_maturity import assess_maturity
from preclose_vision import check_stalk_alignment
from local_targets import LocalTargets, CandidateCorrespondenceError, candidate_crop_region, body_pixel_evidence
from target_parking import parking_plan, optimized_parking_plan, approach_reachability
from centered_grasp import TrackingError, PullPlanError
from perception_recovery import can_correct_survey, correct_survey, observation_adjustment_available, can_review_proposals, review_proposals, uncertain_review_candidate, missing_stalk_views, current_body_proposals, exhausted_observation_recovery
from perception_recovery import unconfirmed_stalk_options, review_unconfirmed_stalks, uncertain_region_evidence

import httpx
import numpy as np
from PIL import Image, ImageDraw
from dotenv import dotenv_values
from env import StrawberryEnv
from run import Recorder
from scene import ROOT
from project_paths import project_source


def obj(properties, required=None):
    return dict(type='object', properties=properties,
                required=list(properties) if required is None else required, additionalProperties=False)


NUM = {'type': 'number'}
STR = {'type': 'string'}
DELTA = {'type': 'number', 'minimum': -.08, 'maximum': .08}
DURATION = {'type': 'number', 'minimum': .1, 'maximum': 3}
ACTION = {'oneOf': [
    obj({'action': {'const': 'bend_pull'}, 'dx': DELTA, 'dy': DELTA, 'dz': DELTA,
         'pitch_deg': {'type':'number','minimum':-45,'maximum':45}, 'seconds': DURATION}),
    obj({'action': {'const': 'move_tcp'}, 'dx': DELTA, 'dy': DELTA, 'dz': DELTA, 'seconds': DURATION}),
    obj({'action': {'const': 'drive'}, 'linear': {'type':'number','minimum':-.06,'maximum':.06}, 'angular': {'type':'number','minimum':-.25,'maximum':.25}, 'seconds': DURATION}),
    obj({'action': {'const': 'gripper'}, 'opening': {'type':'number','minimum':-.15,'maximum':1}, 'seconds': DURATION}),
    obj({'action': {'const': 'observe'}, 'seconds': DURATION}),
]}
SCHEMAS = {
    'confirm_detachment':obj({'basis_frame_id':STR,'reference_frame_id':STR,
        'views':{'type':'array','minItems':2,'maxItems':2,'items':obj({'camera':{'enum':['harvest_close','wrist']},
            'held_fruit_visible':{'type':'boolean'},'separation':{'enum':['visible_gap','connected','occluded']},'reason':STR})}}),
    'locate_stalk_proposal': obj({'frame_id':STR,'candidate_key':STR,'proposal_id':STR,
                                'confirmed_short_stalk':{'const':True},'reason':STR}),
    'correct_candidate_body': obj({'frame_id':STR,'candidate_key':STR,
        'x':{'type':'integer','minimum':0,'maximum':479},
        'y':{'type':'integer','minimum':0,'maximum':359},'reason':STR}),
    'survey_candidates': obj({'frame_id':STR, 'coverage':STR,
        'candidates':{'type':'array','maxItems':8,'items':obj({
            'x':{'type':'integer','minimum':0,'maximum':479},
            'y':{'type':'integer','minimum':0,'maximum':359},
            'maturity':{'enum':['ripe','turning','green','uncertain']},
            'visibility':{'enum':['clear','partial','occluded']},'reason':STR})}}),
    'refine_parking': obj({'basis_frame_id':STR}),
    'review_proposals': obj({}),
    'correct_survey': obj({}),  # Expanded to the numbered full-review schema below.
    'adjust_observation': obj({'basis_frame_id':STR,'candidate_key':STR,'reason':STR,
        'unconfirmed_stalk_proposals':{'type':'array','minItems':1,'maxItems':16,
            'items':obj({'proposal_id':STR,'reason':STR})}},['basis_frame_id','candidate_key','reason']),
    'search_after_unlisted_review': obj({'basis_frame_id':STR,'all_missing_claims_reported':{'const':True},
        'bodies':{'type':'array','minItems':1,'maxItems':8,'items':obj({'x':{'type':'integer','minimum':0,'maximum':479},
        'y':{'type':'integer','minimum':0,'maximum':359},'reason':STR})}}),
    'inspect_unlisted': obj({'basis_frame_id':STR,
        'bodies':{'type':'array','minItems':1,'maxItems':8,'items':obj({'x':{'type':'integer','minimum':0,'maximum':479},
        'y':{'type':'integer','minimum':0,'maximum':359},'reason':STR})}}),
    'search_row': obj({'basis_frame_id':STR,'direction':{'type':'integer','enum':[-1,1]}}),
    'deliver_to_basket': obj({'basis_frame_id':STR}),
    'clear_target': obj({}),
    'approach_target': obj({'basis_frame_id': STR, 'stage': {'enum':['pregrasp','grasp']}}),
    'locate_pixel': obj({'frame_id': STR, 'camera': {'enum': ['top','wrist','harvest_close']},
                         'x': {'type':'integer','minimum':0,'maximum':479},
                         'y': {'type':'integer','minimum':0,'maximum':359}}),
    'locate_crop': obj({'frame_id': STR, 'crop_id': STR,
                        'x': {'type':'integer','minimum':0},
                        'y': {'type':'integer','minimum':0}}),
    'play': obj({'basis_frame_id': STR, 'expectation': STR,
                 'actions': {'type': 'array', 'items': ACTION, 'minItems': 1, 'maxItems': 4}}),
    'inspect': obj({'frame_id': STR, 'camera': {'enum': ['top', 'wrist','harvest_close']},
                    'region': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}},
                   ['frame_id', 'camera']),
    'write_working': obj({'content': STR}),
    'read_working': obj({}),
    'history': obj({'last': {'type': 'integer', 'minimum': 1, 'maximum': 10}}),
    'read_guide': obj({}),
    'propose_guide': obj({'content': STR}),
    'finish': obj({'reason': STR}),
}
TOOLS = [{'type': 'function', 'function': {'name': k, 'description': k, 'parameters': v}}
         for k, v in SCHEMAS.items()]


def finite(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Expected finite number in [{low}, {high}]')
    return float(value)


def validate_actions(args, frame):
    if set(args) != {'basis_frame_id', 'expectation', 'actions'} or args['basis_frame_id'] != frame:
        raise ValueError('Invalid or stale basis_frame_id')
    if not isinstance(args['expectation'], str) or not 1 <= len(args['expectation']) <= 4000:
        raise ValueError('expectation must be nonempty text')
    actions = args['actions']
    if not isinstance(actions, list) or not 1 <= len(actions) <= 4:
        raise ValueError('Expected 1–4 actions')
    fields = {'move_tcp': {'dx', 'dy', 'dz'}, 'bend_pull': {'dx','dy','dz','pitch_deg'}, 'drive': {'linear', 'angular'},
              'gripper': {'opening'}, 'observe': set()}
    for a in actions:
        if not isinstance(a, dict) or a.get('action') not in fields:
            raise ValueError('Unknown action')
        name = a['action']
        if set(a) != fields[name] | {'action', 'seconds'}:
            raise ValueError('Unexpected action fields')
        finite(a['seconds'], .1, 3)
        for key in fields[name]:
            bounds = (-45,45) if key=='pitch_deg' else (-.08, .08) if name in ('move_tcp','bend_pull') else (
                (-.06, .06) if key == 'linear' else (-.25, .25) if key == 'angular' else (-.15, 1))
            finite(a[key], *bounds)
    return actions


class ModelServiceError(RuntimeError):
    """Sanitized service failure; retry policy belongs to the budgeted session."""
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class Client:
    def __init__(self, env_file, timeout=60):
        values = {**dotenv_values(env_file), **os.environ}
        def get(*names, default=None):
            return next((values[n] for n in names if values.get(n)), default)
        base = get('LLM_BASE_URL', 'OPENAI_BASE_URL')
        self.model = get('LLM_MODEL', 'OPENAI_MODEL')
        if not base or not self.model:
            raise ValueError('请在 strawberry-sim/.env 配置 LLM_BASE_URL、LLM_MODEL、LLM_API_KEY，或用 --env-file 指定旧配置')
        parsed = urlsplit(base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Invalid API base URL')
        self.endpoint = base.rstrip('/')
        if not self.endpoint.endswith('/chat/completions'):
            self.endpoint += '/chat/completions'
        key = get('LLM_API_KEY', 'OPENAI_API_KEY')
        self.headers = {'Authorization': 'Bearer ' + key} if key else {}
        self.options = {'max_tokens': int(get('OPENAI_MAX_TOKENS', default='4096'))}
        if not 1 <= self.options['max_tokens'] <= 65536:
            raise ValueError('Invalid OPENAI_MAX_TOKENS')
        for env, field in [('OPENAI_REASONING_EFFORT', 'reasoning_effort'), ('OPENAI_THINKING', 'thinking')]:
            if get(env): self.options[field] = {'type': get(env)} if field == 'thinking' else get(env)
        self.http = httpx.Client(timeout=timeout, follow_redirects=False)

    def call(self, messages):
        # Keep text/tool history, but only the last three image-bearing messages.
        image_messages = [i for i,m in enumerate(messages) if isinstance(m.get('content'),list)
                          and any(c.get('type')=='image_url' for c in m['content'])]
        keep = set(image_messages[-3:])
        wire = []
        for i,m in enumerate(messages):
            if i in image_messages and i not in keep:
                m = {**m, 'content': [c for c in m['content'] if c.get('type')!='image_url']}
            wire.append(m)
        try:
            response = self.http.post(self.endpoint, headers=self.headers,
                json={'model': self.model, 'messages': wire, 'tools': getattr(self,'tools',TOOLS),
                      'tool_choice': 'auto', 'stream': False, **self.options})
        except httpx.HTTPError as exc:
            raise ModelServiceError(f'Model transport failed: {type(exc).__name__}',
                                    retryable=isinstance(exc, httpx.TransportError)) from None
        if response.status_code != 200:
            raise ModelServiceError(f'Model HTTP {response.status_code}; response body omitted',
                                    retryable=response.status_code in (408, 429, 500, 502, 503, 504))
        data = response.json()
        choices = data.get('choices', [])
        if len(choices) != 1 or choices[0].get('finish_reason') not in ('tool_calls', 'stop'):
            raise RuntimeError('Incomplete or rejected model response; no actions executed')
        raw = choices[0]['message']
        calls = raw.get('tool_calls', [])
        if raw.get('refusal') or not isinstance(calls, list):
            raise RuntimeError('Model refusal or malformed tool calls')
        if any(c.get('type') != 'function' or not isinstance(c.get('id'), str) or not c['id'] or
               not isinstance(c.get('function'), dict) or not isinstance(c['function'].get('name'), str) or
               not isinstance(c['function'].get('arguments'), str) for c in calls):
            raise RuntimeError('Malformed function call; no actions executed')
        if len({c['id'] for c in calls}) != len(calls):
            raise RuntimeError('Duplicate tool call ids')
        message = {k: raw[k] for k in ('content', 'tool_calls', 'reasoning_content', 'reasoning', 'reasoning_details') if k in raw}
        message['role'] = 'assistant'
        return message, data.get('usage', {})

    def close(self): self.http.close()


def image_block(image):
    buf = io.BytesIO(); image.save(buf, format='JPEG', quality=90)
    return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()}}


def inspect_crop(image, region, local_coordinates=False):
    """Magnify RGB only; rulers use source or displayed crop pixels."""
    x, y, w, h = region
    scale = max(1, min(6, 600 // max(w, h)))
    margin = 32
    crop = image.crop((x, y, x+w, y+h)).resize((w*scale, h*scale), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (crop.width+margin, crop.height+margin), '#202020')
    canvas.paste(crop, (margin, margin))
    draw = ImageDraw.Draw(canvas)
    spacing = max(5, int(np.ceil(36 / scale / 5)) * 5)
    for origin, length, horizontal in ((x, w, True), (y, h, False)):
        for value in range(origin, origin+length):
            if value % spacing: continue
            p = margin + (value-origin)*scale + scale//2
            if horizontal:
                draw.text((p-9, 6), str(p if local_coordinates else value), fill='white')
                draw.line((p, 24, p, margin-1), fill='cyan')
            else:
                draw.text((1, p-5), str(p if local_coordinates else value), fill='white')
                draw.line((25, p, margin-1, p), fill='cyan')
    return canvas, {'source_region_xywh': region, 'scale': scale,
                    'display_origin_xy': [margin, margin],
                    'coordinate_rule': ('Rulers are displayed crop pixels, including the 32px margin. Use locate_crop with crop_id and these x/y values.' if local_coordinates else 'Ruler labels are ORIGINAL image pixels: x along top, y along left. Use original coordinates for locate_pixel, not displayed-image pixels.')}


def crop_source_pixel(mapping, x, y):
    """Convert displayed crop pixels; reject rulers and outside pixels."""
    if type(x) is not int or type(y) is not int:
        raise ValueError('Crop coordinates must be integers')
    sx, sy, w, h = mapping['source_region_xywh']
    ox, oy = mapping['display_origin_xy']; scale = mapping['scale']
    if not (ox <= x < ox+w*scale and oy <= y < oy+h*scale):
        raise ValueError('Select inside crop content, not the ruler margin')
    return sx+(x-ox)//scale, sy+(y-oy)//scale


class StalkFitError(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def localization_feedback(image, x, y):
    """Mark the selected source pixel without hiding its RGB value."""
    x0,y0=max(0,x-40),max(0,y-40)
    x1,y1=min(image.width,x+41),min(image.height,y+41)
    crop,mapping=inspect_crop(image,[x0,y0,x1-x0,y1-y0])
    scale=mapping['scale']; ox,oy=mapping['display_origin_xy']
    cx=ox+(x-x0)*scale+scale//2; cy=oy+(y-y0)*scale+scale//2
    draw=ImageDraw.Draw(crop)
    for a,b,c,d in ((cx-15,cy,cx-4,cy),(cx+4,cy,cx+15,cy),
                    (cx,cy-15,cx,cy-4),(cx,cy+4,cx,cy+15)):
        draw.line((a,b,c,d),fill='cyan',width=2)
    return crop,mapping


def robot_state(e):
    return {'joints_rad': e.state().tolist(), 'tcp_world_m': e.data.site_xpos[e.tcp].tolist(),
            'base_x_y_yaw': e.base_pose().tolist()}


def jaw_contact_state(e):
    """Simulated tactile switches; no fruit/stalk IDs or object positions used."""
    robot_root=e.model.body_rootid[e.grip_body]
    touched=set()
    for c in e.data.contact:
        if c.dist>0:continue
        a,b=e.model.geom_bodyid[c.geom]
        for finger,other in ((a,b),(b,a)):
            if finger in (e.grip_body,e.jaw_body) and e.model.body_rootid[other]!=robot_root:
                touched.add(int(finger))
    return {'fixed':e.grip_body in touched,'moving':e.jaw_body in touched,
            'both':e.grip_body in touched and e.jaw_body in touched,
            'note':'Simulated external-contact switches; not proof of stalk grasp or detachment.'}


def depth_point(depth, calibration, x, y):
    h,w = depth.shape
    if type(x) is not int or type(y) is not int or not 0 <= x < w or not 0 <= y < h:
        raise ValueError('Pixel outside image')
    z = float(depth[y,x])
    if not math.isfinite(z) or not .01 < z < 2:
        raise ValueError('Depth invalid or outside 2m working range')
    focal = h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    local = np.array([(x+.5-w/2)*z/focal, -(y+.5-h/2)*z/focal, -z])
    return np.asarray(calibration['position']) + np.asarray(calibration['rotation']) @ local


def verify_projected_point(point, depth, calibration):
    """Check measured RGB-D point against another view, without scene object IDs."""
    h,w=depth.shape
    local=np.asarray(calibration['rotation']).T@(point-np.asarray(calibration['position']))
    if not np.isfinite(local).all() or local[2]>=-.01:
        return {'consistent':False,'reason':'behind_camera'}
    focal=h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    uv=np.array([w/2-.5+focal*local[0]/-local[2],h/2-.5-focal*local[1]/-local[2]])
    if not (0<=uv[0]<w and 0<=uv[1]<h):
        return {'consistent':False,'reason':'outside_image'}
    x,y=np.rint(uv).astype(int)
    samples=[]
    # One-pixel tolerance for rasterization of a 3mm stalk; no broad search.
    for yy in range(max(0,y-1),min(h,y+2)):
        for xx in range(max(0,x-1),min(w,x+2)):
            try:p=depth_point(depth,calibration,xx,yy)
            except ValueError:continue
            samples.append((float(np.linalg.norm(p-point)),xx,yy))
    if not samples:return {'consistent':False,'reason':'invalid_depth','projected_pixel':uv.tolist()}
    error,xx,yy=min(samples)
    return {'consistent':error<=.008,'projected_pixel':uv.tolist(),
            'sample_pixel':[xx,yy],'surface_error_m':error,
            'note':'Surface consistency only; not semantic recognition of a stalk.'}


def stalk_grasp_point(point,depth,calibration,rgb):
    """Single-fixture geometry prior: 36mm upright stalk, grasp 24mm below tip.

    Estimate the visible green stalk extent locally from RGB-D, never object sites.
    """
    h,w=depth.shape;v,u=np.indices(depth.shape)
    f=h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    local=np.stack(((u+.5-w/2)*depth/f,-(v+.5-h/2)*depth/f,-depth),axis=-1)
    xyz=local@np.asarray(calibration['rotation']).T+calibration['position']
    rgb=np.asarray(rgb,dtype=float)
    mask=(np.linalg.norm(xyz[:,:,:2]-point[:2],axis=-1)<.004)
    mask&=(np.abs(xyz[:,:,2]-point[2])<.04)&np.isfinite(depth)&(depth>.01)&(depth<2)
    mask&=(rgb[:,:,1]>1.15*rgb[:,:,0])&(rgb[:,:,1]>1.15*rgb[:,:,2])
    cloud=xyz[mask]
    low,high=np.quantile(cloud[:,2],[.05,.95]) if len(cloud) else (0.,0.)
    diagnostics={'samples':len(cloud), 'visible_height_m':float(high-low) if len(cloud) else None,
                 'minimum_samples':20, 'allowed_height_m':[.025,.045]}
    if len(cloud)<20:raise StalkFitError('Too few visible green stalk samples; inspect close camera before approach',diagnostics)
    if not .025<=high-low<=.045:
        raise StalkFitError('Visible stalk extent ambiguous; cannot infer safe grasp height',diagnostics)
    z=high-.024
    band=cloud[np.abs(cloud[:,2]-z)<.002]
    if len(band)<3:raise StalkFitError('Grasp band occluded; choose a clearer view',diagnostics)
    target=np.r_[np.median(band[:,:2],axis=0),z]
    if np.linalg.norm(target-point)>.025:raise StalkFitError('Stalk fit too far from selected pixel',diagnostics)
    return target,{'visible_height_m':float(high-low),'samples':len(cloud),
                   'prior':'single fixture: upright 36mm stalk; grasp 24mm below visible tip'}


def tilted_stalk_grasp_point(point, depth, calibration, rgb):
    """Fit a near-upright thin RGB-D segment; never read simulator object poses."""
    h,w=depth.shape;v,u=np.indices(depth.shape)
    f=h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    local=np.stack(((u+.5-w/2)*depth/f,-(v+.5-h/2)*depth/f,-depth),axis=-1)
    xyz=local@np.asarray(calibration['rotation']).T+calibration['position']
    color=np.asarray(rgb,dtype=float)
    mask=np.isfinite(depth)&(depth>.01)&(depth<2)
    mask&=(np.linalg.norm(xyz-point,axis=-1)<.043)
    mask&=(color[:,:,1]>1.15*color[:,:,0])&(color[:,:,1]>1.15*color[:,:,2])
    cloud=xyz[mask];delta=cloud-point
    diagnostics={'samples':len(cloud),'visible_height_m':None,'visible_length_m':None,
                 'minimum_samples':20,'allowed_length_m':[.025,.045], 'max_tilt_deg':20}
    if len(cloud)<20:raise StalkFitError('Too few visible green stalk samples; inspect another view',diagnostics)
    # Search only a narrow cone around vertical. Reject long/short segments instead
    # of changing the length thresholds to manufacture a valid grasp.
    axes=[np.array([0.,0.,1.])]
    for degrees in (5,10,15,20):
        theta=np.deg2rad(degrees)
        axes.extend(np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
                    for phi in np.linspace(0,2*np.pi,24,endpoint=False))
    best=None
    for axis in axes:
        along=delta@axis
        radial=np.linalg.norm(delta-along[:,None]*axis,axis=1)
        inliers=radial<.003
        score=int(inliers.sum())
        if best is None or score>best[0]:best=(score,axis,inliers)
    count,axis,inliers=best
    diagnostics.update(samples=count,axis_world=axis.tolist())
    # Nearby green pixels may all be leaves outside every candidate thin tube.
    # Reject before quantiles: an empty segment is a normal localization failure.
    if count<20:raise StalkFitError('Too few thin-segment samples; inspect and reselect a clearer stalk',diagnostics)
    segment=cloud[inliers];along=segment@axis
    low,high=np.quantile(along,[.05,.95])
    diagnostics.update(visible_length_m=float(high-low),
                       visible_height_m=float(np.ptp(np.quantile(segment[:,2],[.05,.95]))))
    if not .025<=high-low<=.045:
        raise StalkFitError('Visible stalk length ambiguous; choose a clearer segment',diagnostics)
    # Reject disconnected leaf/branch fragments masquerading as a whole stalk.
    interior=np.sort(along[(along>=low)&(along<=high)])
    if np.max(np.diff(interior),initial=0)>.005:
        raise StalkFitError('Stalk segment has an occluded gap larger than 5mm',diagnostics)
    level=high-.024;band=segment[np.abs(along-level)<.002]
    if len(band)<3:raise StalkFitError('Grasp band occluded; choose a clearer view',diagnostics)
    target=np.median(band,axis=0);target+=(level-target@axis)*axis
    if np.linalg.norm(target-point)>.025:
        raise StalkFitError('Stalk fit too far from selected pixel',diagnostics)
    diagnostics['prior']='36mm thin stalk, tilt <=20deg; grasp 24mm along axis below visible tip'
    return target,diagnostics


def command_tcp(env,q):
    """FK of a robot command; no plant/target truth is consulted."""
    import mujoco
    data=env.ikdata
    data.qpos[:]=env.data.qpos
    data.qpos[env.qa]=q
    mujoco.mj_kinematics(env.model,data)
    return data.site_xpos[env.tcp].copy()


def check_observation(state, current, age, max_wait):
    """Contact may change finger opening without moving the arm/camera pose."""
    arm = float(np.max(np.abs(np.array(state['joints_rad'][:5])-current['joints_rad'][:5])))
    tcp = float(np.linalg.norm(np.array(state['tcp_world_m'])-current['tcp_world_m']))
    base = np.array(state['base_x_y_yaw'])-current['base_x_y_yaw']
    yaw = abs(float(np.arctan2(np.sin(base[2]),np.cos(base[2]))))
    if age > max_wait or arm > .04 or tcp > .01 or np.linalg.norm(base[:2]) > .01 or yaw > .01:
        raise ValueError(f'Observation stale: age={age:.1f}s, arm_drift={arm:.4f}rad, tcp_drift={tcp:.4f}m, base_drift={np.linalg.norm(base[:2]):.4f}m, yaw_drift={yaw:.4f}rad; replan')


class CallBudgetExhausted(RuntimeError):
    pass


class MotionSafetyStop(RuntimeError):
    """Expected motion protection stop: save without another settle or model call."""


class Session:
    def __init__(self, env, recorder, client, max_calls=32, max_wait=90, viewer=None, pregrasp_park_plus40=False, target_aware_parking=False, fixed_base=False, preclose_visual_check=False, centered_grasp=False, visual_proposals=False):
        self.e, self.rec, self.client = env, recorder, client
        self.max_calls, self.max_wait, self.viewer = max_calls, max_wait, viewer
        self.frames = {}; self.calls = 0; self.finished = False
        self.require_delivery_before_finish=bool(centered_grasp and visual_proposals)
        self.messages = [{'role': 'system', 'content': project_source('vista_prompt.txt').read_text()}]
        self.events = self.rec.out/'vista-events.jsonl'
        self.wait_ticks = 0
        self.receipts = []
        self.depth_frames = {}
        self.crops = {}
        self.approach_pitch = 0.
        self.locked_target = None
        self.maturity_evidence = None
        self.target_camera = None
        self.target_confirmed = False
        self.target_stage = None
        self.motion_stop_trace = None
        self.pull_reference_frame=None
        self.pull_reference_state=None
        self.pull_completed=None
        self.detachment_confirmation=None
        self.detachment_reviews=[]
        self.delivery_completed=False
        self.grasp_attempted = False
        self.centered=None
        if centered_grasp:
            if not env.meta.get('tilted_stalk_prior') or env.meta.get('basket_mount')!='chassis':
                raise ValueError('Centered grasp requires full greenhouse with chassis basket')
            from centered_grasp import CenteredGrasp
            self.centered=CenteredGrasp(env,self.tick)
            env.meta['centered_grasp']=True
        if preclose_visual_check and not env.meta.get('tilted_stalk_prior'):
            raise ValueError('Preclose visual check requires the full diverse greenhouse')
        self.preclose_visual_check = preclose_visual_check
        self.preclose_recoveries = 0
        self.preclose_recovery_failed = False
        self.visual_rejected_targets = []
        self.search_steps = 0
        self.search_history = []
        self.visual_search_hints = []
        self.target_reachability = None
        if fixed_base and (target_aware_parking or pregrasp_park_plus40):
            raise ValueError('Fixed harvest base cannot mix with parking experiments')
        self.fixed_base = fixed_base
        self.policy_cameras=tuple(env.meta.get('policy_cameras',('top','wrist','harvest_close') if env.meta.get('harvest_mechanism') else ('top','wrist')))
        self.local_targets = LocalTargets() if env.meta.get('tilted_stalk_prior') and not pregrasp_park_plus40 else None
        if self.local_targets is not None:
            self.local_targets.allow_uncertain_deferral=self.require_delivery_before_finish
        self.visual_proposals=visual_proposals
        self.proposal_frames={}
        self.stalk_proposals={}
        self.perception_recoveries=[]
        self.unlisted_search_evidence=None
        if visual_proposals and self.local_targets is None:
            raise ValueError('Visual proposals require local-first full greenhouse mode')
        if client is not None:
            client.tools=[json.loads(json.dumps(t)) for t in TOOLS if (not fixed_base or t['function']['name'] not in ('search_row','refine_parking','search_after_unlisted_review'))
                          and (centered_grasp or t['function']['name'] not in ('confirm_detachment','adjust_observation'))
                          and (visual_proposals or t['function']['name'] not in ('locate_stalk_proposal','correct_survey','review_proposals','search_after_unlisted_review','inspect_unlisted'))
                          and (self.local_targets is not None or t['function']['name'] not in ('survey_candidates','correct_candidate_body'))]
            for tool in client.tools:
                props=tool['function']['parameters'].get('properties',{})
                if 'camera' in props:props['camera']['enum']=list(self.policy_cameras)
                if visual_proposals and tool['function']['name'] in ('survey_candidates','correct_survey','review_proposals'):
                    tool['function']['parameters']=obj({'frame_id':STR,'coverage':STR,
                        'missing_visible_ripe':{'type':'boolean'},
                        'candidates':{'type':'array','maxItems':16,'items':obj({'proposal_id':STR,
                            'maturity':{'enum':['ripe','turning','green','uncertain','not_fruit']},
                            'visibility':{'enum':['clear','partial','occluded']},'reason':STR})}})
                    if tool['function']['name'] in ('correct_survey','review_proposals'):
                        tool['function']['parameters']['properties']['correction_reason']=STR
                        tool['function']['parameters']['required'].append('correction_reason')
                    if tool['function']['name']=='review_proposals':
                        tool['function']['parameters']['properties']['duplicate_links']={
                            'type':'array','maxItems':8,'items':obj({'fragment_id':STR,'parent_id':STR,'reason':STR})}
        if target_aware_parking and (pregrasp_park_plus40 or not env.meta.get('tilted_stalk_prior')):
            raise ValueError('Target-aware parking requires full greenhouse and cannot mix with fixed parking')
        self.target_aware_parking = target_aware_parking
        self.parking_anchor = None
        self.parking_history = []
        self.station_attempts = 0
        self.station_failures = 0
        if pregrasp_park_plus40 and not env.meta.get('tilted_stalk_prior'):
            raise ValueError('Fixed parking trial requires the full diverse greenhouse')
        self.parking_trial={'enabled':bool(pregrasp_park_plus40),
                            'status':'pending' if pregrasp_park_plus40 else 'disabled',
                            'requested_delta_y_m':.04,'grasp_height_offset_m':0.}
        if env.meta.get('harvest_mechanism'):
            self.messages[0]['content'] = project_source('peduncle_mobile_prompt.txt' if env.meta.get('basket_mount')=='chassis' else 'peduncle_prompt.txt').read_text()
            if not env.meta.get('single_fruit_fixture'):
                self.messages[0]['content']=self.messages[0]['content'].replace('单果草莓采摘实验','完整大棚草莓采摘任务')
                self.messages[0]['content']+='\n当前恢复完整大棚：多颗草莓、叶片和枝梗均存在。每回合只采一颗可达的成熟草莓并送回固定筐。近景机位随车移动，选择其清晰可见的果柄；拟合被遮挡或含多个枝梗时应拒绝，不能沿用单果实验成功率。\n'

            if env.meta.get('tilted_stalk_prior'):
                self.messages[0]['content'] += '\n完整大棚拉取和运输前会保持当前姿态0.5秒，检查双指外部接触持续存在；检查失败则不开始拉取/运输。单次双指接触为真不等于稳定夹持，也不能证明夹到同一果柄。\n'
                self.messages[0]['content'] += '\n定位后先读取 reachability：不可达时不要 approach_target；先处理当前停车位置的其他候选；全部无合适目标后，才按 next_action 决定是否微调停车。程序会先收臂、移动4或8厘米、停车，旧目标失效，必须重新拍摄选点。微调与沿行搜索共用4次移动预算。reachable 只表示运动学可达，不表示无碰撞；接近受阻时仍应撤回并重新观察，不能盲目重试。\n'
                self.messages[0]['content']=self.messages[0]['content'].replace('竖直果柄段','倾斜细长果柄段').replace('可见顶端下24mm','沿拟合轴从可见顶端下24mm').replace('单果夹持高度修正','果柄夹持高度修正').replace('采摘停车y=0','采摘停车y为本回合初始底盘y')
                self.messages[0]['content'] += '\n本场景使用多样化布局：所有果实成熟度随机，没有固定成熟目标，当前视野可能无可达成熟果；先比较原地近处成熟候选；全部处理后才允许search_row沿行搜索，预算用尽则finish如实报告。果柄可倾斜，定位使用RGB-D细长轴拟合，沿可见顶端向下24mm，不能假设竖直。近景相机安装在底盘上，随车观察工作区，不自动朝向某颗果实。采摘位置以本回合初始底盘状态为准。遮挡等级只是生成参数，不保证可见或可摘。\n'
        if pregrasp_park_plus40:
            self.messages[0]['content'] += '\n本回合验证固定停车方案：首次成熟果柄确认后的pregrasp请求会先执行一次收臂、沿世界+y移动4厘米并停车，不改变夹持高度规则。这次请求不执行预抓取。必须读取下一帧重新观察定位，再请求pregrasp。每回合只实施一次；refine_parking禁用以免混入其他停车方案。仍可search_row搜索，移动共享原预算。移车后尽量重新选择同一可见果柄；无法可靠对应时如实报告，不猜测旧坐标。\n'
        if target_aware_parking:
            self.messages[0]['content'] += '\n本回合启用目标停车评分，替代固定4/8厘米微调：依据测得的成熟果柄位置，程序按1厘米网格比较前后10厘米内的停车点，兼顾可达性、关节余量、接近路径弧度和移车距离。原地可达时保持当前位置；其他本地候选未处理完也不微调停车。遵循reachability.next_action；pregrasp也会自动检查，需要时只执行收臂移车，不执行接近。移动后旧目标与图像失效，必须重新观察、确认成熟度和果柄位置。推荐是局部运动学评分，不保证视野、接触稳定或采摘成功，不调整夹持高度。接触允许，仍执行跟踪和稳定夹持检查。\n'
        if fixed_base:
            self.messages[0]['content']='\n'.join(line for line in self.messages[0]['content'].splitlines()
                if not any(word in line for word in ('search_row','refine_parking','停车点预算')))
            self.messages[0]['content']+='\n固定采摘模式：采摘期间底盘保持初始位置，不沿行搜索、不优化停车、不自由drive。车前多颗草莓成熟度随机，无固定成熟编号；仅选充分变红、可见且可达的果柄。某颗不可用时可换当前视野内另一颗。最多12次查图/定位，每颗两次定位失败后换另一颗，clear_target不重置预算。没有合适目标或预算用尽就finish，如实报告失败，不摘未熟果。接触允许，但必须检查稳定夹持。成功摘下后仍用deliver_to_basket执行原送筐和返回流程。\n'

        self.messages[0]['content']+=('\n本回合已启用 --preclose-visual-check 闭爪前视觉复核与有界撤回。\n'
                                      if preclose_visual_check else '\n本回合未启用实验性的 --preclose-visual-check。\n')
        if centered_grasp:
            self.messages[0]['content']=self.messages[0]['content'].replace('张爪到外侧55mm','以标定开度移到程序选择的预接近位置')
            self.messages[0]['content']=self.messages[0]['content'].replace('seconds=2.5','seconds=3')
            self.messages[0]['content']+='\n本回合启用实验性的果柄居中控制：pregrasp/grasp 会按视觉果柄轴转腕并以标定指面间隙为中心接近，预接近开度0.2属于正常状态。完成grasp后，用gripper(opening=-0.15, seconds=3)闭爪；程序在3秒内补偿闭合中心移动。不要用move_tcp手动修正夹持高度。拉取和送筐保持抓取腕角，送筐后自动恢复观察腕角。仍须看实际图像确认夹持和摘取；执行完成并不证明成功。\n'
            self.messages[0]['content']+='\n预接近距离按55、45、35mm依次检查运动学可达及机器人自身碰撞，使用第一个通过的候选；目标和夹持高度不变。此检查仅使用机器人几何，不保证与枝叶/果实无接触；实际跟踪保护仍有效。接近受阻时反向退出已走过的指令，退出失败会自动停止保存，不再等待模型或继续动作。\n'
            self.messages[0]['content']+='\n居中模式的 bend_pull 现为分段动作：先保持抓取俯仰向外退出1.4cm，再边继续退出边转腕，最终总位移3.5cm、俯仰-37度；seconds=3为最短时长，程序先估算限速路径所需时间，可延长但最多6秒。独立调用play，仅包含这一项；每回合最多一次，禁止重复拉或另用move_tcp替代。参数仍为dx=-0.035,dy=0,dz=0,pitch_deg=-37,seconds=3。动作executed和双指接触都不证明摘断。下一轮比较pull_visual_reference的拉前两路图与当前两路图，调用confirm_detachment：basis_frame_id为当前帧，reference_frame_id为拉前帧；逐相机报告held_fruit_visible、separation（visible_gap/connected/occluded）及图像理由。必须看见果柄末端与植株之间的分离间隙，不能仅凭果实随爪移动、背景变化或工具成功。两路都需看见果实仍夹持，至少一路看见分离且另一路不矛盾，才允许送筐。看不清就如实报告occluded并inspect或finish，不可为继续流程编造间隙。闭爪后至少预留4次调用（拉取、确认、送筐、结束）。\n'
        if self.local_targets is not None:
            self.messages[0]['content']+='\n果柄拟合使用你实际选点相机的RGB-D。近景拟合因可见长度或遮挡失败后，优先查看同一active候选的 wrist_active_candidate_crop，在腕部视角选择该果实的短果柄；不要继续反复猜近景背景坐标。腕部圆圈是原果肉测量点的投影，不证明该表面在腕部无遮挡；先看图确认对应关系。两路拟合共享原候选两次失败及本站12次感知预算，不能用换视角重置。\n'
            self.messages[0]['content']+='\n原地近果优先：先调用 survey_candidates，一次列出 harvest_close 当前整幅画面内工作区可见果实，最多8颗。x/y选果肉主体内部像素，不选萼片、果柄或边缘；逐颗报告 maturity、visibility 和图像理由，coverage说明扫描范围及被遮挡/不可判断区域。确无候选才给空列表并说明证据，不可只列最清晰或最想抓的一颗而遗漏近处成熟果。程序用RGB-D表面到当前底盘的水平距离排序；这是可见候选中的近远，不是全棚最优。先 inspect 并定位 local_selection.active 所指果实的短柄，不能越过它直接抓远处果。每颗最多两次定位失败；失败后原地换下一颗，不触发移车。成熟度拒绝或已测得不可达也换下一颗。全部候选处理完才允许搜索；感知调用预算用完但还有候选时结束，不能据此开车。每站最多12次查图/选点，survey只允许一次，不能重复提交清零失败计数。已选目标原地可达就保持停车，禁止为提高停车评分而移动。未知视野或两次错误像素都不等于当前位置不可采摘。\n'
            self.messages[0]['content']+='\n候选对应与预算：每轮 active_candidate_crop 是当前 active 候选的最新近景裁剪，圆圈只标记先前视觉选定的果肉表面，不是果柄点。先确认圆圈确实在成熟果体内，然后在其上方短果柄上选点，可直接使用该 crop_id 调用 locate_crop，无需先 inspect。裁剪只放大已有近景相机，不增加相机。只按回执的 active.candidate_key 和 failures 判断是否换果，不能凭自己的文字宣布换果；过期、选中背景或不属于 active 的点不会计为该果实失败，但请求仍消耗预算。12次是查图/定位预算，接近、闭爪、拉取、投放不消耗它；它不是32次总模型预算。即使剩2次查图，仍可定位后使用剩余总调用完成采摘。预算充足且仍有合法复核或观察恢复时，首次finish会返回一次recovery_available提醒；应优先执行next_action，恢复确实不适用才再次finish。必须如实说明未用完的预算，不能把未执行的定位写成失败。\n'
            self.messages[0]['content']+='\nsurvey 中声称成熟的果肉像素必须有红色图像证据；点到背景或枝叶会标为 unverified_measurement。它不代表果实不可达或当前位置无成熟果，不得据此移车，也不能越过更近的未核实候选去抓远果。body_correction_feedback 显示原选点附近的原图坐标尺，十字是错误选点；用 correct_candidate_body 在原选点32像素半径内重新选同一果实的红色果肉内部，每颗最多2次，也消耗12次感知预算。不能改成熟度、删除候选或借纠错换远果；纠错失败仍禁止移车。若局部没有可确认果肉或 next_action 要求结束就如实 finish，不要重复 survey。此保护仍不能证明扫描覆盖了所有果实。\n'

        if visual_proposals:
            self.messages[0]['content']+='\n候选框包含红点周围相连的橙红果体，编号不等于成熟。看整个框及原图：橙色主体夹杂少量红斑仍是 turning；不要只盯青色采样点或红色斑点判断 ripe。橙色上下文只改善候选框和深度测量，抓取前的红色成熟度校验仍独立执行。\n'
            self.messages[0]['content']+='\n果柄编号确认：active两路放大图的黄色圆点及C1/W1编号是RGB-D提出的细长绿色片段，青色圈仍为果肉。有编号时优先用locate_stalk_proposal提交frame_id、candidate_key、proposal_id、confirmed_short_stalk=true及reason，不再估计细果柄像素坐标。必须看图确认黄色点位于当前成熟果连接的短果柄，而非叶片、长枝或邻果；拟合通过不证明身份。C来自近景，W来自腕部；腕图的世界上方不一定是屏幕上方。近景无编号时检查腕部。编号为空时先看 station_search.next_action：只有程序已在同一新帧检查两路、均无可用拟合，才允许 adjust_observation 保持底盘、开爪后撤抬臂再看同一候选；该动作不证明果实不可采，也不授权跳过、闭爪或移车。否则可inspect/手动定位或finish。仍共享12次感知及每候选2次真实失败预算；选择后重新拟合并检查成熟度和另一视角，旧帧/旧候选编号无效。\n'
            self.messages[0]['content']=self.messages[0]['content'].replace(
                '一次列出 harvest_close 当前整幅画面内工作区可见果实，最多8颗。x/y选果肉主体内部像素，不选萼片、果柄或边缘；',
                '逐一审核 harvest_close 编号图中的全部 proposal_id，最多16个，不提交x/y；')
            self.messages[0]['content']+='\n编号候选模式：P1等仅为当前帧RGB红色区域的标签，绝非场景果实ID，也不证明成熟或果实身份。survey_candidates必须逐一审核给出的全部编号，不可遗漏；看整果颜色和形状给maturity/visibility/reason，非果实用not_fruit，看不清用uncertain。程序已选择区域内部RGB-D像素并按距离排序，无需猜整图坐标。检查原始整图是否有漏标的近处成熟果，missing_visible_ripe如实报告；遗漏、截断、近处不确定候选会阻止越过它抓远果或移车。编号仅对当前frame_id有效。登记后先看active两路裁剪的黄色果柄编号，优先locate_stalk_proposal；只有没有可确认编号时才手动locate_crop（裁剪坐标）。红色小区域/转色果/遮挡可能漏检，候选为空不证明没有成熟果。\n'

        if visual_proposals:
            self.messages[0]['content']+='\n覆盖纠错例外：若 station_search.next_action 指向 correct_survey，且只有 missing_visible_ripe 误报阻断，可在最新完整编号图上重新审核全部编号，调用 correct_survey 并提供 correction_reason，解释原来所指的漏标果及当前证据。不得直接把 true 改 false、不核对图像；仍有漏标或不能确定时保持 true 或 finish。此工具仅改覆盖声明，不能删改候选成熟度、重置失败或省略任何编号；最多两次且计入12次感知预算。普通 survey_candidates 仍不能重复登记。\n'
        if centered_grasp and self.local_targets is not None:
            self.messages[0]['content']+='\n观察移车例外：若 station_search.next_action 明确为 observation-only viewpoint change，说明两次复核后只剩程序以新RGB-D确认的窄小切边碎片，且没有待处理的成熟目标。可search_row短移换视角；这不表示碎片不是果或已经未熟，程序会保留未确认记录。不能抓取碎片，停车后必须重新survey和核验。非切边不确定候选、深度未确认、漏果覆盖阻断仍不能使用该例外。\n'
            self.messages[0]['content']+='\n若 uncertain 候选已实际完成 adjust_observation，随后新图 review_proposals 确认成熟，但更新帧两路仍无可用果柄拟合，程序可在 station_search.next_action 明确允许 search_row 结束本停车点观察尝试。此时无需乱点凑定位失败次数；仍须处理其余候选并遵守移动预算，停车后重新审核。只有程序的 next_action 授权此分支，模型不能自行把遮挡当作无果。\n'
            self.messages[0]['content']+='\n跨视角不一致不等于不可达：看 cross_view_checks 及最新 active 候选。跨视角失败，或程序在同一新帧检查两路均无可用果柄拟合时，仅当 next_action 允许，调用 adjust_observation(basis_frame_id,candidate_key,reason) 保持底盘、开爪小幅后撤抬臂以改变遮挡；每颗最多一次、每站最多两次，消耗感知预算且不重置失败次数。移动后所有旧果柄坐标和crop编号作废，必须下一帧重新选该 active 果柄并完成双视角核验。模型不要把其他果的像素提交给当前active；只有返回 candidate_transition 或明确清除已验证不适合的目标后才处理下一颗。一次定位失败后还有一次机会，不要谎称其他候选已验证不可采。旧图只能对照，不可用于新定位。\n'

        if visual_proposals:
            self.messages[0]['content']+='\n拟合与身份分开：有C/W编号仅表示绿色片段符合几何，不强迫你认作果柄。腕部画面会旋转，短果柄可以在果实的屏幕侧方，不能仅因不是屏幕上方就否认。若两路当前编号全部无法确认属于同一active果实的短柄，按station_search调用adjust_observation，附unconfirmed_stalk_proposals=[{proposal_id,reason},...]逐一说明所有编号的不确定证据。保持原候选，小幅开爪换姿态后重新定位；不算果实定位失败，不允许跳果或移车，仍每颗一次、每站两次。不要反复猜遮挡物上的像素来凑失败，也不要在可恢复且预算充足时直接结束。\n'
        if env.meta.get('initial_arm_pose')=='search_stow':
            self.messages[0]['content']+='\n本回合启动与沿行搜索停车后均保持收臂观察姿态，避免机械臂挡住近景果实。先比较两路图并定位确认目标，再用approach_target伸向果柄；不用为了扫描先伸回旧观察姿态。必要的adjust_observation仍按证据使用。送筐后返回本回合初始收臂姿态。\n'
        if visual_proposals:
            self.messages[0]['content']+='\n若 next_action 允许 review_proposals，用最新完整编号图重新审核同一批区域，解释旧错误及新证据。橙色或橙红混色果是 turning，不是充分成熟；不能把检测器红色小点当整果成熟证据。每站2次常规复核；实际完成新的观察姿态调整后可额外复核1次（每站总计最多3次），共用12次感知预算；自动重测当前RGB-D，不允许删编号、改变覆盖声明或重置失败次数。可见性可依据当前图像更新；变得遮挡不代表候选消失，也不能凭此跳过或移车。成熟转未熟需局部橙绿颜色检查通过；逐颗处理，有证据的纠正会保留，证据不足的改记为uncertain并在deferred_corrections说明，不会因一颗而撤销整批。优先按next_action继续adjust_observation或复核，不能把未确认写成不存在成熟果。覆盖误报必须单独correct_survey，不能以检测器没编号推断没有漏果。\n'
        if visual_proposals:
            self.messages[0]['content']+='\n未编号红果距离核查：当 next_action 指向 search_after_unlisted_review，保持 missing_visible_ripe=true，报告原始 harvest_close 480x360 图中所有导致本次 missing_visible_ripe 声明的果体内部像素和视觉理由（不是2倍编号图坐标），all_missing_claims_reported=true。工具先检查最新RGB-D，只有当前没有近处或深度未知候选、所指红果均可靠测得在56cm候选范围之外且留有2cm余量，才执行一次有界沿行观察移动；否则拒绝。此操作不确认成熟、不保证到达所指果、不许可抓取；停车后必须重新survey。不要为了移动而把真实红果声明改成false。\n'
            self.messages[0]['content']+='\n漏标声明先核对坐标：next_action提示inspect_unlisted时，提交原始harvest_close 480x360中的果体像素。即使本地队列非空也可只读核查，返回标记图、RGB、深度与已有编号框对应，不移动、不解除阻断。若十字落在绿色叶片/背景，不能反复把同一坐标说成红果；重查原图，确属误报才用correct_survey说明并纠正。错误像素不证明整图无漏果；仍有真实遗漏或不确定候选就保留阻断。每站最多2次，共享12次感知预算。\n'
        if not fixed_base and self.local_targets is not None:
            self.messages[0]['content']+='\n搜索方向必须结合row_search_guidance：1是世界+Y，-1是世界-Y，图像左右和腕部上下不能直接作世界方向。已通过视觉成熟度和双路定位、仅当前位置不可达的成熟果仍值得换停车点观察；先比较其current_relative_y_m和toward_measured_target_direction，优先向该侧短移，不能把unreachable here当作全局不可采而背离它。连续只看到转色果或空候选时重新比较方向与completed_searches，不要机械沿同方向用尽4次预算。历史测量仅作搜索上下文，移车后必须重新survey及全部定位核验；普通搜索方向仍由你根据当前图像选择。\n'
        if visual_proposals:
            self.messages[0]['content']+='\n复核回执的 unresolved 表示问题仍未解决，工具执行不代表成熟度通过。body_color_context 是整片局部颜色统计，不能未经图像证据一概解释成种子高光；橙色主体应重新考虑 turning，看不清则 uncertain，反复保留 ripe 不能解除阻断。复核未熟后若队列处理完且 next_action 允许，沿行搜索正是寻找下一颗的合法步骤，不要把“当前没有已确认成熟果”理解为不许搜索。\n'
            self.messages[0]['content']+='\n已完成观察调整、随后实际果柄定位通过成熟度但另一相机测得 outside_image 时，程序会在下一新帧重新投影核查。仅当 next_action 明确允许 resolve the failed observation recovery，才可 search_row 记录本停车点视野不足并检查其他候选；保留真实失败次数，不表示果实不存在或许可单视角抓取。不要重复请求未确认目标的 approach_target，也不要凭模型自述“腕部看不到”跳过核查。\n'
            self.messages[0]['content']+='\n编号列表为空时 survey_candidates 必须 candidates=[]，不能自造编号，也不能先 locate_crop。missing_visible_ripe 指工作区内遗漏或无法排除的成熟果，远处明确超出 RGB-D 工作范围的果不是近果遗漏；检测器为空本身不证明没有遗漏。对于 unverified_semantics，先看 uncertain_candidate_review_crop 的同一点两路放大图并 review_proposals，不要猜果柄坐标。一次复核仍 uncertain 且 next_action 允许时，可 adjust_observation 保持底盘、改变手臂遮挡，再用新图审核同一 P 编号。果体中心红色检查不通过的小片段，若编号区域有一致RGB-D测量且程序允许，也只能用于同一区域观察恢复；颜色区域不是已确认果实，不可改写body_pixel_verified或绕过成熟度核验。身份与成熟度未核实不能接近、闭爪或越过它移车；图中高低不等于世界高度，不能凭屏幕位置断言不可达。\n'
            self.messages[0]['content']+='\n同一果实被编号成主体和小碎片时，碎片仍是果实，不能改成not_fruit或省略编号。若最新两路图像都支持其属于同一颗已确认成熟主体，可在review_proposals附加duplicate_links=[{"fragment_id":"P2","parent_id":"P1","reason":"说明两路图像中的同果证据"}]，仍逐个列出所有candidates并保留碎片uncertain/ripe标签。程序只接受相邻小碎片、20毫米内RGB-D以及腕部同一红色区域/深度吻合；这不是身份真值，贴靠的两颗果或遮挡无法判断时不可合并。主体失败次数和终止状态继承，关联不允许跳过未尝试主体。已有duplicate_links必须保留；不确定就保持uncertain。\n'
            self.messages[0]['content']+='\n若成熟目标跨视角定位已经失败一次，且实际完成adjust_observation后，新一帧两路仍无可用果柄拟合，程序可将这次观察恢复计为第二次本停车点定位失败。仅当next_action明确提示resolve the failed observation recovery时调用search_row，程序先记录候选转换，再检查其他候选和原有移动预算；其他未知或未处理候选仍阻止开车。这不表示果实不存在或在其他位置不可采，移车后必须重新观察。不要为了凑失败次数猜像素或重复旧选点。\n'
        if env.meta.get('mobile_layout'):
            self.messages[0]['content']+='\n移动布局的搜索标定覆盖此前距离说明：search_row 每次沿行18厘米，最多4次，离起点不超过正负72厘米；仍需预留10次模型调用用于采摘。依据当前视觉选择方向，停车后重看两路图，不使用旧目标。无成熟果可搜，局部有未核实候选需先纠错；不能为了体现移动而跳过近果。\n'
        if self.require_delivery_before_finish:
            self.messages[0]['content']+='\n采集结束规则覆盖前文所有“或finish”建议：完成送筐和返回后才能finish。未完成时不得因定位困难、已用一些调用或单次预检拒绝提前结束；继续利用剩余调用查图、纠错、恢复或重新规划。只读inspect不消耗每站感知额度，仍消耗总模型调用。果柄未确认不能盲抓，也不要捏造失败次数；若仍被阻断，查图说明具体阻碍，不重复请求finish。总调用上限仍由程序执行，真实运动保护和用户中断仍停止。bend_pull先预检请求路径，若不可行可在同样距离、同样高度和俯仰角下沿当前夹爪向后退，回执明确记录实际位移；两条路径都不可行只拒绝该动作，不结束回合。\n'
            self.messages[0]['content']+='\n局部候选规则更新，覆盖前文“最近不确定果必须先解决”的要求：不确定碎片和无可靠深度区域保留记录，但不阻止选择已测量且视觉成熟的其他候选，以local_selection.active为准。若没有可处理的明确候选，复核过的不确定区域允许search_row短距换视角重新survey，无需把它改成not_fruit或伪造失败次数。优先使用这个有效出口，不要反复inspect/write_working等待额度耗尽。距离变化超过4mm只把该候选重新标为uncertain，其余候选复核仍生效；未知不等于无果，移动后必须用新图重新定位。\n'

    def log(self, kind, **data):
        with self.events.open('a') as f:
            f.write(json.dumps({'kind': kind, 'sim_time': float(self.e.data.time), **data}, ensure_ascii=False) + '\n')
        if kind == 'request':
            print(f"[VISTA {data['call']}/{self.max_calls}] 正在请求模型，观察 {data['basis_frame_id']}", flush=True)
        elif kind == 'response':
            names = [c['function']['name'] for c in data['message'].get('tool_calls', [])]
            print(f"[VISTA] {data['wall_latency_s']:.1f}秒后收到回复：{', '.join(names) or '文本（无工具）'}", flush=True)
        elif kind == 'receipt':
            result = data['result']
            if isinstance(result, list):
                summary = '; '.join(f"{r['action']['action']}: {r['status']}" + (f" ({r['reason']})" if 'reason' in r else '') for r in result)
                if result and 'jaw_contact' in result[-1]:
                    summary += '；双指外部接触=' + str(result[-1]['jaw_contact']['both'])
            elif result.get('status') == 'rejected': summary = '未执行：' + result['reason']
            elif result.get('episode_ended'): summary = '模型请求结束，等待环境验收'
            elif result.get('status') == 'executed': summary = f"{result.get('stage', 'motion')}: executed"
            else: summary = '查图/记忆工具完成（不移动机器人）'
            print('[VISTA] ' + summary, flush=True)

    def tick(self, q, phase, realtime=False):
        start = time.monotonic()
        controller=getattr(self,'centered',None)
        if controller is not None:q,expected=controller.command(q)
        if self.viewer:
            if not self.viewer.is_running(): raise RuntimeError('Viewer closed')
            with self.viewer.lock(): self.rec.tick(q, phase)
            self.viewer.sync()
        else: self.rec.tick(q, phase)
        if getattr(self,'detachment_confirmation',None) and not jaw_contact_state(self.e)['both']:
            self.detachment_confirmation=None
        if controller is not None:
            try:controller.check_tracking(expected)
            except TrackingError as exc:
                exc.phase=phase
                raise
        if realtime or self.viewer: time.sleep(max(0, self.e.dt - (time.monotonic()-start)))

    def observe(self):
        self.stalk_view_evidence=None
        frame = f'f{self.e.frame:07d}'
        state = robot_state(self.e)
        info = {'frame_id': frame, 'simulation_time': float(self.e.data.time),
                'camera_image_size': {'width': 480, 'height': 360}, **state}
        if self.e.meta.get('harvest_mechanism'):info['jaw_contact']=jaw_contact_state(self.e)
        if self.e.meta.get('tilted_stalk_prior'):info['station_search']=self.station_status()
        if getattr(self,'local_targets',None) is not None:info['local_selection']=self.local_targets.status()
        if getattr(self,'preclose_visual_check',False):
            info['preclose_recovery']={'used':getattr(self,'preclose_recoveries',0),'limit':2,
                'failed':getattr(self,'preclose_recovery_failed',False),
                'rejected_measured_targets_m':getattr(self,'visual_rejected_targets',[])}
        if getattr(self,'parking_trial',{}).get('enabled'):info['parking_trial']=self.parking_trial.copy()
        content = [{'type': 'text', 'text': json.dumps(info)}]
        paths = {}
        cameras=getattr(self,'policy_cameras',tuple(self.e.meta.get('policy_cameras',('top','wrist','harvest_close') if self.e.meta.get('harvest_mechanism') else ('top','wrist'))))
        for camera in cameras:
            image = Image.fromarray(self.e.render(camera, 480, 360))
            path = self.rec.out/'observations'/f'{frame}-{camera}.jpg'
            path.parent.mkdir(exist_ok=True); image.save(path)
            paths[camera] = path
            depth, calibration = self.e.depth_view(camera, 480, 360)
            depth_path = self.rec.out/'observations'/f'{frame}-{camera}-depth.npz'
            np.savez_compressed(depth_path, depth=depth, **calibration)
            self.depth_frames[(frame,camera)] = depth_path
            display=image;label=camera
            if camera=='harvest_close' and getattr(self,'local_targets',None) is not None and (not self.local_targets.surveyed or can_correct_survey(self.local_targets) or can_review_proposals(self.local_targets) or (self.local_targets.allow_uncertain_deferral and self.target_stage is None and not self.grasp_attempted)):
                if getattr(self,'visual_proposals',False):
                    from visual_proposals import draw_proposals
                    # Use the same saved RGB that the tool later checks.
                    with Image.open(path) as saved:survey_rgb=saved.convert('RGB')
                    batch=current_body_proposals(self.local_targets,survey_rgb,depth,calibration,state['base_x_y_yaw'],depth_point)
                    self.proposal_frames={frame:batch}
                    content.extend([{'type':'text','text':camera+' original image'},image_block(image)])
                    display=draw_proposals(survey_rgb,batch)
                    label+=' numbered body proposals; use proposal_id, not displayed pixel coordinates. '+json.dumps(batch)
                    self.log('body_proposals',frame_id=frame,**batch)
                else:
                    display,mapping=inspect_crop(image,[0,0,480,360])
                    label+=': survey image with rulers; labels are ORIGINAL 480x360 coordinates. Ignore the 32px ruler margin when reporting x/y.'
            content.extend([{'type': 'text', 'text': label}, image_block(display)])
        self.frames[frame] = paths
        self.edge_viewpoint_evidence=None
        local=getattr(self,'local_targets',None)
        if local is not None and getattr(local,'proposal_review_attempts',0)>=2:
            from perception_recovery import measured_edge_fragments
            reference=getattr(local,'survey_reference_depth',None) or self.depth_frames.get((local.survey_review['frame_id'],'harvest_close'))
            if reference is not None:
                with np.load(reference) as old, np.load(self.depth_frames[(frame,'harvest_close')]) as sample, Image.open(paths['harvest_close']) as rgb:
                    if (np.linalg.norm(sample['position']-old['position'])<=.002
                            and np.max(np.abs(sample['rotation']-old['rotation']))<=.01):
                        fragments=measured_edge_fragments(local,rgb,sample['depth'],sample,state['base_x_y_yaw'],depth_point)
                        self.edge_viewpoint_evidence={'frame_id':frame,'fragments':fragments}
        if getattr(self,'pull_completed',None) and not getattr(self,'delivery_completed',False):
            from detachment_review import gripper_context_region
            reference=getattr(self,'pull_reference_frame',None)
            content.append({'type':'text','text':json.dumps({'pull_visual_reference':reference,
                'pull_completed':self.pull_completed,'detachment_review':getattr(self,'detachment_confirmation',None),
                'instruction':'Compare BEFORE pull images below with CURRENT camera images above. Fruit motion alone is not separation. Explicit confirm_detachment required before delivery. If both connection views are occluded or either shows a remaining connection, do not transport.'})})
            for camera in cameras:
                for label,basis,pose in (('BEFORE',reference,getattr(self,'pull_reference_state',None)),('AFTER',frame,state)):
                    path=self.frames.get(basis,{}).get(camera)
                    depth_path=self.depth_frames.get((basis,camera))
                    if path is None or depth_path is None or pose is None:continue
                    with np.load(depth_path) as calibration,Image.open(path) as source:
                        box=gripper_context_region(pose['tcp_world_m'],calibration,source.size)
                        zoom,mapping=inspect_crop(source,box) if box else (source.copy(),None)
                    content.extend([{'type':'text','text':f'{label} pull: {basis}, {camera}. Robot-centered context, not object identity; inspect physical connection. '+json.dumps(mapping)},image_block(zoom)])
        self.stalk_proposals={}
        crop_info,crop_image=self.active_candidate_crop(frame,state)
        if crop_info:
            content.extend([{'type':'text','text':json.dumps({'active_candidate_crop':crop_info})},image_block(crop_image)])
        local=getattr(self,'local_targets',None)
        if local is not None and local.active and (local.active['failures']>0 or getattr(self,'visual_proposals',False)) and 'wrist' in cameras:
            wrist_info,wrist_image=self.active_candidate_crop(frame,state,camera='wrist')
            if wrist_info:
                content.extend([{'type':'text','text':json.dumps({'wrist_active_candidate_crop':wrist_info})},image_block(wrist_image)])
        if uncertain_review_candidate(local) is not None:
            for camera in cameras:
                review_info,review_image=self.active_candidate_crop(frame,state,camera=camera,review_only=True)
                if review_info:
                    content.extend([{'type':'text','text':json.dumps({'uncertain_candidate_review_crop':review_info})},image_block(review_image)])
                else:
                    content.append({'type':'text','text':json.dumps({'uncertain_candidate_review_crop':{
                        'camera':camera,'candidate_key':uncertain_review_candidate(local)['candidate_key'],
                        'available':False,'instruction':'No valid projected context crop in this view. Do not invent correspondence or count this camera as visual confirmation.'}})})
        correction_info,correction_image=self.body_correction_feedback(frame)
        if correction_info:
            content.extend([{'type':'text','text':json.dumps({'body_correction_feedback':correction_info})},image_block(correction_image)])
        # The per-view fits above are needed before advertising a recovery.
        if self.e.meta.get('tilted_stalk_prior'):info['station_search']=self.station_status()
        if self.stalk_view_evidence is not None:info['stalk_view_evidence']=self.stalk_view_evidence
        if not getattr(self,'fixed_base',False) and ('harvest_close' in paths):
            from search_guidance import row_guidance
            with np.load(self.depth_frames[(frame,'harvest_close')]) as calibration:
                info['row_search_guidance']=row_guidance(state['base_x_y_yaw'],calibration,
                    getattr(self,'visual_search_hints',[]),getattr(self,'search_history',[]))
        content[0]['text']=json.dumps(info)
        self.log('observation', **info, images={k: str(v.relative_to(self.rec.out)) for k,v in paths.items()})
        return frame, state, {'role': 'user', 'content': content}

    def body_correction_feedback(self,frame):
        local=getattr(self,'local_targets',None)
        if local is None or local.active or self.target_stage is not None or self.grasp_attempted:
            return None,None
        item=next((c for c in local.candidates if c['status']=='unverified_measurement'),None)
        if item is None:return None,None
        with Image.open(self.frames[frame]['harvest_close']) as im:
            marked,mapping=localization_feedback(im,item['x'],item['y'])
            from perception_recovery import body_color_conflict, body_context_colors
            proposal=next((p for p in getattr(local,'survey_batch',{}).get('proposals',[])
                           if p['proposal_id']==item.get('proposal_id')),None)
            context=body_context_colors(im,proposal['bbox_xyxy']) if proposal else None
        mapping['coordinate_rule']='Rulers are ORIGINAL image coordinates for correct_candidate_body, not displayed pixels.'
        info={'candidate_key':item['candidate_key'],'frame_id':frame,
              'original_pixel':[item['x'],item['y']],'correction_radius_pixels':32,
              'remaining_attempts':max(0,2-item.get('correction_attempts',0)),
              'mapping':mapping,'instruction':'Rulers use ORIGINAL harvest_close coordinates. Correct the same nearby red body with correct_candidate_body, or finish if uncertain. Cross is the unverified pixel, not a stalk.'}
        if body_color_conflict(local,item):
            info.update(body_color_context=context,pixel_correction_allowed=False,
                instruction='Whole-body orange/red conflict, not an isolated seed pixel. Reconsider turning/green from the whole image using review_proposals; use uncertain when unclear. Keeping ripe leaves the blocker unresolved. Use available semantic observation recovery before ending; never hunt another red pixel.')
        marked.save(self.rec.out/'observations'/f'body-correction-{frame}-{item["candidate_key"]}.png')
        return info,marked

    def active_candidate_crop(self,frame,state,camera='harvest_close',review_only=False):
        local=getattr(self,'local_targets',None)
        active=uncertain_review_candidate(local) if review_only else (local.active if local is not None else None)
        if active is None or self.target_stage is not None or self.grasp_attempted:
            return None,None
        path=self.depth_frames.get((frame,camera))
        if path is None:return None,None
        with np.load(path) as sample:
            region=candidate_crop_region(active['surface_m'],sample,(480,360))
        def record_view(status):
            if review_only or not getattr(self,'visual_proposals',False):return
            prior=getattr(self,'stalk_view_evidence',None)
            if prior is None or prior['frame_id']!=frame or prior['candidate_key']!=active['candidate_key']:
                self.stalk_view_evidence={'frame_id':frame,'candidate_key':active['candidate_key'],'views':{}}
            self.stalk_view_evidence['views'][camera]=status
        if region is None:
            record_view('outside_crop')
            return None,None
        box,body_pixel=region
        with Image.open(self.frames[frame][camera]) as im:
            image,mapping=inspect_crop(im,box,local_coordinates=True)
        sx,sy,_,_=box;scale=mapping['scale'];ox,oy=mapping['display_origin_xy']
        cx=ox+(body_pixel[0]-sx)*scale;cy=oy+(body_pixel[1]-sy)*scale
        radius=5*scale
        ImageDraw.Draw(image).ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='cyan',width=2)
        if review_only:
            info={'frame_id':frame,'candidate_key':active['candidate_key'],'proposal_id':active.get('proposal_id'),
                  'camera':camera,'body_marker_source_pixel':body_pixel,'crop_mapping':mapping,
                  'purpose':'semantic_review_only',
                  'instruction':'Cyan ring projects the SAME measured color region; it does not prove fruit identity, visibility or maturity. Compare both views and whole-frame numbered image; use review_proposals to resolve identity/maturity. Not a stalk selection crop; do not localize or grasp until approved.'}
            self.log('uncertain_candidate_review_crop',**info)
            return info,image
        crop_id=f'active-{frame}-{active["candidate_key"]}'
        if camera!='harvest_close':crop_id+='-'+camera
        self.crops[crop_id]={'mapping':mapping,'camera':camera,'frame':frame,
                            'state':state,'call':self.calls,'candidate_key':active['candidate_key']}
        # Keep only the crop lifetime already allowed by locate_crop.
        self.crops={key:value for key,value in self.crops.items() if value.get('call',-2)>=self.calls-1}
        image.save(self.rec.out/'observations'/f'{crop_id}.png')
        info={'crop_id':crop_id,'frame_id':frame,'candidate_key':active['candidate_key'],
              'candidate_failures':active['failures'],'camera':camera,'crop_mapping':mapping,
              'body_marker_source_pixel':body_pixel,
              'instruction':'Cyan ring projects measured BODY, not stalk; projection does not prove visibility in this camera. Confirm the same fruit and select its visible short stalk with locate_crop. Image up is not necessarily world up in the wrist view. Do not select a different fruit.'}
        if getattr(self,'visual_proposals',False):
            from stalk_proposals import propose_stalk_points,mark_stalk_points
            with np.load(path) as sample,Image.open(self.frames[frame][camera]) as rgb:
                batch=propose_stalk_points(rgb.convert('RGB'),sample['depth'],sample,active['surface_m'],tilted_stalk_grasp_point)
            proposals=[]
            for i,p in enumerate(batch['proposals']):
                px,py=p['pixel']
                if not (sx<=px<sx+box[2] and sy<=py<sy+box[3]):continue
                item={**p,'proposal_id':('W' if camera=='wrist' else 'C')+str(i+1),
                      'camera':camera,'frame_id':frame,'candidate_key':active['candidate_key']}
                self.stalk_proposals[item['proposal_id']]={**item,'state':state}
                proposals.append(item)
            info['stalk_proposals']={**batch,'proposals':proposals}
            record_view('fit_available' if proposals else 'no_fit')
            info['instruction']='Cyan ring is BODY. Yellow labeled points are fitted green segments, not confirmed stalk identity. Wrist image up is NOT world up: a short stalk can point sideways on screen. Judge its connection to this fruit, not screen direction. If confirmed, use locate_stalk_proposal with label and visual reason. If ALL fits remain unconfirmed, see station_search for bounded adjust_observation with unconfirmed_stalk_proposals. No pixel guessing needed. Empty list does not prove absence.'
            mark_stalk_points(image,mapping,proposals)
            image.save(self.rec.out/'observations'/f'{crop_id}.png')
        self.log('active_candidate_crop',**info)
        return info,image

    def execute(self, actions):
        results = []
        for a in actions:
            name = a['action']; e = self.e; q = e.last_action.copy()
            stability=None
            visual_check=None
            start = e.frame
            centered_closed=False
            staged_pull=None
            try:
                e.drive(0, 0)
                if getattr(self,'centered',None) is not None and name!='observe':
                    self.detachment_confirmation=None
                    if getattr(self,'pull_completed',None):
                        raise ValueError('Staged pull already completed; visually confirm then deliver or finish, no repeated free motion')
                if getattr(self,'preclose_recovery_failed',False):
                    raise ValueError('Visual recovery incomplete; finish episode, no further motion allowed')
                if e.meta.get('harvest_mechanism') and not self.grasp_attempted and name in ('move_tcp','bend_pull','drive'):
                    raise ValueError('Use locate_pixel then approach_target(pregrasp/grasp) before grasping; free increments disabled during approach')
                if name in ('move_tcp','bend_pull'):
                    if name=='bend_pull' and e.meta.get('harvest_mechanism') and not jaw_contact_state(e)['both']:
                        raise ValueError('No bilateral external jaw contact; do not pull blindly. Inspect grasp or finish.')
                    if name=='bend_pull' and e.meta.get('tilted_stalk_prior'):
                        stability=self.check_contact_hold()
                        if not stability['stable']:
                            raise ValueError('Bilateral contact did not remain stable for 0.5s; no pull executed. Inspect grasp or finish.')
                    pitch = np.deg2rad(a['pitch_deg']) if name=='bend_pull' else self.approach_pitch
                    pos = e.data.site_xpos[e.tcp].copy() + [a['dx'], a['dy'], a['dz']]
                    controller=getattr(self,'centered',None)
                    if controller is not None and name=='bend_pull':
                        staged_pull=controller.pull([a['dx'],a['dy'],a['dz']],pitch,a['seconds'],
                                                    lambda:jaw_contact_state(e)['both'])
                    else:
                        q, error = controller.ik(pos,q[5],pitch) if controller is not None else e.ik(pos, q[5],approach_pitch=pitch)
                        if error > .006: raise ValueError('Requested TCP pose unreachable')
                    self.approach_pitch = pitch
                elif name == 'gripper':
                    controller=getattr(self,'centered',None)
                    if controller is not None and a['opening']<.2 and not self.grasp_attempted and (a['opening']!=-.15 or a['seconds']!=3):
                        raise ValueError('Centered grasp requires gripper opening=-0.15 and seconds=3')
                    reserve=4 if controller is not None else 3
                    if e.meta.get('harvest_mechanism') and a['opening']<.2 and not self.grasp_attempted and self.calls>0 and self.max_calls-self.calls<reserve:
                        raise ValueError(f'Insufficient call budget to start grasp: reserve {reserve} further calls for pull, visual confirmation if centered, delivery and finish.')
                    if e.meta.get('harvest_mechanism') and a['opening']<.2 and self.target_stage!='grasp':
                        raise ValueError('Complete approach_target stage=grasp before closing')
                    if getattr(self,'preclose_visual_check',False) and a['opening']<.2 and not self.grasp_attempted:
                        if getattr(self,'preclose_recoveries',0)>=2:
                            raise ValueError('Two preclose visual recoveries exhausted; finish episode')
                        visual_check=self.check_preclose_visual()
                        if not visual_check['passed']:
                            self.retreat_after_visual_rejection()
                            raise ValueError('Preclose visual check unverified: stalk displaced or occluded; withdrew open and cleared target. Reselect from fresh images or finish.')
                    q[5] = a['opening']
                    if a['opening']<.2:
                        first_close=not self.grasp_attempted;self.grasp_attempted=True
                        if controller is not None and first_close:
                            try:controller.close()
                            except ValueError:
                                self.preclose_recovery_failed=True
                                raise
                            centered_closed=True
                elif name == 'drive':
                    if getattr(self,'fixed_base',False):raise ValueError('Fixed harvest base: free driving disabled')
                    e.drive(a['linear'], a['angular'])
                if not centered_closed and staged_pull is None:
                    lost=0
                    for _ in range(round(a['seconds']/e.dt)):
                        self.tick(q, name)
                        if any(w.number for w in e.data.warning): raise RuntimeError('Physics warning; episode stopped')
                        if name in ('bend_pull','move_tcp') and getattr(self,'centered',None) is not None:
                            lost=0 if jaw_contact_state(e)['both'] else lost+1
                            if lost>=4:raise ValueError('External jaw contact lost for 0.2s; stopped')
                    if name in ('bend_pull','move_tcp') and getattr(self,'centered',None) is not None and np.linalg.norm(e.data.site_xpos[e.tcp]-pos)>.008:
                        raise ValueError('Requested TCP waypoint did not settle')
                receipt={'action': a, 'status': 'executed', 'start_frame': start, 'end_frame': e.frame,
                         'robot': robot_state(e)}
                if e.meta.get('harvest_mechanism'):receipt['jaw_contact']=jaw_contact_state(e)
                if stability is not None:receipt['contact_stability']=stability
                if visual_check is not None:receipt['preclose_visual_check']=visual_check
                if staged_pull is not None:
                    receipt['staged_pull']=staged_pull
                    self.pull_completed={'end_frame':e.frame,'robot':robot_state(e),'motion':staged_pull}
                results.append(receipt)
            except ValueError as exc:
                if (e.frame>start and getattr(self,'centered',None) is not None
                        and not isinstance(exc,PullPlanError)
                        and (isinstance(exc,TrackingError) or name in ('bend_pull','move_tcp'))):
                    self.motion_stop_trace={'action':name,'phase':getattr(exc,'phase',name),
                        'start_frame':start,'failure_frame':e.frame,'reason':str(exc),
                        'tracking':getattr(exc,'tracking',None)}
                results.append({'action': a, 'status': 'rejected', 'reason': str(exc), 'start_frame': start, 'end_frame': e.frame})
                if isinstance(exc,PullPlanError):
                    results[-1].update(recoverable=True,pull_motion_executed=False,
                                       planning_trials=exc.planning_trials)
                if stability is not None:results[-1]['contact_stability']=stability
                if visual_check is not None:results[-1]['preclose_visual_check']=visual_check
                results.extend({'action': remaining, 'status': 'not_executed'} for remaining in actions[len(results):])
                break
            finally: e.drive(0, 0)
        return results

    def check_preclose_visual(self):
        """Fresh cameras after model waiting; no simulator object truth inputs."""
        if (self.locked_target is None or not self.target_confirmed or not self.maturity_evidence
                or not self.maturity_evidence.get('passed')
                or not np.allclose(self.maturity_evidence.get('target_m',[]),self.locked_target,atol=1e-6,rtol=0)):
            raise ValueError('Confirmed mature stalk required for preclose verification')
        frame,_,_=self.observe()
        views={}
        for camera in getattr(self,'policy_cameras',('harvest_close','top','wrist')):
            with np.load(self.depth_frames[(frame,camera)]) as sample, Image.open(self.frames[frame][camera]) as image:
                views[camera]=(np.asarray(image.convert('RGB')).copy(),sample['depth'].copy(),
                               {k:sample[k].copy() for k in ('fovy','rotation','position')})
        check=check_stalk_alignment(self.locked_target,self.maturity_evidence['axis_world'],views)
        check['frame_id']=frame
        self.log('preclose_visual_check',**check)
        return check

    def retreat_after_visual_rejection(self):
        """Withdraw to the previous pregrasp point; never chase a pushed stalk."""
        e=self.e
        retreat=self.locked_target.copy()-np.array([.055,0,0])
        if getattr(self,'local_targets',None) is not None and self.local_targets.active:
            self.local_targets.fail('Preclose visual check rejected; withdrew',terminal=True)
        self.visual_rejected_targets=getattr(self,'visual_rejected_targets',[])+[self.locked_target.tolist()]
        self.preclose_recoveries=getattr(self,'preclose_recoveries',0)+1
        self.preclose_recovery_failed=True
        self.target_stage='visual_recovery_failed'
        self.target_confirmed=False
        self.locked_target=None;self.target_camera=None;self.maturity_evidence=None
        self.target_reachability=None
        self.crops.clear();self.frames.clear();self.depth_frames.clear()
        e.drive(0,0)
        back,error=e.ik(retreat,1,approach_pitch=0)
        if error>.006:raise ValueError('Visual recovery retreat unreachable; finish episode')
        for _ in range(55):
            self.tick(back,'approach_retreat')
            if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning during visual recovery')
        if np.linalg.norm(e.data.site_xpos[e.tcp]-retreat)>.008:
            raise ValueError('Visual recovery retreat incomplete; finish episode')
        self.target_stage=None;self.approach_pitch=0
        self.preclose_recovery_failed=False

    def check_contact_hold(self):
        """Hold existing commands; tactile persistence is not stalk identification."""
        e=self.e
        q=e.last_action.copy()
        e.drive(0,0)
        samples=[bool(jaw_contact_state(e)['both'])]
        if samples[0]:
            for _ in range(10):
                self.tick(q,'grasp_contact_check')
                if any(w.number for w in e.data.warning):
                    raise RuntimeError('Physics warning during contact hold')
                samples.append(bool(jaw_contact_state(e)['both']))
                if not samples[-1]:break
        return {'stable':len(samples)==11 and all(samples),
                'required_duration_s':.5,'observed_duration_s':(len(samples)-1)*e.dt,
                'bilateral_samples':samples,
                'note':'External tactile persistence only; does not prove both jaws hold the same stalk or that fruit is detached.'}

    def station_status(self):
        from unlisted_fruit_search import review_available as unlisted_review_available
        from unlisted_fruit_search import inspection_available as unlisted_inspection_available
        from perception_recovery import edge_viewpoint_search_available, uncertain_viewpoint_search_available
        attempts=getattr(self,'station_attempts',0)
        failures=getattr(self,'station_failures',0)
        local=getattr(self,'local_targets',None)
        if local is not None:
            limited=attempts>=12
            if getattr(self,'centered',None) is not None and self.grasp_attempted:
                if getattr(self,'delivery_completed',False):next_action='finish'
                elif getattr(self,'detachment_confirmation',None):next_action='deliver_to_basket or finish'
                elif getattr(self,'pull_completed',None):next_action='confirm_detachment using before/after views, inspect, or finish; no transport yet'
                else:next_action='one standalone staged bend_pull or finish'
            elif self.target_confirmed and local.active and local.active['status']=='ready':next_action='approach_target or finish'
            elif uncertain_viewpoint_search_available(self):next_action='search_row for a new observation viewpoint, then survey_candidates afresh. Unresolved regions are retained as uncertain, not declared absent. Do not repeat blocked reviews or fabricate pixel failures.'
            elif limited:next_action='finish; perception budget exhausted, not evidence that driving is needed'
            elif not local.surveyed:next_action='survey_candidates'
            elif exhausted_observation_recovery(self):next_action='search_row to resolve the failed observation recovery; other unresolved local candidates still block driving, or finish'
            elif observation_adjustment_available(self):next_action=('adjust_observation for the SAME uncertain candidate, then review_proposals using fresh views; no localization/grasp yet, or finish' if uncertain_review_candidate(local) is not None else 'adjust_observation for the SAME active candidate, then reobserve and relocalize; or finish')
            elif local.active:
                next_action=f"Localize ONLY {local.active['candidate_key']} ({local.active.get('proposal_id','body marker')}); failures {local.active['failures']}/2. Do not select another fruit until candidate_transition."
                options=unconfirmed_stalk_options(self)
                if options:
                    next_action+=' Current fitted fragments: '+json.dumps(options)+'. Confirm visible same-fruit short stalk identity before locate_stalk_proposal. If none can be confirmed, use adjust_observation with unconfirmed_stalk_proposals listing EVERY ID and its image reason; keep the SAME fruit. Screen direction alone is not identity evidence.'
            elif unlisted_inspection_available(self):next_action='inspect_unlisted: mark the claimed missing bodies in ORIGINAL harvest_close 480x360 coordinates. Read marked-pixel RGB-D feedback before repeating correct_survey. This works with a nonempty blocked queue but grants NO movement or candidate skipping; a wrong point does not prove absence.'
            elif unlisted_review_available(self):next_action='search_after_unlisted_review: identify ALL fruit bodies responsible for your missing_visible_ripe claim using ORIGINAL harvest_close 480x360 pixels and reasons. Fresh RGB-D must prove all outside local envelope before bounded observation search. Keep missing_visible_ripe=true. inspect or correct_survey for a genuinely mistaken claim; otherwise finish.'
            elif can_correct_survey(local):next_action='correct_survey: fresh whole-frame review and correction_reason; or finish. No localization/driving while blocked.'
            elif can_review_proposals(local):
                from perception_recovery import body_semantics_review_required
                conflicts=[c.get('proposal_id',c['candidate_key']) for c in local.candidates
                           if body_semantics_review_required(local,c)]
                next_action='review_proposals with fresh full numbered image and correction_reason; depth remeasurement and evidenced maturity correction, or finish'
                if conflicts:
                    next_action+='; '+', '.join(conflicts)+' have measured orange body context conflicting with ripe claims. Reconsider turning/green from the whole image; if unclear use uncertain. Keeping ripe does not resolve the conflict. Do NOT use correct_candidate_body to hunt isolated red pixels.'
            elif local.selection_blocked_reason:next_action='finish; '+local.selection_blocked_reason
            elif not local.can_search:
                unresolved=next(c for c in local.candidates if c['status'] in ('unverified_measurement','unverified_semantics'))
                from perception_recovery import body_color_conflict
                next_action=('finish; nearer proposal identity or maturity uncertain, no skipping or driving' if unresolved['status']=='unverified_semantics' else
                            'finish; whole-body color conflict remains after semantic review budget; no pixel hunting or driving' if body_color_conflict(local,unresolved) else
                            'correct_candidate_body for '+unresolved['candidate_key']+' (unverified pixel) or finish'
                             if unresolved.get('correction_attempts',0)<2 else
                             'finish; unverified body correction attempts exhausted, driving still prohibited')
                if edge_viewpoint_search_available(self):
                    next_action='search_row for observation-only viewpoint change: only measured clipped edge fragments remain uncertain after two reviews. Preserve uncertainty; no grasp authorized. Re-survey after parking, or finish.'
            else:next_action='finish' if getattr(self,'fixed_base',False) else 'search_row or finish'
            if getattr(self,'require_delivery_before_finish',False) and not getattr(self,'delivery_completed',False):
                next_action=next_action.replace('or finish','or inspect fresh views').replace('finish;', 'inspect fresh views;')
                if next_action=='finish':next_action='inspect fresh views; collection continues until the model call budget'
            return {'perception_calls':attempts,'failed_localizations':failures,'max_perception_calls':12,
                    'remaining_perception_calls':max(0,12-attempts),
                    'perception_budget_applies_to':[name for name in ['survey_candidates','correct_survey','review_proposals','adjust_observation','correct_candidate_body','search_after_unlisted_review','inspect_unlisted','inspect','locate_pixel','locate_crop','locate_stalk_proposal']
                                                   if name!='inspect' or not getattr(self,'require_delivery_before_finish',False)],
                    'ordinary_motion_uses_only_model_call_budget':True,
                    'observation_adjustment_also_uses_perception_budget':True,
                    'max_failures_per_candidate':2,'station_limit_reached':limited,
                    'next_action':next_action,'local_selection':local.status()}
        max_attempts,max_failures=(12,4) if getattr(self,'fixed_base',False) else (6,2)
        limited=attempts>=max_attempts or failures>=max_failures
        plan=getattr(self,'target_reachability',None)
        next_action=plan['next_action'] if plan and not plan['current']['reachable'] else ('approach_target or finish' if self.target_confirmed else 'search_row or finish')
        if getattr(self,'fixed_base',False):
            next_action='approach_target or finish' if self.target_confirmed and (not plan or plan['current']['reachable']) else 'finish'
        return {'perception_calls':attempts,'failed_localizations':failures,
                'max_perception_calls':max_attempts,'max_failed_localizations':max_failures,
                'station_limit_reached':limited,
                'next_action':next_action if limited else 'inspect or localize'}

    def dispatch(self,name,args,frame,state,age):
        local=getattr(self,'local_targets',None)
        selecting=name in ('locate_pixel','locate_crop','locate_stalk_proposal')
        self._local_depth_attempted=False
        previous_candidate=None
        if local is not None and selecting and not self.grasp_attempted:
            previous_candidate=local.require_active()['candidate_key']
        # Count outer tools only: crop/number selection delegates to locate_pixel.
        bounded=(self.e.meta.get('tilted_stalk_prior') and not self.grasp_attempted
                 and self.target_stage is None and name in ('inspect','locate_pixel','locate_crop','locate_stalk_proposal','survey_candidates','correct_survey','review_proposals','adjust_observation','correct_candidate_body','search_after_unlisted_review','inspect_unlisted'))
        if name=='inspect' and getattr(self,'require_delivery_before_finish',False):bounded=False
        if bounded:
            status=self.station_status()
            if status['station_limit_reached']:
                return {'status':'rejected','reason':'Station perception budget exhausted; '+status['next_action'],
                        'station_search':status},[]
            self.station_attempts=getattr(self,'station_attempts',0)+1
        try:
            frame_alias=False
            if getattr(self,'visual_proposals',False) and name in ('survey_candidates','correct_survey','review_proposals'):
                from visual_proposals import normalize_review_frame
                frame_alias='basis_frame_id' in args
                args=normalize_review_frame(args,frame)
            result,images=self._dispatch(name,args,frame,state,age)
            if frame_alias:result['frame_argument_normalized']='basis_frame_id -> frame_id; current-frame equality checked'
        except CandidateCorrespondenceError as exc:
            return {'status':'rejected','reason':str(exc),'candidate_failure_counted':False,
                    'attempted_candidate_key':previous_candidate,'local_selection':local.status(),
                    'station_search':self.station_status()},[]
        except (ValueError,KeyError,TypeError) as exc:
            if bounded and selecting:
                # An expired crop or malformed request never measured the active
                # fruit, so it must not spend that fruit's localization retries.
                if local is None or self._local_depth_attempted:
                    self.station_failures=getattr(self,'station_failures',0)+1
                    if local is not None:self.local_failure(str(exc))
            elif local is not None and name=='approach_target' and self.target_stage is None and self.locked_target is None and local.active:
                self.local_failure(str(exc),terminal=True)
            raise
        if local is not None and selecting and not self.grasp_attempted:
            charge_failure=result.get('candidate_failure_counted',True)
            result['attempted_candidate_key']=previous_candidate
            result['candidate_failure_counted']=False
            if result.get('status')=='rejected' or not result.get('target_confirmed'):
                if charge_failure:
                    if (local.active and result.get('cross_view_checks') and not result.get('target_confirmed')
                            and result.get('maturity_check',{}).get('passed')):
                        local.active['cross_view_failed']=True
                        local.active['cross_view_checks']=result['cross_view_checks']
                        local.active['cross_view_failure_frame']=int(frame[1:])
                        local.active['cross_view_target_m']=result.get('locked_target_m')
                        result['reason']='Cross-view depth inconsistent or occluded; this does not prove unreachable. Reobserve the SAME active fruit; do not select another fruit pixel.'
                        if result.get('reachability'):
                            result['reachability']['next_action']='Visual confirmation still required; follow station_search.next_action'
                    self.local_failure(result.get('reason','Cross-view confirmation failed'),terminal=bool(result.get('maturity_check') and not result['maturity_check']['passed']))
                    result['candidate_failure_counted']=True
            elif not result['reachability']['current']['reachable']:
                if result.get('maturity_check',{}).get('passed') and result.get('target_confirmed'):
                    if not hasattr(self,'visual_search_hints'):self.visual_search_hints=[]
                    self.visual_search_hints.append({'frame_id':frame,
                        'candidate_key':previous_candidate,
                        'measured_stalk_m':list(result['locked_target_m']),
                        'source':'Model-confirmed short stalk; RGB-D maturity and cross-view checks passed; current centered pose unreachable'})
                    self.visual_search_hints=self.visual_search_hints[-8:]
                local.fail('Measured stalk unreachable from current parking',terminal=True)
                if local.active is not None:self.forget_local_target()
                else:
                    self.target_reachability=self.plan_parking()
                    result['reachability']=self.target_reachability
                result.update(status='rejected',reason='Candidate unreachable here; process remaining local candidates before any parking move')
            else:local.ready()
            result['local_selection']=local.status()
            next_candidate=local.active['candidate_key'] if local.active else None
            if previous_candidate!=next_candidate:
                result['candidate_transition']={'previous':previous_candidate,'next':next_candidate,
                    'instruction':'Old crops invalidated. Inspect the new active fruit before selecting its stalk; do not reuse the previous crop or pixel.'}
                result['target_confirmed']=self.target_confirmed
        if (local is not None and name=='approach_target' and self.target_stage is None
                and result.get('status')=='rejected' and result.get('maturity_check') and local.active):
            self.local_failure('Maturity changed before approach',terminal=True)
            result['local_selection']=local.status()
        if bounded:
            if selecting and result.get('status')=='rejected' and result.get('candidate_failure_counted',True):
                self.station_failures=getattr(self,'station_failures',0)+1
            result['station_search']=self.station_status()
        return result,images

    def forget_local_target(self):
        self.locked_target=None;self.target_camera=None;self.target_confirmed=False
        self.maturity_evidence=None;self.target_reachability=None
        self.stalk_view_evidence=None
        getattr(self,'crops',{}).clear()

    def local_failure(self,reason,terminal=False):
        local=self.local_targets
        previous=local.active
        if previous is None:return
        local.fail(reason,terminal=terminal)
        if previous is not local.active:self.forget_local_target()

    def require_local_search(self, observation_only=False):
        local=getattr(self,'local_targets',None)
        if local is not None and not local.can_search:
            from perception_recovery import edge_viewpoint_search_available, uncertain_viewpoint_search_available
            if observation_only and (edge_viewpoint_search_available(self) or uncertain_viewpoint_search_available(self)):return
            from unlisted_fruit_search import review_available
            evidence=getattr(self,'unlisted_search_evidence',None)
            if observation_only and evidence and review_available(self) and evidence['frame_id'] in self.frames:return
            raise ValueError('Local-first: survey and process other nearby candidates before moving; two pixel failures are not a reason to drive')

    def _dispatch(self, name, args, frame, state, age):
        if not isinstance(args, dict): raise ValueError('Arguments must be an object')
        if name=='inspect_unlisted':
            from unlisted_fruit_search import inspection_available,inspect_claim_pixels
            if (set(args)!={'basis_frame_id','bodies'} or args['basis_frame_id']!=frame
                    or not inspection_available(self)):
                raise ValueError('Unlisted inspection requires current missing-fruit block before approach; at most twice per station')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            local=self.local_targets
            local.unlisted_inspections=getattr(local,'unlisted_inspections',0)+1
            with Image.open(self.frames[frame]['harvest_close']) as rgb, np.load(self.depth_frames[(frame,'harvest_close')]) as sample:
                checks=inspect_claim_pixels(rgb.convert('RGB'),sample['depth'],sample,state['base_x_y_yaw'],
                    args['bodies'],depth_point,getattr(local,'survey_batch',{}).get('proposals',[]))
                images=[]
                for item in checks:
                    marked,mapping=localization_feedback(rgb,*item['pixel'])
                    item['feedback_mapping']=mapping
                    images.extend([{'type':'text','text':'Claimed missing-fruit point; rulers use ORIGINAL image coordinates. '+json.dumps(item)},image_block(marked)])
            entry={'kind':'unlisted_pixel_inspection','frame_id':frame,'checks':checks,'movement_authorized':False}
            self.perception_recoveries.append(entry)
            return {'status':'inspected','inspection':entry,'local_selection':local.status()},images
        if name=='search_after_unlisted_review':
            from unlisted_fruit_search import review_available,measure_unlisted
            if (set(args)!={'basis_frame_id','all_missing_claims_reported','bodies'} or args['basis_frame_id']!=frame
                    or args['all_missing_claims_reported'] is not True or not review_available(self)):
                raise ValueError('Unlisted review requires current empty local queue, missing-fruit block, and ALL unnumbered body claims; before approach only')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            with Image.open(self.frames[frame]['harvest_close']) as source, np.load(self.depth_frames[(frame,'harvest_close')]) as sample:
                evidence=measure_unlisted(source.convert('RGB'),sample['depth'],
                    {k:sample[k].copy() for k in sample.files if k!='depth'},state['base_x_y_yaw'],args['bodies'],depth_point)
            evidence['frame_id']=frame
            # Permit only this immediately validated search call. Never cache
            # permission across model requests, stale images or failed moves.
            self.unlisted_search_evidence=evidence
            try:return self.search_row(evidence['direction']),[]
            finally:self.unlisted_search_evidence=None
        if name=='review_proposals':
            local=getattr(self,'local_targets',None)
            if not self.visual_proposals or not can_review_proposals(local) or self.target_stage is not None or self.grasp_attempted:
                raise ValueError('Proposal review only for unresolved pre-approach evidence: two ordinary reviews plus one earned by a completed fresh observation adjustment')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            check_observation(local.survey_state,state,age,self.max_wait)
            reference=getattr(local,'survey_reference_depth',None) or self.depth_frames.get((local.survey_review['frame_id'],'harvest_close'))
            if reference is None:raise ValueError('Original survey camera calibration unavailable')
            with np.load(reference) as old, np.load(self.depth_frames[(frame,'harvest_close')]) as new:
                if (np.linalg.norm(new['position']-old['position'])>.002 or
                        np.max(np.abs(new['rotation']-old['rotation']))>.01):
                    raise ValueError('Survey camera moved; original proposal regions are no longer valid')
            from visual_proposals import reviewed_candidates
            review={k:v for k,v in args.items() if k not in ('correction_reason','duplicate_links')}
            batch=self.proposal_frames.get(frame)
            if batch is None:raise ValueError('Fresh numbered image required')
            with Image.open(self.frames[frame]['harvest_close']) as source:
                rgb=source.convert('RGB')
                measured,blocked=reviewed_candidates(review,batch,frame,rgb)
            wrist_view=None
            if args.get('duplicate_links') or getattr(local,'duplicate_links',[]):
                with Image.open(self.frames[frame]['wrist']) as image, np.load(self.depth_frames[(frame,'wrist')]) as sample:
                    wrist_view=(image.convert('RGB'),{k:sample[k].copy() for k in sample.files})
            entry=review_proposals(local,review,batch,measured,blocked,args.get('correction_reason'),rgb,state['base_x_y_yaw'],
                                   args.get('duplicate_links'),wrist_view)
            self.forget_local_target()
            self.perception_recoveries.append({'kind':'proposal_review',**entry})
            unresolved=bool(entry['unresolved_claims'] or local.selection_blocked_reason)
            return {'status':'unresolved' if unresolved else 'reviewed','review':entry,
                    'maturity_approval':False,'next_action':self.station_status()['next_action'],
                    'local_selection':local.status()},[]
        if name=='correct_survey':
            local=getattr(self,'local_targets',None)
            if not getattr(self,'visual_proposals',False) or not can_correct_survey(local):
                raise ValueError('No correctable coverage block, or two correction attempts exhausted')
            if self.target_stage is not None or self.grasp_attempted:
                raise ValueError('Coverage correction only before approach')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            check_observation(local.survey_state,state,age,self.max_wait)
            from visual_proposals import reviewed_candidates
            review={k:v for k,v in args.items() if k!='correction_reason'}
            batch=self.proposal_frames.get(frame)
            if batch is None:raise ValueError('Fresh numbered whole-frame observation required')
            with Image.open(self.frames[frame]['harvest_close']) as source:
                _,blocked=reviewed_candidates(review,batch,frame,source.convert('RGB'))
            entry=correct_survey(local,review,batch,blocked,args.get('correction_reason'))
            self.perception_recoveries.append({'kind':'survey_correction',**entry})
            return {'status':'corrected' if blocked is None else 'blocked','review':entry,
                    'local_selection':local.status()},[]
        if name=='adjust_observation':
            required={'basis_frame_id','candidate_key','reason'}
            if (not required<=set(args) or set(args)-required-{'unconfirmed_stalk_proposals'} or args['basis_frame_id']!=frame
                    or not isinstance(args['reason'],str) or not 1<=len(args['reason'].strip())<=1200):
                raise ValueError('Observation adjustment requires current frame, active candidate and visual reason')
            stalk_review=(review_unconfirmed_stalks(self,args['unconfirmed_stalk_proposals'],frame)
                          if 'unconfirmed_stalk_proposals' in args else None)
            if not stalk_review and not observation_adjustment_available(self,frame,call_charged=True):
                raise ValueError('Observation adjustment requires cross-view failure, one unresolved semantic review, or fresh same-candidate views with no fitted stalk; once per candidate and twice per station')
            uncertain=uncertain_review_candidate(self.local_targets)
            active=uncertain if uncertain is not None else self.local_targets.require_active()
            region_review=uncertain_region_evidence(self.local_targets,uncertain)
            if args['candidate_key']!=active['candidate_key']:raise ValueError('Keep the SAME active candidate')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            local=self.local_targets
            anchor=getattr(local,'observation_anchor',{'tcp_world_m':state['tcp_world_m'],'roll':state['joints_rad'][4]})
            index=getattr(local,'observation_adjustments',0)
            # Frame the already measured body for an unconfirmed fit as well;
            # pure retraction can otherwise put the wrist view outside the crop.
            # Semantic/weak-region recovery retains its original limited path.
            if missing_stalk_views(self,frame) or (stalk_review and active.get('body_pixel_verified') is True):
                try:
                    plan=self.centered.observation_plan(anchor,index,body_point=active['surface_m'])
                except ValueError as exc:
                    if not stalk_review:raise
                    framing_error=str(exc)
                    plan=self.centered.observation_plan(anchor,index)
                    self.centered.observation_plan_diagnostics={'mode':'retract_raise',
                        'camera_framing_fallback_reason':framing_error,'visibility_certified':False}
            else:plan=self.centered.observation_plan(anchor,index)  # Entire path before any motion.
            if getattr(local,'survey_review',None) is not None:
                # Any arm adjustment can precede a later review of another
                # candidate. Keep the fixed camera reference before clearing
                # frame handles, including adjustments for a ripe candidate.
                reference=getattr(local,'survey_reference_depth',None) or self.depth_frames.get((local.survey_review['frame_id'],'harvest_close'))
                if reference is None:raise ValueError('Original survey camera calibration unavailable')
                local.survey_reference_depth=reference
            local.observation_anchor=anchor;local.observation_adjustments=index+1
            active['observation_adjusted']=True
            self.forget_local_target()
            self.frames.clear();self.depth_frames.clear();self.crops.clear();self.stalk_proposals.clear();self.proposal_frames.clear()
            self.stalk_view_evidence=None
            self.e.drive(0,0)
            start_frame=self.e.frame
            try:
                result=self.centered.adjust_observation(plan)
            except (ValueError,RuntimeError) as exc:
                self.motion_stop_trace={'phase':'observation_adjust','start_frame':start_frame,
                    'failure_frame':self.e.frame,'reason':str(exc),'tracking':getattr(exc,'tracking',None)}
                raise MotionSafetyStop('Observation adjustment interrupted; '+str(exc)) from exc
            # This is an executed, checked posture change, not uncommanded
            # observation drift. Current-frame freshness and original camera
            # calibration/body correspondence are still checked on review.
            local.survey_state=robot_state(self.e)
            active['observation_recovery_completed_frame']=self.e.frame
            entry={'kind':'observation_adjust','semantic_review_only':uncertain is not None,'candidate_key':active['candidate_key'],
                   'start_frame':start_frame,'end_frame':self.e.frame,'reason':args['reason'],**result}
            if stalk_review:entry['unconfirmed_stalk_review']=stalk_review
            if region_review:entry['uncertain_region_review']=region_review
            self.perception_recoveries.append(entry)
            return {'status':'executed','stage':'observation_adjust','recovery':entry,
                    'next_action':('Use next fresh views for review_proposals of the SAME uncertain candidate; no grasp/localization before maturity confirmation.' if uncertain is not None else 'Use next observation to refit and confirm the SAME active candidate. All old stalk targets and crop labels invalidated.'),
                    'local_selection':local.status()},[]
        if name=='correct_candidate_body':
            local=getattr(self,'local_targets',None)
            if local is None or self.target_stage is not None or self.grasp_attempted:
                raise ValueError('Body correction only before approach in local-first mode')
            if set(args)!={'frame_id','candidate_key','x','y','reason'} or args['frame_id']!=frame:
                raise ValueError('Body correction requires current frame_id and candidate_key')
            if not isinstance(args['reason'],str) or not 1<=len(args['reason'])<=600:
                raise ValueError('Explain the local red-body correction')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            from perception_recovery import body_color_conflict
            candidate=next((c for c in local.candidates if c['candidate_key']==args['candidate_key']),None)
            if body_color_conflict(local,candidate):
                raise ValueError('Whole-body RGB evidence conflicts with the ripe claim: measured orange context with few red pixels. Use review_proposals on the latest full numbered image to reconsider maturity; do not hunt for a red pixel. No candidate was skipped or relabeled; body-correction budget unchanged.')
            item=local.begin_body_correction(args['candidate_key'],args['x'],args['y'])
            with np.load(self.depth_frames[(frame,'harvest_close')]) as sample:
                with np.load(self.depth_frames[(item['source_frame'],'harvest_close')]) as original:
                    if (np.linalg.norm(sample['position']-original['position'])>.002 or
                            np.max(np.abs(sample['rotation']-original['rotation']))>.01):
                        raise ValueError('Survey camera moved; original pixel neighbourhood no longer valid. Finish rather than relabel')
                point=depth_point(sample['depth'],sample,args['x'],args['y'])
            with Image.open(self.frames[frame]['harvest_close']) as source:
                rgb=source.convert('RGB')
            evidence=body_pixel_evidence(rgb,args['x'],args['y'])
            passed,reason=local.correct_body(item,{
                'x':args['x'],'y':args['y'],'source_frame':frame,'surface_m':point.tolist(),
                'body_pixel_check':evidence,'body_pixel_verified':evidence['passed']},state['base_x_y_yaw'])
            item['correction_history'][-1]['model_reason']=args['reason']
            if passed:self.forget_local_target()
            marked,mapping=localization_feedback(rgb,args['x'],args['y'])
            return {'status':'corrected' if passed else 'rejected','reason':reason,
                    'candidate_key':item['candidate_key'],'body_pixel_check':evidence,
                    'feedback_crop_mapping':mapping,'local_selection':local.status()},[image_block(marked)]
        if name=='survey_candidates':
            local=getattr(self,'local_targets',None)
            if local is None:raise ValueError('Candidate survey requires local-first full greenhouse mode')
            if self.target_stage is not None or self.grasp_attempted:raise ValueError('Survey only before approach')
            if getattr(self,'visual_proposals',False):
                from visual_proposals import reviewed_candidates
                batch=self.proposal_frames.get(frame)
                if batch is None:raise ValueError('No current-frame body proposals; use latest observation')
                check_observation(state,robot_state(self.e),age,self.max_wait)
                with Image.open(self.frames[frame]['harvest_close']) as source:
                    measured,blocked=reviewed_candidates(args,batch,frame,source.convert('RGB'))
                local.register(measured,state['base_x_y_yaw'],args['coverage'])
                local.selection_blocked_reason=blocked
                local.survey_review=json.loads(json.dumps(args))
                local.survey_batch=json.loads(json.dumps(batch))
                local.survey_state=json.loads(json.dumps(state))
                return {'status':'registered','local_selection':local.status()},[]
            if set(args)!={'frame_id','coverage','candidates'} or args['frame_id']!=frame:raise ValueError('Survey requires current frame_id')
            if not isinstance(args['coverage'],str) or not 1<=len(args['coverage'])<=1500:raise ValueError('Explain coverage and visible/occluded areas')
            if not isinstance(args['candidates'],list) or len(args['candidates'])>8:raise ValueError('Survey up to eight visible fruits')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            measured=[]
            with Image.open(self.frames[frame]['harvest_close']) as source:
                survey_rgb=source.convert('RGB')
            with np.load(self.depth_frames[(frame,'harvest_close')]) as sample:
                for item in args['candidates']:
                    if (not isinstance(item,dict) or set(item)!={'x','y','maturity','visibility','reason'}
                            or item['maturity'] not in ('ripe','turning','green','uncertain')
                            or item['visibility'] not in ('clear','partial','occluded')
                            or not isinstance(item['reason'],str) or not 1<=len(item['reason'])<=600):
                        raise ValueError('Invalid visual candidate description')
                    point=depth_point(sample['depth'],sample,item['x'],item['y'])
                    evidence=body_pixel_evidence(survey_rgb,item['x'],item['y']) if item['maturity']=='ripe' else None
                    measured.append({**item,'surface_m':point.tolist(),'source_frame':frame,
                                     'body_pixel_verified':evidence is None or evidence['passed'],
                                     'body_pixel_check':evidence})
            local.register(measured,state['base_x_y_yaw'],args['coverage'])
            return {'status':'registered','local_selection':local.status()},[]
        if getattr(self,'preclose_recovery_failed',False) and name in ('play','approach_target','search_row','refine_parking','deliver_to_basket'):
            raise ValueError('Visual recovery incomplete; finish episode, no further motion allowed')
        if getattr(self,'preclose_recoveries',0)>=2 and name in ('approach_target','search_row','refine_parking'):
            raise ValueError('Two preclose visual recoveries exhausted; finish episode')
        if getattr(self,'fixed_base',False) and name in ('search_row','refine_parking'):
            raise ValueError('Fixed harvest base: select a visible reachable ripe target or finish; no driving')
        if name=='refine_parking':
            if set(args)!={'basis_frame_id'} or args['basis_frame_id']!=frame:
                raise ValueError('Use latest basis_frame_id for parking refinement')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            if getattr(self,'target_aware_parking',False):
                if self.locked_target is None or not self.maturity_evidence:
                    raise ValueError('Confirm a mature target before parking')
                evidence,marked=self.check_maturity(frame,self.locked_target,self.maturity_evidence['axis_world'])
                self.maturity_evidence=evidence
                if not evidence['passed']:
                    return {'status':'rejected','reason':'Visual maturity changed; no parking executed','maturity_check':evidence},[image_block(marked)]
            return self.refine_parking(),[]
        if name=='search_row':
            from perception_recovery import edge_viewpoint_search_available, uncertain_viewpoint_search_available
            if set(args)!={'basis_frame_id','direction'} or args['basis_frame_id']!=frame or type(args['direction']) is not int or args['direction'] not in (-1,1):
                raise ValueError('Use latest basis_frame_id and direction -1 or 1')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            recovery=None
            if exhausted_observation_recovery(self,frame):
                active=self.local_targets.active
                recovery={'kind':'observation_recovery_exhausted','frame_id':frame,
                          'candidate_key':active['candidate_key'],'proposal_id':active.get('proposal_id'),
                          'completed_frame':active['observation_recovery_completed_frame'],
                          'views':dict(self.stalk_view_evidence['views']),
                          'failures_before':active['failures'],
                          'reason':'Mature stalk cross-view localization failed; completed arm observation recovery still has no usable two-view fit. Unavailable at this station only, not evidence of fruit absence.'}
                from perception_recovery import out_of_view_after_recovery
                outside=out_of_view_after_recovery(self,frame)
                if active.get('failures')==0 or outside:
                    recovery['reason']='Uncertain body became visually ripe after completed arm observation recovery, but fresh two-camera stalk fitting remains unavailable. Deferred at this station only; reobserve after moving, not evidence of fruit absence or global unreachability.'
                    if outside:
                        recovery['reason']='After completed semantic observation recovery and mature stalk localization, the other camera still excludes the measured stalk in a fresh calibrated projection. Defer only at this station; preserve maturity and actual failure count, no grasp authorized.'
                        recovery['cross_view_failure_frame']=active['cross_view_failure_frame']
                        recovery['cross_view_target_m']=active['cross_view_target_m']
                    recovery['semantic_review_frame']=active['semantic_recovery_review_frame']
                    recovery['resolution']='station_viewpoint_exhausted'
                    active['status']='unavailable'
                    active['station_deferred']=True
                    active['last_reason']=recovery['reason']
                    self.forget_local_target()
                else:
                    self.local_failure(recovery['reason'])
                    self.station_failures+=1
                recovery['failures_after']=active['failures']
                self.perception_recoveries.append(recovery)
                self.log('observation_recovery_exhausted',**{k:v for k,v in recovery.items() if k!='kind'})
                if not self.local_targets.can_search and not edge_viewpoint_search_available(self,frame) and not uncertain_viewpoint_search_available(self,frame):
                    return {'status':'rejected','reason':'Recovery exhausted for this candidate; process remaining local candidates before driving',
                            'candidate_transition':recovery,'local_selection':self.local_targets.status(),
                            'station_search':self.station_status()},[]
            if self.local_targets is not None and not self.local_targets.can_search and not edge_viewpoint_search_available(self,frame) and not uncertain_viewpoint_search_available(self,frame):
                self.require_local_search()
            result=self.search_row(args['direction'])
            if recovery is not None:result['candidate_transition']=recovery
            return result,[]
        if name=='confirm_detachment':
            from detachment_review import review_separation
            if getattr(self,'centered',None) is None or not getattr(self,'pull_completed',None):
                raise ValueError('Complete one centered staged pull before visual separation review')
            self.detachment_confirmation=None
            if (set(args)!={'basis_frame_id','reference_frame_id','views'} or args['basis_frame_id']!=frame or
                    args['reference_frame_id']!=self.pull_reference_frame or self.pull_reference_frame not in self.frames):
                raise ValueError('Use latest frame and the recorded BEFORE pull reference')
            if not frame.startswith('f') or int(frame[1:])<self.pull_completed['end_frame']:
                raise ValueError('Review a fresh frame after the pull')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            result=review_separation(args['views'],self.policy_cameras)
            if not jaw_contact_state(self.e)['both']:
                result.update(allowed=False,reason='External contact no longer bilateral; cannot transport')
            result.update(frame_id=frame,reference_frame_id=self.pull_reference_frame)
            self.detachment_reviews=getattr(self,'detachment_reviews',[])+[result]
            if result['allowed']:
                self.detachment_confirmation={**result,'robot':state}
            return result,[]
        if name=='deliver_to_basket':
            if set(args)!={'basis_frame_id'} or args['basis_frame_id']!=frame:
                raise ValueError('Use latest basis_frame_id for delivery')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            if getattr(self,'centered',None) is not None:
                review=getattr(self,'detachment_confirmation',None)
                if not review or not review['allowed']:
                    raise ValueError('Visual separation not confirmed: use confirm_detachment with before/after camera evidence, or finish')
                check_observation(review['robot'],state,age,self.max_wait)
            result=self.deliver_to_basket()
            self.delivery_completed=True
            return result,[]
        if name=='clear_target':
            if args or self.target_stage is not None or self.grasp_attempted:
                raise ValueError('Target can only be cleared before approach; finish if already moving/grasping')
            had_target=self.locked_target is not None
            self.locked_target=None;self.target_camera=None;self.target_confirmed=False;self.maturity_evidence=None
            self.target_reachability=None
            if had_target and getattr(self,'local_targets',None) is not None and self.local_targets.active:
                self.local_failure('Model cleared incorrect or unsuitable visual target',terminal=True)
            return {'target_cleared':True},[]
        if name=='locate_stalk_proposal':
            if not getattr(self,'visual_proposals',False):raise ValueError('Numbered stalk proposals not enabled')
            if set(args)!={'frame_id','candidate_key','proposal_id','confirmed_short_stalk','reason'} or args['frame_id']!=frame:
                raise ValueError('Use current frame and active candidate for stalk proposal')
            if args['confirmed_short_stalk'] is not True or not isinstance(args['reason'],str) or not 1<=len(args['reason'])<=600:
                raise ValueError('Confirm visible short stalk identity with an image reason, or finish')
            active=self.local_targets.require_active()
            item=self.stalk_proposals.get(args['proposal_id'])
            if item is None or item['frame_id']!=frame or item['candidate_key']!=active['candidate_key'] or args['candidate_key']!=active['candidate_key']:
                raise ValueError('Unknown/stale stalk proposal or different active candidate')
            if self.target_stage is not None or self.grasp_attempted:raise ValueError('Numbered stalk selection only before approach')
            check_observation(item['state'],state,age,self.max_wait)
            check_observation(state,robot_state(self.e),age,self.max_wait)
            x,y=item['pixel']
            result,images=self._dispatch('locate_pixel',{'frame_id':frame,'camera':item['camera'],'x':x,'y':y},frame,state,age)
            result['stalk_proposal_selection']={k:item[k] for k in ('proposal_id','camera','frame_id','candidate_key','pixel')}
            result['stalk_proposal_selection']['model_reason']=args['reason']
            return result,images
        if name == 'locate_crop':
            if set(args) != {'frame_id','crop_id','x','y'} or args['frame_id'] != frame:
                raise ValueError('Use current frame_id and crop_id with displayed crop x/y')
            crop = self.crops.get(args['crop_id'])
            if crop is None or self.calls not in (crop['call'], crop['call']+1):
                raise ValueError('Unknown or expired crop; inspect current active candidate again. No candidate localization failure counted.')
            local=getattr(self,'local_targets',None)
            if local is not None and crop.get('candidate_key')!=local.require_active()['candidate_key']:
                raise CandidateCorrespondenceError('Crop belongs to a different candidate; use the latest active_candidate_crop. No candidate localization failure counted.')
            check_observation(crop['state'], state, age, self.max_wait)
            x,y = crop_source_pixel(crop['mapping'],args['x'],args['y'])
            result,images = self._dispatch('locate_pixel',{'frame_id':frame,'camera':crop['camera'],'x':x,'y':y},frame,state,age)
            result['crop_selection']={'crop_id':args['crop_id'],'source_frame':crop['frame'],
                                      'sampled_frame':frame,'source_pixel':[x,y]}
            return result,images
        if name == 'locate_pixel':
            if set(args) != {'frame_id','camera','x','y'}: raise ValueError('Invalid locate_pixel fields')
            if args['frame_id'] != frame: raise ValueError('Use current observation for depth localization')
            path = self.depth_frames.get((frame,args['camera']))
            if path is None: raise ValueError('Unknown camera')
            with np.load(path) as sample:
                h,w=sample['depth'].shape
                if type(args['x']) is not int or type(args['y']) is not int or not 0<=args['x']<w or not 0<=args['y']<h:
                    raise ValueError('Pixel outside image; no candidate localization failure counted')
                point = depth_point(sample['depth'], sample, args['x'], args['y'])
                local=getattr(self,'local_targets',None)
                if local is not None and not self.grasp_attempted:
                    try:local.check_target(point)
                    except CandidateCorrespondenceError as exc:
                        with Image.open(self.frames[frame][args['camera']]) as source:
                            selected=source.convert('RGB')
                        marked,mapping=localization_feedback(selected,args['x'],args['y'])
                        return {'status':'rejected','reason':str(exc),'candidate_failure_counted':False,
                                'frame_id':frame,'camera':args['camera'],
                                'selected_pixel':[args['x'],args['y']],
                                'selected_rgb':list(selected.getpixel((args['x'],args['y']))),
                                'selected_surface_m':point.tolist(),
                                'delta_from_active_body_m':(point-np.asarray(local.active['surface_m'])).tolist(),
                                'feedback_crop_mapping':mapping,
                                'note':'Cyan cross is the ACTUAL selected original-image pixel. Its depth is outside the active body neighbourhood; it may hit background or another object. No fit was attempted and no candidate retry was charged. Feedback rulers are original-image pixels for locate_pixel, not locate_crop. Use the latest active_candidate_crop for crop coordinates.'},[image_block(marked)]
                self._local_depth_attempted=True
            checks={};images=[];fit=None;fit_camera=None
            if self.e.meta.get('harvest_mechanism') and not self.grasp_attempted:
                target=point
                if (self.e.meta.get('upright_stalk_prior') or self.e.meta.get('tilted_stalk_prior')) and self.locked_target is None:
                    fit_camera=args['camera'] if self.e.meta.get('tilted_stalk_prior') else 'harvest_close'
                    fit_depth=self.depth_frames.get((frame,fit_camera))
                    rgb=self.frames.get(frame,{}).get(fit_camera)
                    if fit_depth is None or rgb is None:raise ValueError('Selected-camera RGB-D required for stalk fit')
                    with np.load(fit_depth) as sample, Image.open(rgb) as fit_rgb:
                        try:
                            fitter=tilted_stalk_grasp_point if self.e.meta.get('tilted_stalk_prior') else stalk_grasp_point
                            target,fit=fitter(point,sample['depth'],sample,fit_rgb)
                        except StalkFitError as exc:
                            selected=Image.open(self.frames[frame][args['camera']]).convert('RGB')
                            marked,mapping=localization_feedback(selected,args['x'],args['y'])
                            return {'status':'rejected','reason':str(exc),'frame_id':frame,
                                    'camera':args['camera'],'selected_pixel':[args['x'],args['y']],
                                    'selected_rgb':list(selected.getpixel((args['x'],args['y']))),
                                    'stalk_fit':exc.diagnostics,'feedback_crop_mapping':mapping,
                                    'stalk_fit_camera':fit_camera,
                                    'suggested_camera':'wrist' if fit_camera=='harvest_close' else 'harvest_close',
                                    'note':'Cyan cross marks your selected pixel. Green samples are not semantic stalk recognition. No target locked or motion executed.'},[image_block(marked)]
                gate_target=self.locked_target if self.locked_target is not None else target
                if getattr(self,'local_targets',None) is not None:self.local_targets.check_target(gate_target)
                axis=(fit or self.maturity_evidence or {}).get('axis_world',[0,0,1])
                maturity,marked=self.check_maturity(frame,gate_target,axis)
                self.maturity_evidence=maturity
                if not maturity['passed']:
                    self.target_confirmed=False
                    self.target_reachability=None
                    return {'status':'rejected','frame_id':frame,'reason':'Visual maturity check failed; select a clearly red fruit and its stalk.',
                            'maturity_check':maturity,'stalk_fit':fit},[image_block(marked)]
                self.confirm_target(target,state,args['camera'])
                for camera in getattr(self,'policy_cameras',('harvest_close','top','wrist')):
                    other=self.depth_frames.get((frame,camera))
                    if camera==args['camera'] or other is None:continue
                    with np.load(other) as sample:
                        check=verify_projected_point(self.locked_target,sample['depth'],sample)
                    checks[camera]=check
                    if check['consistent']:self.target_confirmed=True
                    rgb=self.frames.get(frame,{}).get(camera)
                    if rgb and 'projected_pixel' in check:
                        im=Image.open(rgb).convert('RGB');x,y=map(round,check['projected_pixel'])
                        draw=ImageDraw.Draw(im)
                        draw.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=1)
                        box=(max(0,x-32),max(0,y-32),min(im.width,x+33),min(im.height,y+33))
                        crop=im.crop(box);crop=crop.resize((crop.width*4,crop.height*4))
                        images.extend([{'type':'text','text':f'{camera}: projected candidate (cyan ring), original crop box {box}, enlarged 4x; check this is the SHORT STALK.'},image_block(crop)])
            if self.locked_target is not None and self.e.meta.get('tilted_stalk_prior') and not self.grasp_attempted:
                self.target_reachability=self.plan_parking()
                if getattr(self,'parking_trial',{}).get('status')=='pending':
                    self.target_reachability['next_action']='approach_target(pregrasp): apply fixed +4cm parking first, then reobserve and reassess reachability'
                elif getattr(self,'parking_trial',{}).get('enabled') and not self.target_reachability['current']['reachable']:
                    self.target_reachability['next_action']='clear_target then search_row or finish; fixed parking already attempted'
            return {'frame_id':frame, 'surface_world_m':point.tolist(),
                    'reachability':getattr(self,'target_reachability',None),
                    'tcp_to_surface_m':(point-np.array(state['tcp_world_m'])).tolist(),
                    'locked_target_m':self.locked_target.tolist() if self.locked_target is not None else None,
                    'target_confirmed':self.target_confirmed,
                    'cross_view_checks':checks,
                    'stalk_fit':fit,
                    'stalk_fit_camera':fit_camera,
                    'maturity_check':self.maturity_evidence,
                    'note':'RGB-D visible surface. Check reachability before approach, even if target_confirmed. Geometry agreement is not stalk identification or collision clearance.'}, images
        if name=='approach_target':
            if set(args)!={'basis_frame_id','stage'} or args['basis_frame_id']!=frame or args['stage'] not in ('pregrasp','grasp'):
                raise ValueError('Invalid approach_target arguments')
            if not self.e.meta.get('harvest_mechanism') or self.locked_target is None:
                raise ValueError('Select visible short stalk using locate_pixel first')
            if not self.target_confirmed:raise ValueError('Confirm same stalk from another camera within 8mm before approach; clear_target if first selection was wrong')
            if getattr(self,'local_targets',None) is not None:self.local_targets.check_target(self.locked_target)
            if self.grasp_attempted:raise ValueError('Approach already completed; do not overwrite a held target')
            evidence=self.maturity_evidence
            if not evidence or not evidence['passed'] or not np.allclose(evidence['target_m'],self.locked_target,atol=1e-6,rtol=0):
                raise ValueError('Fresh visual maturity approval required for this exact target')
            if args['stage']=='pregrasp':
                trial=getattr(self,'parking_trial',{})
                if trial.get('enabled') and trial['status']!='applied':
                    if trial['status']!='pending':raise ValueError('Fixed parking trial failed; finish this episode')
                    check_observation(state,robot_state(self.e),age,self.max_wait)
                    evidence,marked=self.check_maturity(frame,self.locked_target,evidence['axis_world'])
                    self.maturity_evidence=evidence
                    if not evidence['passed']:
                        return {'status':'rejected','reason':'Visual maturity changed; no parking executed','maturity_check':evidence},[image_block(marked)]
                    return self.apply_parking_trial(),[]
                if self.e.meta.get('tilted_stalk_prior'):
                    self.target_reachability=self.plan_parking()
                    if (getattr(self,'target_aware_parking',False)
                            and self.target_reachability['recommended_delta_y_m'] is not None):
                        check_observation(state,robot_state(self.e),age,self.max_wait)
                        evidence,marked=self.check_maturity(frame,self.locked_target,evidence['axis_world'])
                        self.maturity_evidence=evidence
                        if not evidence['passed']:
                            return {'status':'rejected','reason':'Visual maturity changed; no parking executed','maturity_check':evidence},[image_block(marked)]
                        return self.refine_parking(plan=self.target_reachability),[]
                    if not self.target_reachability['current']['reachable']:
                        return {'status':'rejected','reason':'Target approach is kinematically unreachable; no arm motion executed',
                                'reachability':self.target_reachability},[]
                evidence,marked=self.check_maturity(frame,self.locked_target,evidence['axis_world'])
                self.maturity_evidence=evidence
                if not evidence['passed']:
                    return {'status':'rejected','reason':'Visual maturity changed or is obscured; no approach executed','maturity_check':evidence},[image_block(marked)]
                if getattr(self,'target_aware_parking',False):
                    print(f'[VISTA] 目标停车：保持当前 y={self.e.base_pose()[1]:.3f}m，按重新确认的果柄接近',flush=True)
            if args['stage']=='grasp' and self.target_stage!='pregrasp':raise ValueError('First use stage=pregrasp')
            check_observation(state,robot_state(self.e),age,self.max_wait)
            return self.execute_approach(args['stage']),[]
        if name == 'play':
            actions = validate_actions(args, frame)
            # Scene can evolve while waiting; reject excessive robot drift and old requests.
            check_observation(state, robot_state(self.e), age, self.max_wait)
            if getattr(self,'centered',None) is not None and any(a['action']=='bend_pull' for a in actions):
                if len(actions)!=1:raise ValueError('Run staged pull alone, then inspect fresh images before delivery')
                if getattr(self,'pull_completed',None):raise ValueError('Staged pull already completed; do not pull repeatedly')
                self.pull_reference_frame=frame
                self.pull_reference_state=state
            return self.execute(actions), []
        if name == 'inspect':
            if set(args)-{'frame_id', 'camera', 'region'}: raise ValueError('Unexpected inspect fields')
            path = self.frames.get(args.get('frame_id'), {}).get(args.get('camera'))
            if path is None: raise ValueError('Unknown recorded frame/camera')
            image = Image.open(path).copy()
            mapping = None
            crop_id = None
            if 'region' in args:
                r = args['region']
                if not isinstance(r, list) or len(r) != 4 or any(type(v) is not int for v in r): raise ValueError('Invalid crop')
                x,y,w,h = r
                if x < 0 or y < 0 or w < 1 or h < 1 or x+w > image.width or y+h > image.height:
                    raise ValueError(f'Crop outside {image.width}x{image.height} image; region is [x,y,width,height], not [x1,y1,x2,y2]')
                image, mapping = inspect_crop(image, r, local_coordinates=True)
                if args['frame_id'] == frame:
                    crop_id = f'crop-{self.calls}-{len(self.crops)+1}'
                    self.crops[crop_id]={'mapping':mapping,'camera':args['camera'],'frame':frame,
                                         'state':state,'call':self.calls,
                                         'candidate_key':self.local_targets.active['candidate_key'] if getattr(self,'local_targets',None) is not None and self.local_targets.active else None}
            return {'source_frame': args['frame_id'], 'camera': args['camera'], 'region': args.get('region'), 'crop_mapping': mapping, 'crop_id':crop_id,
                    'selection_rule':'locate_crop uses displayed crop pixels and latest frame_id; valid only this or next model call. Depth is sampled from latest frame. Historical crops cannot be selected.'}, [image_block(image)]
        if name in ('write_working', 'propose_guide'):
            if set(args) != {'content'} or not isinstance(args['content'], str) or len(args['content']) > 12000: raise ValueError('Invalid note')
            file = 'working.md' if name == 'write_working' else 'guide-proposal.md'
            (self.rec.out/file).write_text(args['content'])
            return {'saved': file, 'verified': False}, []
        if name == 'read_guide':
            if args: raise ValueError('read_guide takes no arguments')
            path = ROOT/'vista-guide.md'
            return {'guide': path.read_text() if path.exists() else 'No verified cross-episode experience yet.'}, []
        if name == 'read_working':
            if args: raise ValueError('read_working takes no arguments')
            path = self.rec.out/'working.md'
            return {'content': path.read_text() if path.exists() else ''}, []
        if name == 'history':
            if set(args) != {'last'} or type(args['last']) is not int or not 1 <= args['last'] <= 10:
                raise ValueError('history.last must be 1–10')
            return {'receipts': self.receipts[-args['last']:]}, []
        if name == 'finish':
            from unlisted_fruit_search import inspection_available as unlisted_inspection_available
            if set(args) != {'reason'} or not isinstance(args['reason'], str) or not 1 <= len(args['reason']) <= 4000: raise ValueError('Invalid finish reason')
            if (getattr(self,'require_delivery_before_finish',False)
                    and not getattr(self,'delivery_completed',False)
                    and not getattr(self,'motion_stop_trace',None)
                    and not getattr(self,'preclose_recovery_failed',False)):
                return {'episode_ended':False,'status':'continue_collection',
                        'submitted_finish_reason':args['reason'],
                        'remaining_model_calls':max(0,self.max_calls-self.calls),
                        'station_search':self.station_status(),
                        'instruction':'Collection is unfinished. Do not voluntarily finish or repeat finish requests. '
                        'Use remaining calls to inspect fresh views, correct evidence or try an available recovery/plan. '
                        'An unconfirmed stalk does not authorize a blind grasp or invented failure. '
                        'If movement is blocked, continue read-only inspection and explain the blocker; '
                        'the runner ends at the call budget, user interruption or a real motion safety stop. '
                        'Only completed delivery authorizes finish; no motion was executed by this response.'},[]
            local=getattr(self,'local_targets',None)
            if (getattr(self,'visual_proposals',False) and getattr(self,'centered',None) is not None
                    and self.target_stage is None and not self.grasp_attempted
                    and not self.target_confirmed and not getattr(self,'motion_stop_trace',None)
                    and not getattr(self,'preclose_recovery_failed',False)
                    and not getattr(self,'finish_recovery_reminded',False)
                    and self.max_calls-self.calls>=10 and getattr(self,'station_attempts',0)<12
                    and (observation_adjustment_available(self,frame) or can_review_proposals(local)
                         or unconfirmed_stalk_options(self,frame) or unlisted_inspection_available(self))):
                self.finish_recovery_reminded=True
                return {'episode_ended':False,'status':'recovery_available',
                        'submitted_finish_reason':args['reason'],
                        'remaining_model_calls':self.max_calls-self.calls,
                        'next_action':self.station_status()['next_action'],
                        'instruction':'Evidence remains unresolved, not proof of no ripe fruit. Try the available recovery. This reminder occurs once per episode; finish again if recovery is unsuitable. No motion was executed.'},[]
            self.finished = True
            return {'episode_ended': True, 'reason': args['reason'], 'success_decided_by': 'independent evaluator'}, []
        raise ValueError('Unknown tool')

    def lock_target(self, point, state):
        # Explicit envelope for the single-fruit fixture, not scene object coordinates.
        if any(np.linalg.norm(point-np.asarray(old))<.015 for old in getattr(self,'visual_rejected_targets',[])):
            raise ValueError('This measured neighbourhood already failed preclose verification; select another stalk or finish')
        base=np.asarray(state['base_x_y_yaw'][:2])
        # Full scenes may retain a nearby out-of-reach measurement for parking
        # refinement. The preflight still gates all approach motion.
        radius=.56 if self.e.meta.get('tilted_stalk_prior') else .48
        if not .17 <= point[2] <= .45 or np.linalg.norm(point[:2]-base)>radius:
            raise ValueError('Depth hits ground/background or is outside stalk workspace; keep previous target and choose a visible stalk pixel')
        if self.locked_target is not None:
            jump=float(np.linalg.norm(point-self.locked_target))
            if jump>.015:raise ValueError(f'Target jump {jump:.3f}m rejected; previous target retained')
            # Repeated queries verify consistency, not silently move the target.
        else:self.locked_target=point.copy()

    def deliver_to_basket(self):
        """Known-station motion primitive. No fruit state, welds or teleportation."""
        e=self.e
        if not e.meta.get('harvest_mechanism') or not self.grasp_attempted:
            raise ValueError('Delivery requires a completed grasp in the peduncle scene')
        if not jaw_contact_state(e)['both']:raise ValueError('Cannot deliver: no bilateral external contact')
        if e.meta.get('tilted_stalk_prior') and not self.check_contact_hold()['stable']:
            raise ValueError('Cannot deliver: bilateral contact did not remain stable for 0.5s')
        lost=0;holding=True;phases=[];start_frame=e.frame;active_phase='delivery_start'
        waypoints=[]
        def tick(q,phase):
            nonlocal lost,active_phase
            active_phase=phase
            self.tick(q,phase)
            if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning; delivery stopped')
            lost=0 if jaw_contact_state(e)['both'] else lost+1
            if holding and lost>=4:raise ValueError('Delivery stopped: external jaw contact lost for 0.2s; fruit may have dropped')
        def move(p,grip,n,phase,pitch=0):
            controller=getattr(self,'centered',None)
            q,error=controller.ik(np.asarray(p),grip,pitch) if controller is not None else e.ik(np.asarray(p),grip,approach_pitch=pitch)
            if error>.006:raise ValueError('Delivery waypoint unreachable: '+phase)
            budget=controller.waypoint_ticks(q,n) if controller is not None else {'total_ticks':n}
            waypoint={'phase':phase,'target_tcp_m':np.asarray(p).tolist(),**budget}
            waypoints.append(waypoint)
            for _ in range(budget['total_ticks']):tick(q,phase)
            waypoint['actual_error_m']=float(np.linalg.norm(e.data.site_xpos[e.tcp]-p))
            waypoint['remaining_command_rad']=float(np.max(np.abs(q-e.last_action)))
            if waypoint['actual_error_m']>.008:
                raise ValueError('Delivery waypoint did not settle: '+phase)
            self.approach_pitch=pitch;phases.append(phase)
        def drive_to(y,phase):
            q=e.last_action.copy()
            for _ in range(600):
                error=y-e.base_pose()[1]
                if abs(error)<.004:break
                e.drive(float(np.clip(error,-.04,.04)),0);tick(q,phase)
            else:raise ValueError('Delivery parking timeout')
            e.drive(0,0)
            for _ in range(20):tick(q,phase+'_park')
            phases.append(phase)
        try:
            e.drive(0,0)
            p=e.data.site_xpos[e.tcp].copy()
            move(p+[-.085,0,.05],-.15,75,'delivery_carry',-.35)
            if e.meta.get('basket_mount')=='chassis':
                from mounted_basket import chassis_point, ready_point
                basket=chassis_point(e,e.meta['basket_in_chassis_m'])
                side=chassis_point(e,[.12,-.145,.33])
                move(side,-.15,70,'delivery_aisle_turn',-.6)
                move(basket+[0,0,.20],-.15,65,'delivery_above',-.6)
                move(basket+[0,0,.14],-.15,45,'delivery_approach',-.6)
                move(basket+[0,0,.115],-.15,35,'delivery_lower',-.6)
                holding=False
                move(basket+[0,0,.115],1,45,'delivery_release',-.6)
                move(basket+[0,0,.20],1,55,'delivery_clear',-.6)
                move(side,1,60,'delivery_return',-.6)
                move(ready_point(e),1,65,'delivery_stow')
                observation=None
                if getattr(self,'centered',None) is not None:
                    observation=self.centered.restore_observation()
                    phases.append('delivery_restore_observation')
                return {'status':'executed','stage':'delivery','basket_mount':'chassis','base_travel':False,
                        'phases':phases,'release_commanded':True,'returned_to_ready':True,
                        'observation_pose':observation,'waypoints':waypoints,
                        'success_decided_by':'independent evaluator','robot':robot_state(e)}
            drive_to(float(e.meta['collection_base_y']),'delivery_drive')
            basket=np.asarray(e.meta['basket'])
            move([.10,e.base_pose()[1]+.08,.34],-.15,90,'delivery_aisle_turn')
            move(basket+[0,0,.26],-.15,90,'delivery_above')
            move(basket+[0,0,.14],-.15,55,'delivery_approach')
            move(basket+[0,0,.115],-.15,40,'delivery_lower')
            holding=False
            move(basket+[0,0,.115],1,45,'delivery_release')
            move(basket+[0,0,.23],1,60,'delivery_clear')
            move([.19,e.base_pose()[1]-.04,.34],1,65,'delivery_stow')
            drive_to(float(e.meta['harvest_base_y']),'delivery_return')
            return {'status':'executed','stage':'delivery','phases':phases,
                    'release_commanded':True,'success_decided_by':'independent evaluator','robot':robot_state(e)}
        except ValueError as exc:
            if e.frame>start_frame:
                self.motion_stop_trace={'action':'deliver_to_basket','phase':active_phase,
                    'start_frame':start_frame,'failure_frame':e.frame,'reason':str(exc),
                    'completed_phases':phases,'tracking':getattr(exc,'tracking',None),
                    'waypoints':waypoints}
            raise
        finally:e.drive(0,0)

    def execute_approach(self,stage):
        """Bound command motion and track each segment using robot proprioception only."""
        if getattr(self,'local_targets',None) is not None:self.local_targets.check_target(self.locked_target)
        if getattr(self,'preclose_recovery_failed',False):
            raise ValueError('Visual recovery incomplete; finish episode')
        if getattr(self,'preclose_recoveries',0)>=2:
            raise ValueError('Two preclose visual recoveries exhausted; finish episode')
        e=self.e
        target=self.locked_target.copy()
        controller=getattr(self,'centered',None)
        if controller is not None:
            e.drive(0,0);previous_stage=self.target_stage;self.target_stage='approaching';before_frame=e.frame
            try:
                diagnostics=controller.approach(stage,target,self.maturity_evidence['axis_world'])
            except ValueError as exc:
                if e.frame==before_frame:
                    self.target_stage=previous_stage
                    raise
                diagnostics={'stage':stage,'start_frame':before_frame,'failure_frame':e.frame,
                             'approach_error':str(exc),'approach_tracking':controller.last_tracking}
                try:controller.retreat()
                except ValueError as recovery_exc:
                    self.preclose_recovery_failed=True
                    diagnostics.update(retreat_completed=False,retreat_error=str(recovery_exc),
                                       retreat_tracking=controller.last_tracking)
                    stopped=ValueError('Centered approach/retreat incomplete; finish episode')
                    stopped.approach_trace=diagnostics
                    raise stopped from exc
                self.target_stage=None;self.locked_target=None;self.target_camera=None
                self.target_confirmed=False;self.maturity_evidence=None;self.target_reachability=None
                diagnostics.update(retreat_completed=True,return_frame=e.frame)
                stopped=ValueError('Centered approach stopped and withdrew; reobserve target: '+str(exc))
                stopped.approach_trace=diagnostics
                raise stopped from exc
            self.target_stage=stage;self.approach_pitch=0
            return {'stage':stage,'status':'executed','measured_target_m':target.tolist(),
                    **diagnostics,'robot':robot_state(e)}
        destination=target-np.array([.055,0,0]) if stage=='pregrasp' else target
        start=e.data.site_xpos[e.tcp].copy()
        segmented=stage=='grasp' and e.meta.get('tilted_stalk_prior',False)
        final_q,final_ik_error=e.ik(destination,1,approach_pitch=0)
        if final_ik_error>.006:raise ValueError('Measured target unreachable')
        self.last_approach_trace=[]
        e.drive(0,0);self.target_stage='approaching'
        reason=None;residual=None
        if segmented:
            # Preserve the final joint solution and original slew direction.
            # Bound each command's FK displacement instead of solving a new wrist
            # posture at every Cartesian waypoint or pausing against the stalk.
            command=e.last_action.copy();blocked=0;settled=0
            for index in range(1,181):
                if not blocked:
                    proposed=np.clip(final_q,command-.04,command+.04)
                    origin=command_tcp(e,command)
                    delta=proposed-command
                    if np.linalg.norm(command_tcp(e,proposed)-origin)>.005:
                        low,high=0.,1.
                        for _ in range(12):
                            fraction=(low+high)/2
                            if np.linalg.norm(command_tcp(e,command+fraction*delta)-origin)<=.005:low=fraction
                            else:high=fraction
                        proposed=command+low*delta
                    command=proposed
                expected=command_tcp(e,command)
                self.tick(command,'visual_grasp')
                if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning; episode stopped')
                residual=float(np.linalg.norm(e.data.site_xpos[e.tcp]-expected))
                blocked=blocked+1 if residual>.008 else 0
                self.last_approach_trace.append({'segment':index,'waypoint_m':expected.tolist(),
                    'tcp_error_m':residual,'actual_tcp_m':e.data.site_xpos[e.tcp].tolist(),
                    'holding_for_tracking':blocked>0})
                if blocked>=3:
                    reason=f'tracking error {residual:.4f}m persisted for 3 control ticks';break
                if np.max(np.abs(command-final_q))<1e-8:settled+=1
                if settled>=55:break
            else:reason='segmented approach timeout'
            residual=float(np.linalg.norm(e.data.site_xpos[e.tcp]-destination))
            if reason is None and residual>.008:reason=f'final TCP error {residual:.4f}m'
        else:
            for _ in range(55):
                self.tick(final_q,'visual_'+stage)
                if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning; episode stopped')
            residual=float(np.linalg.norm(e.data.site_xpos[e.tcp]-destination))
            self.last_approach_trace.append({'segment':1,'waypoint_m':destination.tolist(),
                                            'tcp_error_m':residual,'ik_error_m':final_ik_error})
            if residual>.008:reason=f'TCP error {residual:.4f}m'
        if reason:
            retreat=target-np.array([.055,0,0])
            back,error=e.ik(retreat,1,approach_pitch=0)
            if error>.006:raise ValueError('Approach blocked; retreat unreachable; finish episode')
            for _ in range(55):
                self.tick(back,'approach_retreat')
                if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning during retreat')
            if np.linalg.norm(e.data.site_xpos[e.tcp]-retreat)>.008:
                raise ValueError('Approach blocked and retreat incomplete; finish episode')
            self.target_stage=None;self.locked_target=None;self.target_camera=None
            self.target_confirmed=False;self.maturity_evidence=None
            self.target_reachability=None
            error=ValueError('Approach blocked: '+reason+'; withdrew open and cleared target. Reselect stalk; do not repeat grasp blindly')
            error.approach_trace=self.last_approach_trace
            raise error
        self.target_stage=stage;self.approach_pitch=0
        return {'stage':stage,'status':'executed','measured_target_m':target.tolist(),
                'tcp_error_m':float(np.linalg.norm(e.data.site_xpos[e.tcp]-destination)),
                'approach_trace':self.last_approach_trace,'robot':robot_state(e)}

    def apply_parking_trial(self):
        if getattr(self,'fixed_base',False):raise ValueError('Fixed harvest base: parking disabled')
        trial=self.parking_trial
        if not trial['enabled'] or trial['status']!='pending':
            raise ValueError('Fixed parking trial is not pending')
        before=self.e.base_pose().copy()
        trial.update(status='attempting',before_base_pose=before.tolist())
        try:
            result=self.search_row(1,distance=.04)
        except Exception as exc:
            trial.update(status='failed',reason=str(exc))
            raise
        trial.update(status='applied',after_base_pose=self.e.base_pose().tolist(),
                     actual_delta_y_m=float(self.e.base_pose()[1]-before[1]))
        result.update(parking_trial=trial.copy(),requires_relocalization=True,
                      pregrasp_executed=False,
                      note='Fixed +4cm parking completed; no grasp-height offset. Old target and images cleared. Reobserve and localize before pregrasp.')
        return result

    def plan_parking(self):
        controller=getattr(self,'centered',None)
        if controller is not None:
            try:
                p=controller.plan(self.locked_target,self.maturity_evidence['axis_world'])
                current={'reachable':p['max_ik_error_m']<=.006,'max_ik_error_m':p['max_ik_error_m']}
            except ValueError as exc:current={'reachable':False,'reason':str(exc)}
            return {'mode':'centered-local-first','current':current,'candidates':[],
                    'recommended_delta_y_m':None,'next_action':'approach_target' if current['reachable'] else 'select next local candidate in place'}
        if getattr(self,'local_targets',None) is not None:
            current=approach_reachability(self.e,self.locked_target)
            if current['reachable'] or self.local_targets.active is not None:
                return {'mode':'local-first','current':current,'candidates':[],
                        'recommended_delta_y_m':None,
                        'next_action':'approach_target' if current['reachable'] else 'select next local candidate in place'}
        if getattr(self,'fixed_base',False):
            current=approach_reachability(self.e,self.locked_target)
            return {'mode':'fixed-harvest-base','current':current,'candidates':[],
                    'recommended_delta_y_m':None,
                    'next_action':'approach_target' if current['reachable'] else 'clear_target and select another visible ripe target or finish'}
        if not getattr(self,'target_aware_parking',False):
            return parking_plan(self.e,self.locked_target)
        plan=optimized_parking_plan(self.e,self.locked_target)
        anchor=getattr(self,'parking_anchor',None)
        # Same local measurement after a successful move: permit approach when
        # reachable instead of chasing a new noisy optimum. This is spatial
        # correspondence only, never an oracle fruit identity.
        if (anchor and np.linalg.norm(self.locked_target-np.asarray(anchor['target_m'])) <= .008
                and abs(self.e.base_pose()[1]-anchor['actual_parking_y_m']) <= .006
                and plan['current']['reachable']):
            plan.update(recommended_delta_y_m=None,selected=plan['current'],next_action='approach_target',
                        held_previous_parking=True)
        if (getattr(self,'search_steps',0)>=4 or self.max_calls-self.calls<10) and plan['current']['reachable']:
            plan.update(recommended_delta_y_m=None,selected=plan['current'],next_action='approach_target',
                        move_budget_exhausted=True)
        return plan

    def refine_parking(self,plan=None):
        if getattr(self,'fixed_base',False):raise ValueError('Fixed harvest base: parking disabled')
        self.require_local_search()
        if getattr(self,'parking_trial',{}).get('enabled'):
            raise ValueError('Adaptive parking disabled in fixed +4cm trial; use search_row or finish')
        if self.locked_target is None or not self.target_confirmed:
            raise ValueError('Confirm a measured target before refining parking')
        if self.grasp_attempted or self.target_stage is not None:
            raise ValueError('Refine parking only before approach/grasp')
        if (not self.maturity_evidence or not self.maturity_evidence['passed']
                or not np.allclose(self.maturity_evidence.get('target_m',[]),self.locked_target,atol=1e-6,rtol=0)):
            raise ValueError('Visual maturity approval required before refining parking')
        plan=self.plan_parking() if plan is None else plan
        self.target_reachability=plan
        delta=plan['recommended_delta_y_m']
        if delta is None:
            return {'status':'rejected','reason':'No kinematic parking refinement needed or available','reachability':plan}
        target=self.locked_target.copy()
        before=self.e.base_pose().copy()
        if getattr(self,'target_aware_parking',False):
            print(f'[VISTA] 目标停车：y={before[1]:.3f}m → {before[1]+delta:.3f}m，移车 {delta*100:+.1f}cm；停车后重新定位',flush=True)
        result=self.search_row(1 if delta>0 else -1, distance=abs(delta))
        if getattr(self,'target_aware_parking',False):
            self.parking_anchor={'target_m':target.tolist(),'actual_parking_y_m':float(self.e.base_pose()[1])}
            self.parking_history.append({'plan':plan,'before_base_pose':before.tolist(),
                                         'after_base_pose':self.e.base_pose().tolist()})
        result['parking_plan']=plan
        result['requires_relocalization']=True
        result['pregrasp_executed']=False
        return result

    def search_row(self,direction, distance=None):
        """Bounded straight-row search using robot state only, never fruit truth."""
        if getattr(self,'fixed_base',False):raise ValueError('Fixed harvest base: row search disabled')
        self.require_local_search(observation_only=True)
        e=self.e
        from mobile_scene import search_limits
        step, radius = search_limits(e.meta)
        if distance is None:distance=step
        if direction not in (-1,1) or not np.isfinite(distance) or not .001<=distance<=step:
            raise ValueError('Invalid bounded row displacement')
        if not e.meta.get('tilted_stalk_prior') or e.meta.get('single_fruit_fixture'):
            raise ValueError('Row search requires the full diverse greenhouse')
        if self.grasp_attempted or self.target_stage is not None:
            raise ValueError('Search only before approach/grasp; cannot drive with an extended or holding gripper')
        if self.search_steps>=4 or self.max_calls-self.calls<10:
            raise ValueError('Search budget exhausted; reserve 10 calls for selection, grasp and delivery; finish if no target')
        base=e.base_pose().copy()
        initial=float(e.meta['harvest_base_y'])
        goal=float(base[1]+distance*direction)
        if abs(goal-initial)>radius+.005 or abs(base[0])>.015 or abs(base[2])>.03:
            raise ValueError(f'Outside straight-row search envelope (+/-{radius:.2f}m from start); choose opposite direction or finish')
        observation_recovery=None
        if self.local_targets is not None and not self.local_targets.can_search:
            from perception_recovery import uncertain_viewpoint_search_available
            evidence=getattr(self,'unlisted_search_evidence',None)
            if evidence:
                observation_recovery={'kind':'unlisted_fruit_viewpoint_search','evidence':evidence,
                    'original_selection':self.local_targets.status(),
                    'note':'Missing-fruit claim retained; RGB-D places reported bodies outside local envelope. Observation movement only.'}
            elif uncertain_viewpoint_search_available(self):
                observation_recovery={'kind':'uncertain_viewpoint_search',
                    'frame_id':next(reversed(self.proposal_frames)),
                    'original_selection':self.local_targets.status(),
                    'previous_survey':self.local_targets.survey_review,
                    'note':'Unknown regions deferred for a new viewpoint, not absent or unripe; no grasp permission and no synthetic localization failure.'}
            else:
                observation_recovery={'kind':'edge_fragment_viewpoint_search',
                    'frame_id':self.edge_viewpoint_evidence['frame_id'],
                    'fragments':self.edge_viewpoint_evidence['fragments'],
                    'original_selection':self.local_targets.status(),
                    'note':'Unresolved fragments retained as uncertain; moving to reobserve, not declaring absence or permitting grasp.'}
            self.perception_recoveries.append(observation_recovery)
            self.log(observation_recovery['kind'],**{k:v for k,v in observation_recovery.items() if k!='kind'})
        # Invalidate before motion, including failed/partial moves and stale crops.
        self.locked_target=None;self.target_camera=None;self.target_confirmed=False
        self.maturity_evidence=None;self.crops.clear();self.frames.clear();self.depth_frames.clear()
        self.target_reachability=None
        self.search_steps+=1
        def tick(q,phase):
            self.tick(q,phase)
            if any(w.number for w in e.data.warning):raise RuntimeError('Physics warning during search')
        def move(point,phase):
            q,error=e.ik(np.asarray(point),1,approach_pitch=0)
            if error>.006:raise ValueError('Search stow waypoint unreachable')
            for _ in range(70):tick(q,phase)
            if np.linalg.norm(e.data.site_xpos[e.tcp]-point)>.008:
                raise ValueError('Search stow blocked; no further motion allowed')
            return q
        self.target_stage='searching'
        try:
            e.drive(0,0)
            stow=np.array([.19,base[1]-.04,.34])
            keep_stowed=e.meta.get('initial_arm_pose')=='search_stow'
            if keep_stowed and e.state()[5]>.9 and np.linalg.norm(e.data.site_xpos[e.tcp]-stow)<.004:
                q=e.last_action.copy()
            else:q=move(stow,'search_stow')
            for _ in range(240):
                pose=e.base_pose()
                if abs(pose[0]-base[0])>.01 or abs(pose[2]-base[2])>.03:
                    raise ValueError('Search deviated from straight aisle; stopped')
                error=goal-pose[1]
                if abs(error)<.004:break
                e.drive(float(np.clip(error,-.04,.04)),0)
                tick(q,'search_drive')
            else:raise ValueError('Search parking timeout')
            e.drive(0,0)
            for _ in range(20):tick(q,'search_park')
            if not keep_stowed:
                move([.19,e.base_pose()[1]-.15,.33],'search_observe_pose')
            elif np.linalg.norm(e.data.site_xpos[e.tcp]-[.19,e.base_pose()[1]-.04,.34])>.008:
                raise ValueError('Search observation stow did not settle')
            self.target_stage=None;self.approach_pitch=0
            self.station_attempts=0;self.station_failures=0
            if not hasattr(self,'search_history'):self.search_history=[]
            self.search_history.append({'direction':direction,'from_y_m':float(base[1]),
                'to_y_m':float(e.base_pose()[1]),'actual_delta_y_m':float(e.base_pose()[1]-base[1])})
            if getattr(self,'local_targets',None) is not None:
                self.local_targets=LocalTargets()
                self.local_targets.allow_uncertain_deferral=getattr(self,'require_delivery_before_finish',False)
            return {'status':'executed','stage':'search','search_steps':self.search_steps,
                    'observation_recovery':observation_recovery,
                    'max_search_steps':4,'requested_distance_m':distance,'actual_delta_y_m':float(e.base_pose()[1]-base[1]),
                    'search_radius_m':radius,'robot':robot_state(e),
                    'note':'Stopped at new view. Old targets and crops invalidated; use next observation to select a fully red fruit and its stalk.'}
        finally:e.drive(0,0)

    def check_maturity(self,frame,target,axis):
        depth=self.depth_frames.get((frame,'harvest_close'))
        rgb=self.frames.get(frame,{}).get('harvest_close')
        if depth is None or rgb is None:
            raise ValueError('Current close RGB-D required for visual maturity check')
        with np.load(depth) as sample, Image.open(rgb) as image:
            return assess_maturity(target,axis,sample['depth'],sample,image)

    def confirm_target(self,point,state,camera):
        if self.locked_target is not None and np.linalg.norm(point-self.locked_target)>.008:
            raise ValueError('Views disagree by more than 8mm; no approach allowed. Inspect or clear_target and select again.')
        self.lock_target(point,state)
        if self.target_camera is None:self.target_camera=camera
        elif camera!=self.target_camera:self.target_confirmed=True

    def run(self):
        pool = ThreadPoolExecutor(max_workers=1)
        consecutive_service_failures = 0
        self.model_service_failures = []
        try:
            for self.calls in range(1, self.max_calls+1):
                self.e.drive(0, 0)
                # Settle command before taking the decision snapshot.
                for _ in range(10): self.tick(self.e.last_action, 'decision_settle')
                frame, state, observation = self.observe()
                observation['content'].append({'type':'text','text':json.dumps({
                    'current_model_call':self.calls,'max_model_calls':self.max_calls,
                    'remaining_calls_after_this_response':self.max_calls-self.calls,
                    'budget_instruction':('Reserve calls for pull, visual separation confirmation, delivery and finish. Centered grasp needs 4 further calls after closing. After visual confirmation, prioritize delivery.' if getattr(self,'centered',None) is not None else 'Reserve 3 further calls after closing for pull, delivery and finish; visually verify detachment before delivery.')})})
                self.messages.append(observation)
                started = time.monotonic()
                self.log('request', call=self.calls, basis_frame_id=frame)
                future = pool.submit(self.client.call, list(self.messages))
                next_notice = started + 10
                while not future.done():
                    if time.monotonic()-started > self.max_wait: raise ModelServiceError('Model deadline exceeded; no actions executed')
                    if time.monotonic() >= next_notice:
                        print(f'[VISTA] 等待模型回复 {time.monotonic()-started:.0f}秒；仿真继续保持姿态', flush=True)
                        next_notice = time.monotonic() + 10
                    self.tick(self.e.last_action, 'model_wait', realtime=True)
                    self.wait_ticks += 1
                try:
                    message, usage = future.result()
                except ModelServiceError as exc:
                    consecutive_service_failures += 1
                    retry = exc.retryable and consecutive_service_failures < 3 and self.calls < self.max_calls
                    delay = 2 ** consecutive_service_failures if retry else 0
                    failure = dict(call=self.calls, basis_frame_id=frame, reason=str(exc),
                                   consecutive_failures=consecutive_service_failures,
                                   retry_scheduled=retry, retry_delay_s=delay)
                    self.model_service_failures.append(failure)
                    self.log('model_service_failure', **failure)
                    if not retry:
                        raise
                    # No response was accepted and no action was dispatched. Replace
                    # this failed observation next iteration; obtain fresh images and
                    # budget metadata instead of executing an eventual stale response.
                    self.messages.pop()
                    print(f'[VISTA] {exc}；{delay}秒后重新观察并重试，重试计入 {self.max_calls} 次上限。', flush=True)
                    until = time.monotonic() + delay
                    while time.monotonic() < until:
                        self.tick(self.e.last_action, 'model_wait', realtime=True)
                        self.wait_ticks += 1
                    continue
                consecutive_service_failures = 0
                self.messages.append(message)
                self.log('response', call=self.calls, message=message, usage=usage,
                         wall_latency_s=time.monotonic()-started)
                calls = message.get('tool_calls') or []
                if len(calls) != 1:
                    result = {'status': 'rejected', 'reason': 'Exactly one tool call required; no action executed'}
                    for call in calls:
                        self.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)})
                    self.messages.append({'role': 'user', 'content': '请按工具协议返回恰好一个工具调用。需要多步动作请放在一个play.actions数组中；本轮未执行任何动作。'})
                    self.log('receipt', call=self.calls, result=result, received_calls=len(calls))
                    continue
                call = message['tool_calls'][0]
                try:
                    args = json.loads(call['function']['arguments'], parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
                    result, images = self.dispatch(call['function']['name'], args, frame, state, time.monotonic()-started)
                except (ValueError, KeyError, TypeError) as exc:
                    result, images = {'status': 'rejected', 'reason': str(exc)}, []
                    if hasattr(exc,'approach_trace'):result['approach_trace']=exc.approach_trace
                    if isinstance(exc,TrackingError) and not getattr(self,'motion_stop_trace',None):
                        self.motion_stop_trace={'phase':getattr(exc,'phase','unknown'),
                            'failure_frame':self.e.frame,'reason':str(exc),'tracking':exc.tracking}
                if getattr(self,'motion_stop_trace',None):result={'status':'stopped','motion_stop':self.motion_stop_trace,'tool_result':result}
                self.log('receipt', call=self.calls, result=result)
                if getattr(self,'motion_stop_trace',None):
                    raise MotionSafetyStop('Motion interrupted; '+json.dumps(self.motion_stop_trace))
                if getattr(self,'preclose_recovery_failed',False):
                    raise MotionSafetyStop('Motion recovery incomplete; '+str(result))
                if call['function']['name'] == 'play':
                    self.receipts.append({'call': self.calls, 'basis_frame_id': frame, 'result': result})
                self.messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)})
                if images: self.messages.append({'role': 'user', 'content': [{'type':'text','text':json.dumps(result)}, *images]})
                if self.finished: return
            raise CallBudgetExhausted('Model call budget exhausted')
        except TrackingError as exc:
            self.motion_stop_trace={'phase':getattr(exc,'phase','unknown'),
                'failure_frame':self.e.frame,'reason':str(exc),'tracking':exc.tracking}
            self.log('motion_stop',**self.motion_stop_trace)
            raise MotionSafetyStop('Tracking protection; '+json.dumps(self.motion_stop_trace)) from exc
        finally:
            self.e.drive(0, 0)
            pool.shutdown(wait=False, cancel_futures=True)


def evaluate(e, rows):
    """Hidden task evaluation only; never included in model observations."""
    inside = []
    for i in range(len(e.fruit_bodies)):
        rel = e.data.body('basket').xmat.reshape(3,3).T@(e.fruit_pos(i)-e.basket_position())
        dof = e.fruit_dofs[i]
        half=e.meta.get('basket_acceptance_half_xy_m',[.066,.048])
        if e.meta['ripe'][i] and not e.attached[i] and abs(rel[0]) < half[0] and abs(rel[1]) < half[1] and .012 < rel[2] < .05 and np.linalg.norm(e.data.qvel[dof:dof+3]) < .05 and not e.contacts(i): inside.append(i)
    if e.meta.get('basket_mount')=='chassis':
        from mounted_basket import ready_point
        delivery=[r for r in rows if r['phase'].startswith('delivery_')]
        released=any(r['phase']=='delivery_release' for r in delivery)
        parked=bool(delivery and max(np.linalg.norm(np.asarray(r['base_pose'][:2])-delivery[0]['base_pose'][:2]) for r in delivery)<.012)
        returned=bool(np.linalg.norm(e.data.site_xpos[e.tcp]-ready_point(e))<.012 and parked and any(r['phase']=='delivery_stow' for r in delivery))
        if e.meta.get('centered_grasp'):
            returned=bool(returned and any(r['phase']=='delivery_restore_observation' for r in delivery)
                          and abs(e.state()[4]-e.meta.get('observation_roll_rad',float('inf')))<.015)
        detached=np.flatnonzero(~e.attached).tolist()
        single=len(inside)==len(detached)==1 and inside==detached
        return {'ripe_fruit_in_basket':inside,'detached_fruit_count':len(detached),'single_fruit_delivered':single,
                'visited_collection':False,'placement_executed':released,'delivery_base_parked':parked,
                'returned_to_harvest':returned,'returned_to_ready':returned,
                'success':bool(single and released and returned)}
    visited = any(r['base_pose'][1] > .23 for r in rows)
    returned = np.linalg.norm(e.base_pose()[:2]-np.asarray(e.meta.get('initial_base_pose',[0.,0.,0.]))[:2]) < .012
    return {'ripe_fruit_in_basket': inside, 'visited_collection': visited, 'returned_to_harvest': bool(returned),
            'success': bool(inside and visited and returned)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-file', type=Path, default=ROOT/'.env')
    p.add_argument('--out', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-calls', type=int, default=32)
    p.add_argument('--task', help='Override task for a bounded interface check or another simulation instruction')
    p.add_argument('--deep-thinking', action='store_true', help='Use configured thinking mode instead of default low-latency mode')
    p.add_argument('--peduncle-test', action='store_true', help='RGB-D stalk picking in the restored full greenhouse')
    p.add_argument('--single-fruit', action='store_true', help='Explicitly use the simplified single-fruit regression fixture with --peduncle-test')
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--mobile-layout',action='store_true',help='Variable fruit clusters and randomized aisle start; requires --allow-row-search')
    p.add_argument('--mobile-layout-level',choices=['moderate','stress'],default='moderate',help='Moderate nearby opportunities (default), or original wide-gap stress layout')
    p.add_argument('--visual-proposals',action='store_true',help='Experimental numbered RGB-D body proposals; model reviews maturity without guessing body pixels')
    p.add_argument('--centered-grasp',action='store_true',help='Experimental calibrated jaw centering, visual wrist alignment, roll-preserving delivery and observation-pose return')
    p.add_argument('--preclose-visual-check',action='store_true',help='Experimental fresh RGB-D stalk support before closing; bounded open retreat on missing evidence')
    base_mode=p.add_mutually_exclusive_group()
    base_mode.add_argument('--fixed-base',action='store_true',help='Default full-scene mode: fixed initial pose, no search/parking during harvest')
    base_mode.add_argument('--allow-row-search',action='store_true',help='Opt into the previous moving search mode')
    p.add_argument('--target-aware-parking',action='store_true',help='Score bounded parking candidates from the visually measured mature stalk; reobserve after driving')
    p.add_argument('--pregrasp-park-plus40',action='store_true',help='Full-scene trial: once, after target confirmation, stow and park +4cm before reobserving; no grasp-height offset')
    p.add_argument('--record-images', action='store_true', help='Save synchronized policy-camera RGB (two for the chassis basket) at every 20Hz control tick (slow on CPU renderer)')
    p.add_argument('--video', action='store_true', help='Save 5fps GIF; adds rendering overhead')
    p.add_argument('--check-config', action='store_true', help='Validate local config without network or simulation')
    p.add_argument('--check-render', action='store_true', help='Test camera rendering with optional viewer; no model calls')
    a = p.parse_args()
    if a.mobile_layout_level!='moderate' and not a.mobile_layout:p.error('--mobile-layout-level stress requires --mobile-layout')
    if a.mobile_layout and (not a.peduncle_test or a.single_fruit or not a.allow_row_search or a.fixed_base or a.target_aware_parking or a.pregrasp_park_plus40):p.error('--mobile-layout requires full --peduncle-test --allow-row-search; excludes parking experiments')
    if a.visual_proposals and (not a.peduncle_test or a.single_fruit or a.pregrasp_park_plus40):p.error('--visual-proposals requires local-first full --peduncle-test')
    if a.centered_grasp and (not a.peduncle_test or a.single_fruit or a.pregrasp_park_plus40 or a.target_aware_parking or a.preclose_visual_check):p.error('--centered-grasp requires full --peduncle-test and excludes other grasp/parking experiments')
    if a.single_fruit and not a.peduncle_test:p.error('--single-fruit requires --peduncle-test')
    if a.preclose_visual_check and (not a.peduncle_test or a.single_fruit):p.error('--preclose-visual-check requires full --peduncle-test')
    if a.pregrasp_park_plus40 and (not a.peduncle_test or a.single_fruit):p.error('--pregrasp-park-plus40 requires --peduncle-test without --single-fruit')
    if a.target_aware_parking and (a.pregrasp_park_plus40 or not a.peduncle_test or a.single_fruit):p.error('--target-aware-parking requires full --peduncle-test and excludes --pregrasp-park-plus40')
    if a.fixed_base and (a.target_aware_parking or a.pregrasp_park_plus40):p.error('--fixed-base excludes parking experiments')
    if a.fixed_base and (not a.peduncle_test or a.single_fruit):p.error('--fixed-base requires full --peduncle-test')
    a.fixed_base=bool(a.peduncle_test and not a.single_fruit and not (a.allow_row_search or a.target_aware_parking or a.pregrasp_park_plus40))
    if not 1 <= a.max_calls <= 128: p.error('--max-calls must be 1–128')
    if a.check_render:
        check_render(a.viewer, a.seed)
        return
    client = Client(a.env_file)
    if not a.deep_thinking:
        client.options['thinking'] = {'type':'disabled'}
        client.options.pop('reasoning_effort', None)
        client.options['max_tokens'] = min(client.options['max_tokens'], 2048)
    if a.check_config:
        print(json.dumps({'configured': True, 'model': client.model, 'credentials_printed': False})); client.close(); return
    e = None; viewer = None; r = None; error = None; session = None; stop_reason = None
    print(f'[VISTA] 模型：{client.model}；正在初始化场景和相机……', flush=True)
    try:
        if a.peduncle_test:
            from peduncle import PeduncleEnv
            e = PeduncleEnv(a.seed, single=a.single_fruit, fixed_base=a.fixed_base, mobile_layout=a.mobile_layout, mobile_layout_level=a.mobile_layout_level, render_quality='fast',
                            initial_arm_pose='search_stow' if a.centered_grasp and not a.single_fruit else 'scene')
        else:
            e = StrawberryEnv(a.seed, render_quality='fast')
        out = a.out or ROOT/'recordings'/'episodes'/f'{time.strftime("%Y%m%d-%H%M%S")}-seed-{a.seed}-{time.time_ns()}'
        r = Recorder(e, out, a.record_images, a.video)
        print(f'[VISTA] 日志目录：{out}', flush=True)
        if a.fixed_base:print('[VISTA] 固定采摘：初始底盘 y=0，不搜索移车、不优化停车；车前成熟度随机，成功摘下后仍送筐返回',flush=True)
        (out/'system_prompt.txt').write_text(project_source('vista_prompt.txt').read_text())
        sources = ['project_paths.py', 'vista.py', 'vista_prompt.txt', 'peduncle.py', 'peduncle_prompt.txt', 'env.py', 'run.py', 'recording_quality.py', 'visual_maturity.py', 'preclose_vision.py', 'local_targets.py', 'target_parking.py', 'scene.py', 'chassis.py', 'mounted_basket.py', 'peduncle_mobile_prompt.txt', 'natural_assets.py', 'scene_details.py', 'grasp_pose.py', 'centered_grasp.py', 'visual_proposals.py', 'stalk_proposals.py', 'detachment_review.py', 'perception_recovery.py', 'proposal_identity.py', 'mobile_scene.py', 'unlisted_fruit_search.py', 'search_guidance.py']
        (out/'run-config.json').write_text(json.dumps({
            'seed': a.seed, 'max_calls': a.max_calls, 'model': client.model, 'task': a.task,
            'pregrasp_park_plus40':a.pregrasp_park_plus40,'target_aware_parking':a.target_aware_parking,'fixed_base':a.fixed_base,
            'preclose_visual_check':a.preclose_visual_check,'mobile_layout':a.mobile_layout,'mobile_layout_level':a.mobile_layout_level,
            'centered_grasp':a.centered_grasp,'visual_proposals':a.visual_proposals,
            'finish_policy':'delivery_or_call_budget' if a.centered_grasp and a.visual_proposals else 'model_finish',
            'initial_arm_pose':e.meta.get('initial_arm_pose','scene'),
            'local_first_selection':bool(a.peduncle_test and not a.single_fruit and not a.pregrasp_park_plus40),
            'scene_mode': 'single-fruit' if a.single_fruit else 'full-greenhouse',
            'basket_mount':e.meta.get('basket_mount','ground'),'policy_cameras':r.cameras,
            'wrist_camera_profile':e.meta.get('wrist_camera_profile','legacy-gripper-v1'),
            'scene_version':e.meta['scene_version'], 'diversity_profile':e.meta.get('diversity_profile','legacy'),
            'model_options': client.options, 'quality': 'fast', 'control_hz': 20,
            'mujoco_gl': os.environ.get('MUJOCO_GL'),
            'camera_recording': '20Hz' if a.record_images else 'decision_snapshots_only',
            'source_sha256': {name: hashlib.sha256(project_source(name).read_bytes()).hexdigest() for name in sources}
        }, indent=2))
        snapshot=out/'source';snapshot.mkdir()
        for name in sources:(snapshot/name).write_bytes(project_source(name).read_bytes())
        if a.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(e.model, e.data)
            viewer.opt.geomgroup[3:5] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            viewer.cam.lookat[:] = [.12,e.base_pose()[1],.2]; viewer.cam.distance = 1.2; viewer.cam.azimuth = 145; viewer.cam.elevation = -25
        session = Session(e, r, client, max_calls=a.max_calls, viewer=viewer,pregrasp_park_plus40=a.pregrasp_park_plus40,target_aware_parking=a.target_aware_parking,fixed_base=a.fixed_base,preclose_visual_check=a.preclose_visual_check,centered_grasp=a.centered_grasp,visual_proposals=a.visual_proposals)
        (out/'system_prompt.txt').write_text(session.messages[0]['content'])
        if a.task:
            session.messages[0]['content'] += '\n本次具体任务覆盖默认采摘目标：\n' + a.task
            (out/'system_prompt.txt').write_text(session.messages[0]['content'])
        session.run()
        stop_reason = 'model_finish'
    except CallBudgetExhausted:
        stop_reason = 'call_budget_exhausted'
        print('[VISTA] 已达到本回合请求上限，停车并保存记录；任务未因此判为成功。', flush=True)
    except MotionSafetyStop as exc:
        error=f'{type(exc).__name__}: {exc}'
        stop_reason='motion_stopped'
        print('[VISTA] 运动保护停止，保存失败记录；不再推进仿真或请求模型。',flush=True)
    except ModelServiceError as exc:
        error=f'{type(exc).__name__}: {exc}'
        stop_reason='model_service_error'
        print('[VISTA] 模型服务请求未恢复，停车并保存记录；本回合未完成。',flush=True)
    except KeyboardInterrupt:
        error = 'KeyboardInterrupt: user interrupted episode'
        stop_reason = 'interrupted'
        print('[VISTA] 用户中断，保存已提交的控制步。',flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        stop_reason = 'error'
        raise
    finally:
        try:
            if r:
                e.drive(0, 0)
                if viewer: viewer.close(); viewer = None
                # Independent settling check; no model or scripted harvesting fallback.
                if stop_reason not in ('interrupted','error','motion_stopped','model_service_error'):
                    for _ in range(10): r.tick(e.last_action, 'terminal_settle')
                report = r.finish()
                report.update(source='VISTA-style vision and tool loop', formal_training_demo=False,
                              model=client.model, error=error, **evaluate(e, r.rows),
                              task_override=a.task,
                              stop_reason=stop_reason,
                              episode_completed=bool(session and session.finished and error is None),
                              model_calls=session.calls if session else 0,
                              model_service_failures=getattr(session,'model_service_failures',[]),
                              parking_trial=session.parking_trial if session else None,
                              target_aware_parking=a.target_aware_parking,
                              preclose_visual_check=a.preclose_visual_check,
                              centered_grasp=a.centered_grasp,visual_proposals=a.visual_proposals,
                              local_selection=session.local_targets.status() if session and session.local_targets is not None else None,
                              preclose_recoveries=session.preclose_recoveries if session else 0,
                              preclose_recovery_failed=session.preclose_recovery_failed if session else False,
                              motion_stop_trace=getattr(session,'motion_stop_trace',None),
                              staged_pull=getattr(session,'pull_completed',None),
                              perception_recoveries=getattr(session,'perception_recoveries',[]),
                              visual_detachment_reviews=getattr(session,'detachment_reviews',[]),
                              fixed_base=a.fixed_base,
                              parking_history=session.parking_history if session else [],
                              search_steps=session.search_steps if session else 0,
                              base_drive_duration_s=sum(bool(np.any(row['base_command'])) for row in r.rows)*e.dt,
                              base_path_length_m=float(np.linalg.norm(np.diff(np.asarray([row['base_pose'][:2] for row in r.rows]).reshape(-1,2),axis=0),axis=1).sum()),
                              model_wait_ticks=session.wait_ticks if session else 0,
                              full_rate_camera_recording=a.record_images)
                from recording_quality import atomic_json, audit_recording
                atomic_json(r.out/'report.json',report)
                audit_recording(r.out)
                try:
                    from record_catalog import register_episode
                    register_episode(r.out)
                except Exception as catalog_error:
                    print(f'[VISTA] 原始数据已保存，记录列表更新失败：{catalog_error}', flush=True)
                print(json.dumps({'out': str(r.out), 'harvest_success': report['success'],
                                  'episode_completed': report['episode_completed'], 'stop_reason': stop_reason, 'error': error}))
        finally:
            if viewer: viewer.close()
            if e: e.close()
            client.close()


def check_render(with_viewer, seed):
    """Exercise both camera sizes, context switching, and post-viewer final render."""
    e = StrawberryEnv(seed, render_quality='fast')
    viewer = None
    try:
        if with_viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(e.model, e.data)
        count = 0
        for _ in range(3):
            if viewer:
                with viewer.lock(): e.step(e.last_action)
                viewer.sync()
            else: e.step(e.last_action)
            for camera in ('top', 'wrist'):
                pixels = e.render(camera, 480, 360)
                assert pixels.shape == (360, 480, 3) and pixels.std() > 1
                depth, calibration = e.depth_view(camera, 480, 360)
                assert depth.shape == (360,480) and np.isfinite(depth).all()
                count += 1
        if viewer: viewer.close(); viewer = None
        pixels = e.render('detail', 960, 720)
        assert pixels.shape == (720, 960, 3) and pixels.std() > 1
        print(json.dumps({'render_check': 'passed', 'viewer': with_viewer,
                          'mujoco_gl': os.environ.get('MUJOCO_GL'),
                          'camera_frames': count, 'depth_frames': count, 'post_viewer_render': True, 'model_calls': 0}))
    finally:
        if viewer: viewer.close()
        e.close()


if __name__ == '__main__': main()
