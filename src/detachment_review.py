"""Validate explicit camera evidence supplied by the visual decision model.

This is a protocol gate, not an automatic detachment detector. No scene truth.
"""
import numpy as np


def gripper_context_region(tcp,calibration,image_size):
    """Camera projection of robot workspace context, not a fruit identity box."""
    w,h=image_size
    offsets=np.array([[x,y,z] for x in (-.055,.055) for y in (-.055,.055) for z in (-.065,.065)])
    points=np.asarray(tcp)+offsets
    local=(points-np.asarray(calibration['position']))@np.asarray(calibration['rotation'])
    if not np.isfinite(local).all() or np.any(local[:,2]>=-.01):return None
    focal=h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    uv=np.column_stack((w/2-.5+focal*local[:,0]/-local[:,2],h/2-.5-focal*local[:,1]/-local[:,2]))
    low=np.maximum([0,0],np.floor(uv.min(0)-8)).astype(int)
    high=np.minimum([w,h],np.ceil(uv.max(0)+9)).astype(int)
    if np.any(high-low<8):return None
    return [int(low[0]),int(low[1]),int(high[0]-low[0]),int(high[1]-low[1])]


def review_separation(views,cameras):
    if not isinstance(views,list) or len(views)!=len(cameras):
        raise ValueError('Review every policy camera once')
    seen=set()
    for view in views:
        if not isinstance(view,dict) or set(view)!={'camera','held_fruit_visible','separation','reason'}:
            raise ValueError('Invalid visual separation evidence')
        camera=view['camera']
        if camera not in cameras or camera in seen:raise ValueError('Missing or duplicate camera review')
        seen.add(camera)
        if type(view['held_fruit_visible']) is not bool or view['separation'] not in ('visible_gap','connected','occluded'):
            raise ValueError('Report visible gap, connected, or occluded honestly')
        if not isinstance(view['reason'],str) or not 1<=len(view['reason'].strip())<=800:
            raise ValueError('Describe held fruit and stalk-to-plant connection in each camera')
    allowed=(all(v['held_fruit_visible'] for v in views) and
             any(v['separation']=='visible_gap' for v in views) and
             not any(v['separation']=='connected' for v in views))
    return {'allowed':allowed,'views':views,'method':'model_two_view_separation_review',
            'reason':'Model reports held fruit in both views and a visible separation, with no conflicting connected view'
                     if allowed else 'Separation or continued holding not visually established; do not transport',
            'note':'Model visual evidence only; not ground-truth detachment or harvest certification.'}
