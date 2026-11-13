"""Conservative RGB-D colour gate for this simulated fruit/stalk geometry.

No environment, fruit IDs, object poses or maturity labels are inputs. This is
not a calibrated real-fruit classifier. Unknown/occluded evidence fails closed.
"""
import numpy as np
from PIL import ImageDraw


def assess_maturity(target, axis, depth, calibration, rgb):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    h, w = depth.shape
    v, u = np.indices(depth.shape)
    f = h / (2 * np.tan(np.deg2rad(calibration['fovy']) / 2))
    local = np.stack(((u+.5-w/2)*depth/f, -(v+.5-h/2)*depth/f, -depth), axis=-1)
    xyz = local @ np.asarray(calibration['rotation']).T + calibration['position']
    delta = xyz - np.asarray(target)
    along = delta @ axis
    radial = np.linalg.norm(delta - along[..., None]*axis, axis=-1)
    # Fitted grasp is ~12 mm above calyx. Examine body below it, excluding
    # calyx/stalk and anything outside this fruit's small 3-D neighbourhood.
    mask = np.isfinite(depth) & (depth > .01) & (depth < 2)
    mask &= (along < -.019) & (along > -.054) & (radial < .023)
    hsv = np.asarray(rgb.convert('HSV'))
    hue, sat, value = [hsv[..., i] for i in range(3)]
    red = ((hue < 8) | (hue > 244)) & (sat >= 115) & (value >= 30)
    count = int(mask.sum())
    red_fraction = float(red[mask].mean()) if count else 0.
    # Count red against ALL visible samples: dark, green, orange and unknown
    # pixels cannot disappear from the denominator and manufacture confidence.
    ys, xs = np.where(mask)
    width = int(np.ptp(xs)+1) if count else 0
    height = int(np.ptp(ys)+1) if count else 0
    passed = count >= 120 and width >= 12 and height >= 12 and red_fraction >= .80
    result = {'passed': bool(passed), 'method': 'rgbd_stalk_body_red_v1',
              'samples': count, 'red_fraction': red_fraction,
              'minimum_samples': 120, 'minimum_red_fraction': .80,
              'target_m': np.asarray(target).tolist(), 'axis_world': axis.tolist(),
              'reason': 'visually_red' if passed else 'not_red_enough_or_insufficient_visible_body'}
    marked = rgb.convert('RGB').copy()
    if count:
        box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
        result['body_box_xyxy'] = box
        ImageDraw.Draw(marked).rectangle(box, outline='cyan', width=2)
        marked = marked.crop((max(0,box[0]-10),max(0,box[1]-10),min(w,box[2]+10),min(h,box[3]+10)))
    marked = marked.resize((marked.width*3, marked.height*3))
    return result, marked
