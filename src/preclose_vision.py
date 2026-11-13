"""Check visible support for a previously selected stalk using RGB-D only.

This checks spatial consistency, not object identity or a successful grasp. The
original mature-fruit selection remains necessary. Occlusion is unverified,
never a reason to reuse an old target blindly.
"""
import numpy as np


def check_stalk_alignment(target, axis, views):
    target = np.asarray(target, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if target.shape != (3,) or axis.shape != (3,) or not np.isfinite([target, axis]).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError('Finite measured target and stalk axis required')
    axis = axis / np.linalg.norm(axis)
    checks = {}
    for camera, (rgb, depth, calibration) in views.items():
        h, w = depth.shape
        v, u = np.indices(depth.shape)
        f = h / (2 * np.tan(np.deg2rad(calibration['fovy']) / 2))
        local = np.stack(((u+.5-w/2)*depth/f, -(v+.5-h/2)*depth/f, -depth), axis=-1)
        xyz = local @ np.asarray(calibration['rotation']).T + calibration['position']
        color = np.asarray(rgb, dtype=float)
        delta = xyz - target
        along = delta @ axis
        radial = np.linalg.norm(delta - along[..., None]*axis, axis=-1)
        valid = np.isfinite(depth) & (depth > .01) & (depth < 2)
        # Only the previously measured short stalk's interior. Exclude the
        # calyx below and the upstream branch above; do not fit a new height.
        valid &= (along > -.006) & (along < .018) & (radial < .004)
        valid &= (color[..., 1] > 1.15*color[..., 0]) & (color[..., 1] > 1.15*color[..., 2])
        levels = np.sort(along[valid])
        span = float(np.quantile(levels, .95)-np.quantile(levels, .05)) if len(levels) else 0.
        gap = float(np.max(np.diff(levels), initial=0))
        supported = len(levels) >= 12 and span >= .008 and gap <= .005
        checks[camera] = {'supported': bool(supported), 'samples': len(levels),
                          'visible_span_m': span, 'largest_gap_m': gap}
    supported = [camera for camera, check in checks.items() if check['supported']]
    passed = len(supported) >= 2
    return {'passed': passed, 'method': 'preclose_rgbd_stalk_support_v1',
            'target_m': target.tolist(), 'axis_world': axis.tolist(),
            'support_radius_m': .004, 'minimum_visible_span_m': .008,
            'required_views': 2, 'views': checks,
            'reason': 'stalk_supported_at_locked_point' if passed else 'stalk_displaced_or_insufficient_visible_evidence',
            'note': 'Spatial RGB-D support only, not identity or stable grasp. No target or grasp-height update.'}
