"""Row direction context from calibrated cameras and recorded visual measurements."""
import numpy as np


def project(point, calibration, size=(480, 360)):
    local = (np.asarray(point) - calibration['position']) @ calibration['rotation']
    if local[2] >= -.01:
        return None
    w, h = size
    focal = h / (2 * np.tan(np.deg2rad(calibration['fovy']) / 2))
    return np.array([w / 2 + focal * local[0] / -local[2] - .5,
                     h / 2 - focal * local[1] / -local[2] - .5])


def row_guidance(base, calibration, visual_hints, history):
    """Advice only: never grants search, changes direction or reuses grasp targets."""
    base = np.asarray(base, dtype=float)
    anchor = np.array([base[0] + .30, base[1], .26])
    a, b = project(anchor, calibration), project(anchor + [0, .10, 0], calibration)
    mapping = None if a is None or b is None else {
        'world_plus_y_image_delta_xy_per_10cm': (b - a).tolist(),
        'note': 'Projected direction of world +Y along the bed, NOT optical flow. '
                'Negative image x is left; camera motion produces opposite flow.'}
    measured = []
    for hint in visual_hints:
        point = np.asarray(hint['measured_stalk_m'], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            continue
        dy = float(point[1] - base[1])
        measured.append({**hint, 'current_relative_y_m': dy,
                         'toward_measured_target_direction': (1 if dy > 0 else -1) if abs(dy) > .05 else None,
                         'horizontal_distance_m': float(np.linalg.norm(point[:2] - base[:2]))})
    measured.sort(key=lambda x: x['horizontal_distance_m'])
    return {'direction_convention': '1 = world +Y; -1 = world -Y. Never infer world sign from wrist screen left/right.',
            'harvest_close_calibration': mapping,
            'previously_confirmed_ripe_but_unreachable_here': measured,
            'completed_searches': list(history),
            'remaining_search_steps': max(0, 4 - len(history)),
            'instruction': 'After local candidates are resolved, prioritize reobserving a visually confirmed ripe '
                           'target from a better parking point. Unreachable at one station does NOT mean absent '
                           'or globally unreachable. Compare toward_measured_target_direction before choosing. '
                           'If repeated moves show only turning/empty regions, reconsider direction using images '
                           'and history instead of repeating by default. These old points are navigation context '
                           'only, not current fruit identity, visibility, maturity or grasp permission. '
                           'Current local queue, original move/call budgets and fresh survey still apply.'}
