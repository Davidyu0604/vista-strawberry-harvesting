"""Kinematic preflight from measured targets and robot state, without scene truth."""
import numpy as np


def approach_reachability(env, target, base_delta_y=0.):
    # Translating the robot along world Y is equivalent to translating its target
    # oppositely. This keeps live physics state untouched, including on failure.
    target = np.asarray(target, dtype=float) - [0, base_delta_y, 0]
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Expected finite measured target')
    start = env.state().copy()
    errors = []
    for fraction in np.linspace(0, 1, 12):
        point = target - [.055 * (1 - fraction), 0, 0]
        start, error = env.ik(point, 1, start=start, approach_pitch=0)
        errors.append(float(error))
    maximum = max(errors)
    return {'reachable': bool(np.isfinite(errors).all() and maximum <= .006),
            'max_ik_error_m': maximum, 'sample_errors_m': errors,
            'base_delta_y_m': float(base_delta_y),
            'note': 'Kinematics only: sampled pregrasp-to-grasp positions; not collision clearance or guaranteed physical tracking.'}


def parking_plan(env, target):
    current = approach_reachability(env, target)
    result = {'current': current, 'recommended_delta_y_m': None, 'candidates': []}
    if current['reachable']:
        result['next_action'] = 'approach_target'
        return result
    base = env.base_pose()
    for delta in (-.04, .04, -.08, .08):
        if abs(base[1] + delta - env.meta['harvest_base_y']) > .30:
            continue
        candidate = approach_reachability(env, target, delta)
        result['candidates'].append(candidate)
    valid = [c for c in result['candidates'] if c['reachable']]
    if valid:
        best = min(valid, key=lambda c: (abs(c['base_delta_y_m']), c['max_ik_error_m']))
        result['recommended_delta_y_m'] = best['base_delta_y_m']
    result['next_action'] = 'refine_parking' if valid else 'clear_target then search_row or finish'
    return result


def scored_parking_candidate(env, target, delta):
    """Score robot kinematics only; never read fruit bodies or collision labels."""
    import mujoco
    target = np.asarray(target, dtype=float) - [0, delta, 0]
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Expected finite measured target')
    base = env.base_pose()
    start = env.state()
    # A move ends in search_row's observation posture, not today's arm posture.
    if delta:
        start, observe_error = env.ik([.19, base[1]-.15, .33], 1, approach_pitch=0)
    else:
        observe_error = 0.
    pre = target - [.055, 0, 0]
    qpre, pre_error = env.ik(pre, 1, start=start, approach_pitch=0)
    qend, end_error = env.ik(target, 1, start=qpre, approach_pitch=0)
    errors = [float(observe_error), float(pre_error), float(end_error)]
    q = qpre.copy()
    poses = [qpre, qend]
    for f in np.linspace(0, 1, 12):
        q, error = env.ik(pre + f*(target-pre), 1, start=q, approach_pitch=0)
        poses.append(q.copy()); errors.append(float(error))
    # The controller stops within 4 mm: reject candidates that only work at
    # one exact parking coordinate. This is a bounded endpoint robustness check.
    for dy in (-.004, .004):
        for point, initial in ((pre, qpre), (target, qend)):
            q, error = env.ik(point+[0, dy, 0], 1, start=initial, approach_pitch=0)
            poses.append(q.copy()); errors.append(float(error))
    low, high = env.model.actuator_ctrlrange[:5].T
    poses = np.asarray(poses)[:, :5]
    margin = float(np.min(np.minimum(poses-low, high-poses)/(high-low)))
    # Approximate the controller's componentwise joint slew as a path score.
    # This does not simulate the exact 5 mm subdivisions, tracking or contact;
    # straight Cartesian interpolation would hide the arm's approach arc.
    q = qpre.copy(); bow = 0.; steps = 0
    while True:
        data = env.ikdata
        data.qpos[:] = env.data.qpos; data.qpos[env.qa] = q
        mujoco.mj_kinematics(env.model, data)
        tcp = data.site_xpos[env.tcp].copy()
        f = np.clip(np.dot(tcp-pre, target-pre)/.055**2, 0, 1)
        bow = max(bow, float(np.linalg.norm(tcp-(pre+f*(target-pre)))))
        if np.max(np.abs(q-qend)) < 1e-8: break
        q = np.clip(qend, q-.04, q+.04); steps += 1
        if steps > 180: break
    maximum = max(errors)
    finite = bool(np.isfinite(errors).all() and np.isfinite(poses).all() and np.isfinite(bow))
    reachable = finite and maximum <= .006 and margin >= 0 and steps <= 180
    score = maximum/.006 + bow/.02 + .5*(1-2*margin) + .15*abs(delta)/.10 if finite else None
    return {'base_delta_y_m':float(delta), 'parking_y_m':float(base[1]+delta),
            'reachable':bool(reachable), 'max_ik_error_m':maximum if finite else None,
            'joint_margin_fraction':margin if finite else None,
            'approach_arc_deviation_m':bow if finite else None, 'score':score}


def optimized_parking_plan(env, target):
    """Best scored point on a bounded 1 cm grid, not a global optimum."""
    base = env.base_pose()
    candidates = [scored_parking_candidate(env, target, 0.)]
    for cm in range(-10, 11):
        delta = cm/100
        if cm and abs(base[1]+delta-env.meta['harvest_base_y']) <= .30:
            candidates.append(scored_parking_candidate(env, target, delta))
    current = candidates[0]
    valid = [c for c in candidates if c['reachable']]
    best = min(valid, key=lambda c:(c['score'], abs(c['base_delta_y_m']))) if valid else None
    # Hysteresis avoids spending moves on numerical noise or tiny gains.
    if best and current['reachable'] and current['score']-best['score'] < .08:
        best = current
    delta = best['base_delta_y_m'] if best and best is not current else None
    return {'mode':'target-aware-grid-v1', 'measured_target_m':np.asarray(target).tolist(),
            'current':current, 'candidates':candidates, 'selected':best,
            'recommended_delta_y_m':delta,
            'next_action':'refine_parking' if delta is not None else ('approach_target' if best else 'clear_target then search_row or finish'),
            'note':'Local kinematic score: IK error, joint margin, approximate joint-slew approach arc, travel. No visibility/collision guarantee; contact allowed. Reobserve after driving; grasp height unchanged.'}
