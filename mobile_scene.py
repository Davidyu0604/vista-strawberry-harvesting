"""Seeded scene generation only. Layout truth must never enter policy decisions."""
import numpy as np

SEARCH_STEP_M = .18
SEARCH_RADIUS_M = .72
MODERATE_START_HALF_WIDTH_M = .20
MODERATE_FORWARD_OPPORTUNITY_M = (.26, .38)
MODERATE_BACKWARD_OPPORTUNITY_M = (-.38, -.26)
MODERATE_NEAR_FRUIT_INTERVAL_M = (-.14, .10)
MODERATE_MAX_SAMPLING_ATTEMPTS = 1024


def sample_mobile_layout(seed, level="moderate"):
    if level not in ("moderate", "stress"):
        raise ValueError("Unknown mobile layout level")
    rng = np.random.default_rng(seed + 19200)
    base_y = float(rng.uniform(.10, 1.70))
    mode = str(rng.choice(['gap', 'unripe', 'mixed'], p=[.4, .35, .25]))
    if level == 'moderate' and mode == 'gap':mode = 'unripe'
    # Rejection sampling is scene generation, never a policy target oracle.
    for attempt in range(1, MODERATE_MAX_SAMPLING_ATTEMPTS + 1):
        layout = _sample_row(rng, base_y, mode,
                             MODERATE_START_HALF_WIDTH_M if level == 'moderate' else .52,
                             moderate=level == 'moderate')
        # Both row directions have a nearby ripe opportunity. Neither direction
        # is a trap with only unripe fruit; visibility still depends on cameras.
        # This is generation-only. The controller must still use its cameras.
        nearby_ripe = any(m == 'ripe' and MODERATE_FORWARD_OPPORTUNITY_M[0] <= p[1]-base_y <= MODERATE_FORWARD_OPPORTUNITY_M[1]
                          for p,m in zip(layout['positions'],layout['maturity']))
        backward_ripe = any(m == 'ripe' and MODERATE_BACKWARD_OPPORTUNITY_M[0] <= p[1]-base_y <= MODERATE_BACKWARD_OPPORTUNITY_M[1]
                            for p,m in zip(layout['positions'],layout['maturity']))
        near_unripe = any(m in ('turning','green') and MODERATE_NEAR_FRUIT_INTERVAL_M[0] <= p[1]-base_y <= MODERATE_NEAR_FRUIT_INTERVAL_M[1]
                          for p,m in zip(layout['positions'],layout['maturity']))
        if level == 'stress' or (nearby_ripe and backward_ripe and near_unripe):
            layout.update(level=level, sampling_attempts=attempt,
                          initial_region_half_width_m=MODERATE_START_HALF_WIDTH_M if level == 'moderate' else .52,
                          ripe_opportunity_along_row_m=MODERATE_FORWARD_OPPORTUNITY_M[1] if level == 'moderate' else None,
                          forward_ripe_interval_m=list(MODERATE_FORWARD_OPPORTUNITY_M) if level == 'moderate' else None,
                          backward_ripe_interval_m=list(MODERATE_BACKWARD_OPPORTUNITY_M) if level == 'moderate' else None,
                          near_unripe_interval_m=list(MODERATE_NEAR_FRUIT_INTERVAL_M) if level == 'moderate' else None)
            return layout
    raise RuntimeError(f'Could not generate a bilateral moderate mobile layout in {MODERATE_MAX_SAMPLING_ATTEMPTS} attempts')


def _sample_row(rng, base_y, mode, region_half_width, moderate=False):
    positions, maturity, scales = [], [], []
    y = -.75 + rng.uniform(0, .15)
    clusters = 0
    while y < 2.85:
        clusters += 1
        for _ in range(int(rng.integers(1,5) if moderate else rng.integers(0,6))):
            jitter=.08 if moderate else .12
            point = np.array([rng.uniform(.25, .35), y+rng.uniform(-jitter, jitter), rng.uniform(.19, .29)])
            local = abs(point[1]-base_y) < region_half_width
            if mode == 'gap' and local:
                continue
            if any(np.linalg.norm(point-p) <= .080 for p in positions):
                continue
            stages = ['turning', 'green'] if mode == 'unripe' and local else ['ripe', 'turning', 'green']
            maturity.append(str(rng.choice(stages)))
            positions.append(point)
            scales.append(float(rng.uniform(.86, 1.08)))
        y += rng.uniform(.28, .40) if moderate else rng.uniform(.27, .47)
    return {'positions': positions, 'maturity': maturity, 'scales': scales,
            'initial_base_y': base_y, 'start_region': mode, 'cluster_count': clusters}


def search_limits(meta):
    # Public motion calibration only; no target, ripeness or region labels.
    return (SEARCH_STEP_M, SEARCH_RADIUS_M) if meta.get('mobile_layout') else (.10, .30)
