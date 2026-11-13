"""Calibrated basket attached behind the arm; no fruit oracle in control helpers."""
import numpy as np

BASKET_IN_CHASSIS = np.array([0., -.145, .102])
POLICY_CAMERAS = ('harvest_close', 'wrist')


def mount_basket(root, meta):
    world = root.find('worldbody')
    chassis = root.find(".//body[@name='chassis']")
    basket = world.find("body[@name='basket']")
    world.remove(basket)
    chassis.append(basket)
    basket.set('pos', '0 -.145 .102')
    # 142.5 x 120.1 mm outer footprint fits the rear deck. Floor top is
    # 102 mm above the chassis origin, above the 90 mm deck surface.
    scale = np.array([.75, .78, 1.])
    for geom in basket.findall('geom'):
        for key in ('pos', 'size'):
            values = np.fromstring(geom.get(key), sep=' ')
            geom.set(key, ' '.join(map(str, values * scale)))
        geom.set('mass', '0')
    import xml.etree.ElementTree as ET
    ET.SubElement(basket, 'inertial', pos='0 0 .02', mass='.08',
                  diaginertia='.00014 .00018 .00022')
    # Existing electronics stay represented, inside the deck rather than under
    # the basket interior. Cable loops also fit between the chassis plates.
    for geom in chassis.findall('geom'):
        name = geom.get('name', '')
        if name.startswith(('electronics_board', 'compute_module', 'usb_', 'terminal_')):
            pos = np.fromstring(geom.get('pos'), sep=' ')
            pos[2] -= .033
            geom.set('pos', ' '.join(map(str, pos)))
        elif name.startswith('cable_'):
            pts = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
            pts[:, 2] -= .055
            geom.set('fromto', ' '.join(map(str, pts.ravel())))
    meta.update(basket_mount='chassis', basket_origin='fixed rear chassis mount',
                basket_in_chassis_m=BASKET_IN_CHASSIS.tolist(),
                basket=BASKET_IN_CHASSIS.tolist(), basket_coordinates='chassis',
                basket_inner_half_xy_m=[.06375, .05226],
                basket_acceptance_half_xy_m=[.045, .034],
                basket_rim_above_floor_m=.068,
                policy_cameras=list(POLICY_CAMERAS), electronics_location='inside chassis',
                delivery_requires_base_travel=False)


def chassis_point(env, point):
    """Fixed robot calibration + robot pose only; never read the basket body."""
    chassis = env.data.body('chassis')
    return chassis.xpos + chassis.xmat.reshape(3, 3) @ np.asarray(point, float)


def ready_point(env):
    initial = np.asarray(env.meta['initial_tcp_world'], float).copy()
    pose = np.asarray(env.meta['initial_base_pose'])
    initial[:2] -= pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    initial[:2] = np.array([[c, s], [-s, c]]) @ initial[:2]
    return chassis_point(env, initial)
