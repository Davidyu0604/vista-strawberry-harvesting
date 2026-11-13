"""Experimental graspable peduncle with bend/tension separation.

Parameters are engineering fixtures, NOT measurements of real strawberry tissue.
No grasp welds, fruit teleportation or target-based detach commands.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scene import build, add, rod, stem, v, camera
from env import StrawberryEnv

STALK_LENGTH = .036
STALK_RADIUS = .0015
BEND_THRESHOLD = np.deg2rad(25)
EXTENSION_THRESHOLD = .004
TENSION_THRESHOLD = .25


def cross3(a,b):
    """Avoid general-axis NumPy dispatch for tiny vectors in the physics loop."""
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])


def build_peduncle(seed=0, single=True, fixed_base=False, basket_mount=None, mobile_layout=False, mobile_layout_level="moderate"):
    diverse = not single
    if mobile_layout and (single or fixed_base):raise ValueError("Mobile layout requires full moving mode")
    xml, meta = build(seed, diverse=diverse, mobile_layout=mobile_layout, mobile_layout_level=mobile_layout_level)
    pose_rng=np.random.default_rng(seed+7300)
    tilt_rng=np.random.default_rng(seed+8100)
    root = ET.fromstring(xml);world = root.find('worldbody')
    basket_mount = basket_mount or ('ground' if single else 'chassis')
    if basket_mount not in ('ground', 'chassis'):raise ValueError('Unknown basket mount')
    if basket_mount == 'chassis':
        from mounted_basket import mount_basket
        mount_basket(root, meta)
    else:meta['basket_mount']='ground'
    if not single:
        # The smallest fruit inertias destabilize the explicitly applied spring
        # at 1ms. Substep at 0.5ms while retaining the 20Hz control interface.
        root.find('option').set('timestep','.0005')
    contact = root.find('contact')
    if contact is None: contact = add(root,'contact')
    count = 1 if single else len(meta['fruit_positions'])
    if single:
        for body in list(world.findall('body')):
            name = body.get('name','')
            if name.startswith(('fruit_','stem_')) and int(name.split('_')[-1]) != 0:
                world.remove(body)
            elif name.startswith('branch_'):
                world.remove(body)
        for key in ['fruit_positions','ripe','maturity','fruit_scales','fruit_euler']:
            meta[key] = meta[key][:1]
    for i in range(count):
        fruit = world.find(f"body[@name='fruit_{i}']")
        if diverse and (np.linalg.norm(np.asarray(meta['fruit_positions'][i][:2])-[0,meta['mobile_layout_audit']['initial_base_y']])<.6 if mobile_layout else i<8):fruit.attrib.pop('sleep',None)
        angle=float(tilt_rng.uniform(0,np.deg2rad(15))) if diverse else 0.
        azimuth=float(tilt_rng.uniform(-np.pi,np.pi)) if diverse else 0.
        quat=np.array([np.cos(angle/2),np.cos(azimuth)*np.sin(angle/2),np.sin(azimuth)*np.sin(angle/2),0.])
        rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,quat);rotation=rotation.reshape(3,3)
        fruit.attrib.pop('euler',None);fruit.set('quat',v(quat))
        meta.setdefault('fruit_quat',[]).append(quat.tolist())
        # Leaf/branch collision proxies use bit 2 and overlap dense fruit clusters.
        # Keep robot/ground/basket and fruit contacts (bit 1), but do not let
        # coarse vegetation proxies push attached berries out of their rest pose.
        fruit.find(f"geom[@name='fruit_collision_{i}']").set('conaffinity','1')
        scale = meta['fruit_scales'][i]
        bottom = .022*scale; top = bottom+STALK_LENGTH
        rod(fruit,[0,0,bottom],[0,0,top],STALK_RADIUS,(.30,.48,.12,1),
            name=f'peduncle_collision_{i}',conaffinity='1',condim='6',
            friction='2 .02 .002',solref='.004 1',solimp='.95 .99 .001',mass='0')
        add(fruit,'site',name=f'peduncle_tip_{i}',pos=v([0,0,top]),size='.001',rgba='0 0 0 0')
        add(fruit,'site',name=f'peduncle_grasp_{i}',pos=v([0,0,bottom+.015]),size='.001',rgba='0 0 0 0')
        old = world.find(f"body[@name='stem_{i}']");start=np.fromstring(old.get('pos'),sep=' ');world.remove(old)
        end = np.array(meta['fruit_positions'][i])+rotation@np.array([0,0,top])
        stem(world,start,end,np.random.default_rng(seed+900+i),i,scale)
        upstream = world.find(f"body[@name='stem_{i}']")
        if diverse and (np.linalg.norm(np.asarray(meta['fruit_positions'][i][:2])-[0,meta['mobile_layout_audit']['initial_base_y']])<.6 if mobile_layout else i<8):upstream.attrib.pop('sleep',None)
        upstream.find('site').set('pos',v(end-start))
        upstream.find('site').set('quat',v(quat))
        add(contact,'exclude',body1=f'fruit_{i}',body2=f'stem_{i}')
    # Experimental visible rubber insert: wider/longer contact face keeps a
    # bent stalk from slipping off its edge during transport. Hardware unvalidated.
    grip = root.find(".//body[@name='gripper']")
    add(grip,'geom',name='peduncle_rubber_insert',type='box',pos='-.007 0 -.080',
        size='.007 .014 .018',rgba='.16 .19 .17 1',mass='.002',conaffinity='3',
        condim='6',friction='2 .02 .002',solref='.004 1')
    grip.find("site[@name='tcp']").set('pos','.001 0 -.080')
    camera(root.find(".//body[@name='chassis']") if diverse else world,'harvest_close',[.09,-.40,.39],(np.array([.29,-.06,.25]) if diverse else np.array(meta['fruit_positions'][0])+[0,0,.025]),42)
    if diverse:
        base_y=float(pose_rng.uniform(-.03,.03))
        if mobile_layout:base_y=meta['mobile_layout_audit']['initial_base_y']
        meta.update(initial_base_pose=[0.,base_y,0.],harvest_base_y=base_y,
                    initial_tcp_world=(np.array([.19,-.15+base_y,.33])+pose_rng.uniform([-.01,-.015,-.01],[.01,.015,.01])).tolist(),
                    allow_no_reachable_target=True,stalk_tilt_limit_deg=15,
                    fruit_orientation_representation='fruit_quat; fruit_euler removed',
                    camera_target='chassis-mounted workspace view, independent of fruit index')
        meta.pop('fruit_euler',None)
        if fixed_base:
            meta.update(initial_base_pose=[0.,0.,0.],harvest_base_y=0.,
                        initial_tcp_world=[.19,-.15,.33],fixed_harvest_base=True)
    meta.update(wrist_camera_profile='workspace-side-v2' if diverse else 'legacy-gripper-v1')
    meta.update(scene_version=('peduncle-v17-wrist-workspace' if basket_mount=='chassis' else 'peduncle-v15-row-search') if diverse else 'peduncle-v13-experimental',interactive_fruit_count=count,
                harvest_mechanism='grasp 3mm peduncle; bend + tensile extension at upper connection',
                single_fruit_fixture=single,gripper_modification='expanded visible fixed-jaw rubber insert, 14x28x36mm; hardware unvalidated',
                upright_stalk_prior=not diverse, tilted_stalk_prior=diverse,
                vegetation_fruit_collision='disabled for coarse leaf/branch proxies; robot contact retained',
                tissue_calibrated=False,background_fruit_count=0 if single else meta['background_fruit_count'])
    if mobile_layout:meta['scene_version']='peduncle-v23-mobile-bilateral' if mobile_layout_level=='moderate' else 'peduncle-v18-mobile-layout'
    return ET.tostring(root,encoding='unicode'),meta


class PeduncleEnv(StrawberryEnv):
    def __init__(self,seed=0,single=True,fixed_base=False,basket_mount=None,mobile_layout=False,mobile_layout_level="moderate",**kwargs):
        super().__init__(seed,scene_builder=lambda s:build_peduncle(s,single,fixed_base,basket_mount,mobile_layout,mobile_layout_level),**kwargs)
        self.tip_sites=np.array([self.model.site(f'peduncle_tip_{i}').id for i in range(len(self.anchors))])
        self.grasp_sites=np.array([self.model.site(f'peduncle_grasp_{i}').id for i in range(len(self.anchors))])
        self.stalk_geoms=np.array([self.model.geom(f'peduncle_collision_{i}').id for i in range(len(self.anchors))])
        self.fruit_geom_lookup[:]=-1
        self.fruit_geom_lookup[self.stalk_geoms]=np.arange(len(self.anchors))
        self.stalk_metrics={}
        self._tip_velocity=np.zeros(6)
        self._anchor_velocity=np.zeros(6)

    def _fruit_forces(self, nearby):
        awake=(self.data.tree_asleep[self.fruit_trees]<0)|(self.data.tree_asleep[self.stem_trees]<0)|nearby
        bilateral=self.bilateral_contacts() & self.attached
        self.contact_ticks=np.where(bilateral,self.contact_ticks+1,0)
        for i in np.flatnonzero(self.attached & awake):
            b=self.fruit_bodies[i];s=self.stem_bodies[i]
            tip=self.data.site_xpos[self.tip_sites[i]]
            anchor=self.data.site_xpos[self.anchor_sites[i]]
            offset=tip-anchor
            rotation=self.data.xmat[b].reshape(3,3)
            upstream=self.data.xmat[s].reshape(3,3)
            axis=rotation[:,2];rest=self.data.site_xmat[self.anchor_sites[i]].reshape(3,3)[:,2]
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_SITE,
                                    int(self.tip_sites[i]),self._tip_velocity,0)
            mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_SITE,
                                    int(self.anchor_sites[i]),self._anchor_velocity,0)
            angular=self._tip_velocity[:3]-self._anchor_velocity[:3]
            velocity=self._tip_velocity[3:]-self._anchor_velocity[3:]
            force=-80*offset-.08*velocity
            bend_torque=.008*cross3(axis,rest)-.00008*angular
            extension=float(np.linalg.norm(offset))
            bend=float(np.arccos(np.clip(axis@rest,-1,1)))
            tension=80*extension
            if self.contact_ticks[i]>=20 and i not in self.grasp_origins:
                self.grasp_origins[i]=self.data.site_xpos[self.tcp].copy()
            pull=float(np.linalg.norm(self.data.site_xpos[self.tcp]-self.grasp_origins[i])) if i in self.grasp_origins else 0.
            if not bilateral[i]: self.grasp_origins.pop(i,None)
            self.stalk_metrics[int(i)]={'bend_rad':bend,'extension_m':extension,'elastic_force_n':tension,'stalk_bilateral':bool(bilateral[i]),'tcp_pull_m':pull}
            if self.contact_ticks[i]>=20 and pull>.012 and bend>BEND_THRESHOLD and extension>EXTENSION_THRESHOLD and tension>TENSION_THRESHOLD:
                self.attached[i]=False
                self.events.append({'event':'detached','fruit':int(i),'time':float(self.data.time),
                                    'mechanism':'peduncle_bend_pull','bilateral_contact':True,**self.stalk_metrics[int(i)]})
                continue
            self.data.xfrc_applied[b,:3]=force+[0,0,self.model.body_mass[b]*9.81]
            self.data.xfrc_applied[b,3:]=cross3(tip-self.data.xipos[b],force)+bend_torque
            self.data.xfrc_applied[s,:3]-=force
            self.data.xfrc_applied[s,3:]-=cross3(anchor-self.data.xipos[s],force)+bend_torque
