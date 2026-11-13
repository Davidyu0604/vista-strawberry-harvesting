"""Deterministic greenhouse scene with the pinned SO101 model (SI units)."""
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from natural_assets import textures, leaf_mesh, ridge_mesh
from scene_details import decorate

ROOT = Path(__file__).resolve().parent
from chassis import BASE_Z, build_chassis
JOINTS = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']

def v(x): return ' '.join(f'{float(a):.7g}' for a in x)
def add(p,tag,**kw): return E.SubElement(p,tag,{k:str(val) for k,val in kw.items()})
def geom(p,typ,pos,size,rgba,**kw):
    return add(p,'geom',type=typ,pos=v(pos),size=v(size),rgba=v(rgba),**kw)
def rod(p,a,b,r=.008,color=(.55,.61,.59,1),**kw):
    return add(p,'geom',type='capsule',fromto=v([*a,*b]),size=str(r),rgba=v(color),**kw)
def camera(p,name,pos,target,fovy=48):
    z=np.asarray(pos)-np.asarray(target); z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z); x/=np.linalg.norm(x); y=np.cross(z,x)
    return add(p,'camera',name=name,pos=v(pos),xyaxes=v([*x,*y]),fovy=str(fovy))

def berry_mesh():
    path=ROOT/'assets/berry.obj';verts=[];faces=[];uv=[];rings=20;sides=32
    for j in range(rings+1):
        t=np.pi*j/rings;z=.022*np.cos(t);radius=.017*np.sin(t)*(1+.32*np.cos(t))
        for k in range(sides+1):
            a=2*np.pi*k/sides;verts.append((radius*np.cos(a),radius*np.sin(a),z));uv.append((k/sides,1-j/rings))
    for j in range(rings):
        for k in range(sides):
            a=j*(sides+1)+k;b=a+1;c=a+sides+1;d=b+sides+1
            faces.extend([(a+1,c+1,b+1),(b+1,c+1,d+1)])
    path.write_text('\n'.join(['v '+v(a) for a in verts]+['vt '+v(a) for a in uv]+['f '+' '.join(f'{i}/{i}' for i in face) for face in faces])+'\n')
    return path

def berry_textures():
    from PIL import Image,ImageDraw
    paths={};width=512;height=256
    for maturity,base in [('ripe',[190,9,12]),('turning',[224,87,46]),('green',[140,166,61])]:
        pixels=np.clip(np.array(base)+np.random.default_rng(41).integers(-3,4,(height,width,1)),0,255).astype('uint8')
        img=Image.fromarray(pixels);draw=ImageDraw.Draw(img)
        for j in range(1,7):
            theta=np.pi*j/8;r=.017*np.sin(theta)*(1+.32*np.cos(theta))
            rx=max(2,width*.00065/(2*np.pi*r));ry=max(2,height*.001/(np.pi*.022*np.sin(theta)))
            for k in range(9):
                u=((k+j*.5)/9)%1
                for offset in [-width,0,width]:
                    x=u*width+offset;y=j/8*height
                    draw.ellipse((x-rx-1,y-ry-1,x+rx+1,y+ry+1),fill=(117,74,24))
                    draw.ellipse((x-rx,y-ry,x+rx,y+ry),fill=(224,179,61))
        path=ROOT/'assets/natural'/f'berry_{maturity}.png';img.save(path);paths[maturity]=path
    return paths

def seed_mesh():
    """One shared non-colliding mesh for 54 seeds, avoiding thousands of extra geoms."""
    path=ROOT/'assets/berry_seeds.obj';vertices=[];faces=[]
    for j in range(1,7):
        t=np.pi*j/8;radius=.017*np.sin(t)*(1+.32*np.cos(t))
        for k in range(9):
            a=2*np.pi*(k+j*.5)/9
            center=np.array([(radius+.00025)*np.cos(a),(radius+.00025)*np.sin(a),.022*np.cos(t)])
            start=len(vertices);stacks=4;slices=6
            for row in range(stacks+1):
                lat=np.pi*row/stacks
                for col in range(slices):
                    lon=2*np.pi*col/slices
                    vertices.append(center+np.array([.00065*np.sin(lat)*np.cos(lon),.00065*np.sin(lat)*np.sin(lon),.001*np.cos(lat)]))
            for row in range(stacks):
                for col in range(slices):
                    aa=start+row*slices+col+1;bb=start+row*slices+(col+1)%slices+1
                    if row>0:faces.append((aa,aa+slices,bb))
                    if row<stacks-1:faces.append((bb,aa+slices,bb+slices))
    path.write_text('\n'.join(['v '+v(a) for a in vertices]+['f '+' '.join(map(str,a)) for a in faces])+'\n')
    return path

def diverse_task_positions(seed):
    """Restart the whole layout if earlier samples block a later fruit."""
    rng=np.random.default_rng(seed+9100)
    spread=np.array([.025,.035,.020])
    for _ in range(256):
        positions=[]
        for i in range(8):
            nominal=np.array([.275+(i%2)*.035,-.145+(i//2)*.095,.215+(i%2)*.052])
            for _ in range(128):
                point=nominal+rng.uniform(-spread,spread)
                if all(np.linalg.norm(point-p)>.080 for p in positions):
                    positions.append(point);break
            else:break
        if len(positions)==8:return positions
    raise RuntimeError('Unable to sample collision-separated task layout')


def build(seed=0, diverse=False, mobile_layout=False, mobile_layout_level="moderate"):
    if mobile_layout and not diverse:raise ValueError("Mobile layout requires full diverse greenhouse")
    from mobile_scene import sample_mobile_layout
    mobile = sample_mobile_layout(seed, mobile_layout_level) if mobile_layout else None
    rng=np.random.default_rng(seed)
    task_rng=np.random.default_rng(seed+170)
    diversity_rng=np.random.default_rng(seed+4200)
    occlusion=str(diversity_rng.choice(['low','medium','high'])) if diverse else 'legacy'
    if mobile and mobile_layout_level=='moderate' and occlusion=='high':occlusion='medium'
    retention={'legacy':0.,'low':.2,'medium':.6,'high':1.}[occlusion]
    natural=textures()
    root=E.parse(ROOT/'assets/so101/robot.xml').getroot()
    root.set('model','SO101 ground strawberry greenhouse with wheeled base')
    root.find('compiler').set('meshdir',str(ROOT/'assets/so101'))
    add(root,'option',timestep='.001',integrator='implicitfast',cone='elliptic',impratio='10',iterations='80',jacobian='sparse')
    add(root.find('option'),'flag',sleep='enable')
    root.find('option').set('sleep_tolerance','0.0005')
    visual=add(root,'visual'); add(visual,'global',offwidth='1280',offheight='960'); add(visual,'quality',shadowsize='1024')
    add(visual,'map',znear='.0001',zfar='10')
    add(visual,'headlight',ambient='.38 .38 .38',diffuse='.5 .5 .5',specular='.15 .15 .15')
    asset=root.find('asset'); add(asset,'mesh',name='berry_mesh',file=str(berry_mesh()))
    add(asset,'mesh',name='berry_seeds',file=str(seed_mesh()))
    for maturity,path in berry_textures().items():
        add(asset,'texture',name='berry_tex_'+maturity,type='2d',file=str(path))
        add(asset,'material',name='berry_mat_'+maturity,texture='berry_tex_'+maturity,rgba='1 1 1 1',specular='.15',shininess='.2')
    add(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.48 .68 .77',rgb2='.91 .94 .85',width='512',height='3072')
    add(asset,'texture',name='floor_tex',type='2d',file=str(natural/'soil.png'))
    add(asset,'texture',name='leaf_tex',type='2d',file=str(natural/'leaf.png'))
    add(asset,'texture',name='mulch_tex',type='2d',file=str(natural/'mulch.png'))
    add(asset,'material',name='soil_mat',texture='floor_tex',texrepeat='1 1',rgba='.78 .71 .60 1',specular='.05')
    add(asset,'material',name='mulch_mat',texture='mulch_tex',texrepeat='1 1',rgba='.65 .70 .65 1',specular='.18',shininess='.2')
    for tone,color in [('dark','.24 .44 .15 1'),('mid','.32 .55 .20 1'),('light','.43 .62 .25 1')]:
        add(asset,'material',name=f'leaf_{tone}',texture='leaf_tex',rgba=color,specular='.15')
    for name,scale in [('small',.95),('medium',1.2),('large',1.4)]:
        add(asset,'mesh',name=f'leaf_{name}',file=str(leaf_mesh()),scale=v([scale]*3))
    add(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='18 18',reflectance='0')
    root.find("actuator/position[@name='gripper']").set('forcerange','-.3 .3')
    world=root.find('worldbody'); base=world.find("body[@name='base']"); base.set('pos',v([0,0,BASE_Z]))
    for body in base.iter('body'): body.set('gravcomp','1')
    # Exact robot meshes retained; contact friction appropriate to a rubber-covered gripper.
    for g in base.iter('geom'):
        if g.get('class')=='collision':
            g.set('conaffinity','3');g.set('friction','1.2 .01 .001'); g.set('rgba','0 0 0 0')
    grip=base.find(".//body[@name='gripper']")
    for g in list(grip.findall('geom')):
        if g.get('class')=='collision' and g.get('mesh')=='wrist_roll_follower_so101_v1': grip.remove(g)
    for z,x,halfwidth in [(-.045,-.024,.010),(-.062,-.021,.009),(-.079,-.0175,.008),(-.096,-.0135,.006)]:
        geom(grip,'box',[x,0,z],[.0045,halfwidth,.0085],[0,0,0,0],group='3',conaffinity='3',friction='1.5 .01 .001',name=f'fixed_pad_{abs(int(z*1000))}')
    add(grip,'site',name='tcp',pos='.007 0 -.080',size='.002',rgba='0 0 0 0')
    # Fixed wrist extrinsics: side offset clears the fingers at the observation
    # pose; aim covers the workspace and TCP. No target-dependent camera tracking.
    wrist_camera=(camera(grip,'wrist',[-.04,-.085,.015],[-.10,.075,-.08],80) if diverse
                  else camera(grip,'wrist',[0,-.065,.015],[0,0,-.11],65))
    wrist_camera.set('xyaxes',v(-np.fromstring(wrist_camera.get('xyaxes'),sep=' ')))
    geom(world,'plane',[0,0,0],[4,5,.1],[1,1,1,1],material='floor_mat')
    light=float(rng.uniform(.76,.94))
    add(world,'light',pos='-.3 -.6 1.75',dir='.15 .3 -1',directional='false',diffuse=v([light]*3),castshadow='true')
    add(world,'light',pos='-1 2 2',dir='.3 -.4 -1',diffuse='.35 .38 .32',castshadow='false')
    # A physically driven four-wheel platform. Level chassis constraint suppresses roll/pitch; passive vertical travel loads the tires;
    # x/y/yaw remain dynamic, wheels roll against the floor (no pose teleportation).
    chassis=build_chassis(root,world,base,add,geom,rod,v)
    basket=np.array([.04,.56,.024])
    basket_body=add(world,'body',name='basket',pos=v(basket))
    geom(basket_body,'box',[0,0,-.012],[.085,.067,.012],[.65,.74,.57,1],name='basket_floor')
    for dx,dy,sx,sy in [(-.09,0,.005,.072),(.09,0,.005,.072),(0,-.072,.095,.005),(0,.072,.095,.005)]:
        geom(basket_body,'box',[dx,dy,.028],[sx,sy,.04],[.77,.84,.66,1])
    for dy in [-.05,-.025,0,.025,.05]:
        geom(basket_body,'box',[0,dy,.002],[.082,.001,.001],[.40,.50,.33,1],contype='0',conaffinity='0')
    # Keep stable simplified collision envelopes, replace their boxy visuals with
    # continuous ridges, genuinely open planting holes and mildly wrinkled mulch.
    crowns=[]
    for row,x in enumerate([.53,-.94]):
        geom(world,'box',[x,1.0,.055],[.25,2.0,.055],[0,0,0,0],group='3')
        geom(world,'box',[x,1.0,.115],[.21,2.0,.015],[0,0,0,0],group='3')
        for side in [-1,1]:
            geom(world,'box',[x+side*.205,1.0,.079],[.035,2.0,.025],[0,0,0,0],group='3',euler=v([0,side*.5,0]))
        centers=[]
        for y in np.arange(-.80,2.9,.30):
            for dx,dy in [(-.07,0),(.09,.12)]:
                centers.append([dx+rng.uniform(-.012,.012),y+dy+rng.uniform(-.025,.025)])
        for mulch in [False,True]:
            name=f'row_{row}_{"mulch" if mulch else "soil"}'
            add(asset,'mesh',name=name,file=str(ridge_mesh(seed,row,centers,mulch)))
            add(world,'geom',type='mesh',mesh=name,pos=v([x,0,0]),material='mulch_mat' if mulch else 'soil_mat',contype='0',conaffinity='0')
        rod(world,[x-.11,-.98,.135],[x-.11,2.98,.135],.003,(.045,.055,.04,1),contype='0',conaffinity='0')
        for dx,y in centers:
            crown=[x+dx,y,.133];crowns.append((row,crown))
            plant(world,crown,rng,len(crowns)-1,retention if diverse else None)
    # Light tire marks and a few dead leaves remain visual, leaving drive physics unchanged.
    for x in [-.205,.245]:
        for y in np.arange(-1.05,2.9,.06):
            geom(world,'box',[x+rng.uniform(-.004,.004),y,.0005],[.023,.004,.0003],[.24,.19,.12,.45],euler=v([0,0,.25]),contype='0',conaffinity='0')
    for _ in range(28):
        x=rng.uniform(-.62,.20);y=rng.uniform(-1,3)
        add(world,'geom',type='mesh',mesh='leaf_small',pos=v([x,y,.001]),euler=v([0,0,rng.uniform(-np.pi,np.pi)]),rgba='.31 .23 .10 1',contype='0',conaffinity='0')
    # Independent task RNG: visual detail does not silently change task layouts.
    positions=[];ripe=[];sizes=[];maturity=[];angles=[]
    diverse_positions=mobile["positions"] if mobile else (diverse_task_positions(seed) if diverse else None)
    for i in range(len(diverse_positions) if mobile else 8):
        scale=float(task_rng.uniform(.86,1.08) if diverse else (task_rng.uniform(.94,1.04) if i==0 else task_rng.uniform(.86,1.08)))
        nominal=np.array([.275+(i%2)*.035,-.145+(i//2)*.095,.215+(i%2)*.052])
        for _ in range(1000):
            spread=np.array([.025,.035,.020]) if diverse else np.full(3,.005)
            point=diverse_positions[i] if diverse else nominal+task_rng.uniform(-spread,spread)
            if all(np.linalg.norm(point-p)>(.080 if diverse else .022*(scale+ss)+.004) for p,ss in zip(positions,sizes)):break
        else:raise RuntimeError('Could not create separated fruit layout')
        stage=mobile['maturity'][i] if mobile else ('ripe' if i==0 and not diverse else str(task_rng.choice(['ripe','turning','green'],p=[.5,.25,.25])))
        if mobile:scale=mobile['scales'][i]
        tilt=task_rng.uniform([-.13,-.13,-np.pi],[.13,.13,np.pi])
        positions.append(point);sizes.append(scale);maturity.append(stage);ripe.append(stage=='ripe');angles.append(tilt.tolist())
        stem(world,[.43,point[1]+.025,.137],point+[0,0,.022*scale],rng,i,scale)
        name=f'berry_mesh_{i}'
        add(asset,'mesh',name=name,file=str(ROOT/'assets/berry.obj'),scale=v([scale]*3))
        body=add(world,'body',name=f'fruit_{i}',pos=v(point),euler=v(tilt),sleep='init')
        add(body,'freejoint',name=f'fruit_joint_{i}')
        add(body,'inertial',pos='0 0 0',mass=str(.012*scale**3),diaginertia=v(np.array([.000002,.000002,.0000015])*scale**5))
        fruit_visual(body,rng,stage,scale=scale,mesh=name)
        geom(body,'ellipsoid',[0,0,0],np.array([.017,.017,.021])*scale,[0,0,0,0],name=f'fruit_collision_{i}',group='3',conaffinity='3',friction='2.5 .02 .01',condim='6',solref='.002 1',solimp='.95 .99 .001')
    # Each background plant carries an irregular 2–5 fruit cluster, not a uniform row.
    background_count=0
    for row,crown in crowns:
        if mobile:
            if row==0 and crown[0]<.53:continue  # Near row is entirely generated by the variable layout.
        elif row==0 and -.27<crown[1]<.27:continue
        for _ in range(int(rng.integers(2,6))):
            side=-1 if crown[0]<( .53 if row==0 else -.94) else 1
            # Keep free fruit bodies separated at reset, rather than overlapping decorations.
            for attempt in range(100):
                point=np.array(crown)+[side*rng.uniform(.055,.15),rng.uniform(-.10,.10),rng.uniform(.045,.10)]
                if all(np.linalg.norm(point-p)>(.080 if diverse else .047) for p in positions):break
            else:continue
            i=len(positions);tilt=rng.uniform([-.2,-.2,-3],[.2,.2,3])
            stage=str(rng.choice(['ripe','turning','green'],p=[.55,.20,.25]))
            stem(world,crown,point+[0,0,.022],rng,i,1.)
            positions.append(point);sizes.append(1.);maturity.append(stage);ripe.append(stage=='ripe');angles.append(tilt.tolist())
            body=add(world,'body',name=f'fruit_{i}',pos=v(point),euler=v(tilt),sleep='init')
            add(body,'freejoint',name=f'fruit_joint_{i}')
            add(body,'inertial',pos='0 0 0',mass='.012',diaginertia='.000002 .000002 .0000015')
            fruit_visual(body,rng,stage,seeds="mesh")
            geom(body,'ellipsoid',[0,0,0],[.017,.017,.021],[0,0,0,0],name=f'fruit_collision_{i}',group='3',conaffinity='3',friction='2.5 .02 .01',condim='6',solref='.002 1',solimp='.95 .99 .001')
            background_count+=1
    # Greenhouse frame, arched roof and translucent cover.
    for y in [-1.3,-.2,.9,2.0,3.1]:
        rod(world,[-1.65,y,0],[-1.65,y,1.8],.018)
        rod(world,[1.05,y,0],[1.05,y,1.8],.018)
        pts=[]
        for t in np.linspace(0,np.pi,17): pts.append([-.3+1.35*np.cos(t),y,1.8+.85*np.sin(t)])
        for a,b in zip(pts[:-1],pts[1:]): rod(world,a,b,.015)
    for x,z in [(-1.65,1.8),(1.05,1.8),(-.3,2.65)]: rod(world,[x,-1.3,z],[x,3.1,z],.012)
    for t0,t1 in zip(np.linspace(0,np.pi,17)[:-1],np.linspace(0,np.pi,17)[1:]):
        a=np.array([-.3+1.35*np.cos(t0),0,1.8+.85*np.sin(t0)])
        b=np.array([-.3+1.35*np.cos(t1),0,1.8+.85*np.sin(t1)])
        mid=(a+b)/2; mid[1]=.9; delta=b-a
        geom(world,'box',mid,[np.linalg.norm(delta)/2,2.2,.002],[.80,.91,.87,.08],euler=v([0,-np.arctan2(delta[2],delta[0]),0]),contype='0',conaffinity='0')
    for x in [-1.67,1.07]: geom(world,'box',[x,.9,1.2],[.004,2.2,.6],[.80,.91,.87,.16],contype='0',conaffinity='0')
    decorate(root,world,chassis,basket_body,crowns,seed,add,geom,rod,v)
    camera(world,'overview',[-1.5,-2.65,1.8],[-.05,.65,.4],58)
    camera(chassis,'top',[.0,-.42,.70],[.25,0,.20],65)
    camera(world,'detail',[-.85,-.80,.78],[.10,-.05,.30],58)
    camera(world,'chassis_side',[-.85,-.27,.32],[.02,-.10,.14],44)
    camera(world,'robot_detail',[-.60,-.72,.48],[.015,-.10,.20],48)
    camera(world,'plant_detail',[.08,-1.02,.48],[.47,-.66,.20],48)
    return E.tostring(root,encoding='unicode'), {'mobile_layout':bool(mobile_layout),'mobile_layout_level':mobile_layout_level if mobile else None,'mobile_layout_audit':({'initial_base_y':mobile['initial_base_y'],'start_region':mobile['start_region'],'cluster_count':mobile['cluster_count'],'near_row_fruit_count':len(mobile['positions']),'sampling_attempts':mobile['sampling_attempts'],'initial_region_half_width_m':mobile['initial_region_half_width_m'],'ripe_opportunity_along_row_m':mobile['ripe_opportunity_along_row_m'],'forward_ripe_interval_m':mobile.get('forward_ripe_interval_m'),'backward_ripe_interval_m':mobile.get('backward_ripe_interval_m'),'near_unripe_interval_m':mobile.get('near_unripe_interval_m')} if mobile else None),'scene_version':'performance-v11','interactive_fruit_count':len(positions),'vegetation_physics':'spring hinged stems and leaf clusters; soft collision proxies; no vegetation self-collision','background_seeds':'54 seeds per fruit, shared non-colliding mesh','reference_brand':'Hiwonder black 4WD 300x245x100 mm; user confirmed','reference_dimensions':'300x245x100 mm chassis envelope; internal layout approximated from photographs','chassis_dimensions_m':[.300,.245,.100],'wheel_visual':'mecanum rollers; simplified cylindrical contact remains','top_camera_mount':'virtual, no physical mast','basket_origin':'fixed ground collection station','collection_base_y':.25,'harvest_base_y':0.0,'chassis_reference_url':'https://www.hiwonder.com/products/large-metal-4wd-vehicle-chassis-black','fruit_scales':sizes,'maturity':maturity,'fruit_euler':angles,'protected_ripe_fruit':None if diverse else 0,'diversity_profile':('mobile-greenhouse-v6-bilateral' if mobile_layout_level=='moderate' else 'mobile-greenhouse-v1') if mobile else ('greenhouse-v1' if diverse else 'legacy'),'occlusion_profile':occlusion,'occlusion_leaf_retention':retention,'task_position_jitter_m':None if mobile else ([.025,.035,.020] if diverse else [.005]*3),'background_fruit_count':background_count,'base_model':'level four-wheel differential drive with passive vertical travel; parking brake during picking','arm_mount_height_m':BASE_Z,'soil_ridge_height_m':.13,'seed':seed,'fruit_positions':np.array(positions).tolist(),'ripe':ripe,'basket':basket.tolist(),'joint_names':JOINTS,'action_units':'radians, absolute joint position','fps':20}

def stem(world,start,end,rng,index,scale):
    start=np.asarray(start);end=np.asarray(end)
    body=add(world,'body',name=f'stem_{index}',pos=v(start),gravcomp='1',sleep='init')
    for axis in ['1 0 0','0 1 0']:
        add(body,'joint',type='hinge',axis=axis,stiffness='.8',damping='.015',armature='.00001',range='-.6 .6')
    arch=(start+end)/2+[0,rng.uniform(-.012,.012),rng.uniform(.035,.060)]
    prev=start
    for t in [.25,.5,.75,1.]:
        point=(1-t)**2*start+2*(1-t)*t*arch+t*t*end
        rod(body,prev-start,point-start,.0012,(.25,.37,.10,1),mass='.0002',contype='2',conaffinity='0',solref='.015 1');prev=point
    add(body,'site',name=f'stem_anchor_{index}',pos=v(end-start-[0,0,.022*scale]),size='.001',rgba='0 0 0 0')

def plant(world,pos,rng,index,occlusion_retention=None):
    x,y,z=pos
    for k in range(int(rng.integers(5,8))):
        a=2*np.pi*k/5+rng.uniform(-.45,.45);length=rng.uniform(.045,.10)
        end=np.array([x+length*np.cos(a),y+length*np.sin(a),z+rng.uniform(.055,.115)])
        if x>0 and -.21<end[1]<-.08 and end[0]<.39:
            if occlusion_retention is None or rng.random()>occlusion_retention:continue
        branch=add(world,'body',name=f'branch_{index}_{k}',pos=v(pos),gravcomp='1',sleep='init')
        for axis in ['1 0 0','0 1 0']:
            add(branch,'joint',type='hinge',axis=axis,stiffness='.025',damping='.002',armature='.000002',range='-.8 .8')
        rod(branch,[0,0,0],end-pos,.0014,(.22,.37,.10,1),mass='.001',contype='2',conaffinity='0',solref='.015 1')
        for spread in [-.95,0,.95]:
            angle=a+spread;size=str(rng.choice(['small','medium','large']))
            midpoint=end+np.array([.035*np.cos(angle),.035*np.sin(angle),0])
            if x>0 and -.25<midpoint[1]<.23 and midpoint[0]<.40 and midpoint[2]<.36:
                if occlusion_retention is None:
                    if abs(midpoint[1]+.145)<.055 and midpoint[0]<.37 and midpoint[2]<.30:continue
                elif rng.random()>occlusion_retention:continue
            leaf=add(branch,'body',name=f'leaf_{index}_{k}_{spread}',pos=v(end-pos),euler=v([rng.uniform(-.22,.22),rng.uniform(-.30,.22),angle]),gravcomp='1')
            add(leaf,'geom',type='mesh',mesh=f'leaf_{size}',material=f'leaf_{rng.choice(["dark","mid","light"])}',contype='0',conaffinity='0',mass='0')
            scale={'small':.95,'medium':1.2,'large':1.4}[size]
            geom(leaf,'ellipsoid',np.array([.025,0,.0015])*scale,np.array([.025,.015,.0025])*scale,[0,0,0,0],group='3',mass='.0003',contype='2',conaffinity='0',solref='.015 1',friction='.5 .005 .0001')

def fruit_visual(body,rng,mature,seeds=True,scale=1.,mesh="berry_mesh"):
    color={'ripe':[rng.uniform(.65,.85),.035,.045,1], 'turning':[.88,.34,.18,1], 'green':[.55,.65,.24,1]}[mature]
    material={'material':'berry_mat_'+mature} if seeds=='mesh' else {}
    add(body,'geom',type='mesh',mesh=mesh,rgba=v([1,1,1,1] if seeds=='mesh' else color),contype='0',conaffinity='0',group='2',**material)
    for k in range(5):
        a=k*2*np.pi/5
        geom(body,'ellipsoid',np.array([.008*np.cos(a),.008*np.sin(a),.021])*scale,np.array([.011,.003,.0015])*scale,[.17,.34,.07,1],euler=v([0,.2,a]),contype='0',conaffinity='0')
    if seeds=="mesh":
        add(body,'geom',type='mesh',mesh='berry_seeds',rgba='.88 .70 .24 1',contype='0',conaffinity='0',mass='0',group='4')
    elif seeds:
        for j in range(1,7):
            t=np.pi*j/8; radius=.017*np.sin(t)*(1+.32*np.cos(t))
            for k in range(9):
                a=2*np.pi*(k+j*.5)/9
                geom(body,'ellipsoid',np.array([(radius+.00025)*np.cos(a),(radius+.00025)*np.sin(a),.022*np.cos(t)])*scale,np.array([.00065,.00065,.001])*scale,[.88,.70,.24,1],contype='0',conaffinity='0')

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=0); args=p.parse_args()
    xml,meta=build(args.seed); (ROOT/'scene.xml').write_text(xml); (ROOT/'scene.json').write_text(json.dumps(meta,indent=2))
    print(ROOT/'scene.xml')
