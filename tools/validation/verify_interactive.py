"""Physical probes for spring vegetation and a formerly-background fruit."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json,time,xml.etree.ElementTree as ET
import numpy as np
import env as module
import mujoco
from scene import ROOT
from PIL import Image

def leaf_probe():
    original=module.build
    def build(seed):
        xml,meta=original(seed);root=ET.fromstring(xml)
        body=ET.SubElement(root.find('worldbody'),'body',name='test_probe',mocap='true',pos='0 0 2')
        ET.SubElement(body,'geom',type='sphere',size='.009',rgba='.1 .4 .9 1',contype='1',conaffinity='2')
        return ET.tostring(root,encoding='unicode'),meta
    module.build=build
    e=module.StrawberryEnv(0);module.build=original
    try:
        for _ in range(40):e.step(e.last_action)
        options=[]
        for g in range(e.model.ngeom):
            b=int(e.model.geom_bodyid[g]);name=e.model.body(b).name
            if name.startswith('leaf_') and e.model.geom_contype[g]==2:
                p=e.data.geom_xpos[g]
                clearance=np.min(np.linalg.norm(e.data.xpos[e.fruit_bodies]-p,axis=1))
                options.append((clearance,g))
        _,g=max(options);leaf=int(e.model.geom_bodyid[g]);branch=int(e.model.body_parentid[leaf]);j=int(e.model.body_jntadr[branch]);qa=e.model.jnt_qposadr[j:j+2]
        baseline=e.data.qpos[qa].copy();center=e.data.geom_xpos[g].copy();peak=0.;contacts=0
        probe=e.model.body('test_probe').id;mid=int(e.model.body_mocapid[probe])
        for t in np.linspace(0,1,35):
            e.data.mocap_pos[mid]=center+[0,0,.04-.07*t];e.step(e.last_action)
            peak=max(peak,float(np.linalg.norm(e.data.qpos[qa]-baseline)))
            contacts+=sum(leaf in e.model.geom_bodyid[c.geom] and probe in e.model.geom_bodyid[c.geom] for c in e.data.contact)
        e.data.mocap_pos[mid]=[0,0,2]
        for _ in range(80):e.step(e.last_action)
        residual=float(np.linalg.norm(e.data.qpos[qa]-baseline))
        assert contacts>0 and peak>.02,(contacts,peak)
        assert residual<peak*.5,(peak,residual)
        assert not e.events
        return {'leaf':e.model.body(leaf).name,'probe_contact_samples':contacts,'peak_deflection_rad':peak,'residual_rad':residual,'warnings':sum(w.number for w in e.data.warning)}
    finally:e.close()

def background_pick():
    e=module.StrawberryEnv(0)
    try:
        choices=[i for i,p in enumerate(e.anchors) if i>=8 and e.meta['ripe'][i] and .28<p[0]<.42]
        i=min(choices,key=lambda i:e.anchors[i,0]);p=e.anchors[i].copy();e.target=i
        # Independent fixture: park beside the selected plant before starting dynamics.
        e.data.qpos[e.base_qa]=[0,p[1]+.12,0];e.park_pose=e.base_pose()
        mujoco.mj_forward(e.model,e.data)
        q,err=e.ik(np.array([.19,p[1]-.03,.33]),1)
        e.data.qpos[e.qa]=q;e.data.ctrl[:6]=q;e.last_action=q.copy();mujoco.mj_forward(e.model,e.data)
        for _ in range(30):e.step(e.last_action)
        p=e.fruit_pos(i)
        def move(pos,grip,n):
            q,error=e.ik(pos,grip);assert error<.006,error
            for _ in range(n):e.step(q)
        move(p-[.055,0,0],1,55);move(p,1,55);move(p,-.15,55)
        bilateral=e.contacts(i)
        move(p-[.085,0,0],-.15,80)
        assert not e.attached[i],(i,bilateral,e.fruit_pos(i),e.events)
        Image.fromarray(e.render('wrist',640,480)).save(ROOT/'reports/03_场景调试/performance-v11/background-pull.png')
        return {'fruit_index':i,'maturity':e.meta['maturity'][i],'bilateral_contact_before_pull':bilateral,'events':e.events,'warnings':sum(w.number for w in e.data.warning),'fixture':'base initialized beside plant; this test does not validate autonomous travel to plant'}
    finally:e.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('check',choices=['leaf','fruit']);a=p.parse_args()
    result=leaf_probe() if a.check=='leaf' else background_pick()
    (ROOT/f'reports/03_场景调试/performance-v11/{a.check}-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
