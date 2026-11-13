"""Hiwonder 300 x 245 x 100 mm chassis envelope; internal layout is photo-derived."""
import numpy as np
WHEEL_RADIUS=.05
TRACK=.20
BASE_Z=.1024

def build_chassis(root,world,base,add,geom,rod,v):
    c=add(world,'body',name='chassis',pos='0 0 0')
    for name,axis in [('x','1 0 0'),('y','0 1 0'),('z','0 0 1')]:
        kw={'range':'-.01 .015'} if name=='z' else {}
        add(c,'joint',name='base_'+name,type='slide',axis=axis,damping='2',**kw)
    add(c,'joint',name='base_yaw',type='hinge',axis='0 0 1',damping='1')
    # 1.3 kg frame/motors + four 0.15 kg wheels = bare chassis 1.9 kg.
    # Add estimated 0.35 kg electronics; arm mass comes from its own model.
    add(c,'inertial',pos='0 -.06 .065',mass='1.65',diaginertia='.013 .016 .010')
    black=[.055,.06,.065,1]; gray=[.50,.52,.53,1]
    def box(name,pos,size,color=black,collision=True):
        return geom(c,'box',pos,size,color,name=name,mass='0',contype='1' if collision else '0',conaffinity='1' if collision else '0')
    # Steel plates: collision surfaces coincide with visible surfaces.
    box('deck',[0,-.06,.087],[.075,.145,.003])
    box('bottom',[0,-.06,.051],[.075,.145,.002])
    for x in [-.073,.073]:
        for z in [.057,.079]:box(f'rail_{x}_{z}',[x,-.06,z],[.002,.145,.004])
        for y,half in [(-.161,.044),(.041,.044)]:box(f'side_{x}_{y}',[x,y,.068],[.002,half,.007])
    for y in [-.203,.083]:box(f'end_{y}',[0,y,.068],[.071,.002,.015])
    # SO101 mounting pad, clear of board stack behind it.
    box('arm_adapter',[.02,0,.095],[.048,.059,.005])
    world.remove(base);c.append(base)
    # Board and ports are behind the arm footprint, not underneath it.
    box('electronics_board',[0,-.145,.098],[.061,.043,.002],[.035,.20,.28,1])
    box('compute_module',[.017,-.148,.105],[.031,.027,.005],[.035,.10,.075,1])
    for y in [-.17,-.145,-.12]:box('usb_'+str(y),[-.048,y,.107],[.010,.009,.007],[.55,.57,.57,1])
    box('terminal_block',[.044,-.108,.105],[.014,.009,.005],[.40,.44,.45,1])
    for i in range(4):box('terminal_tab_'+str(i),[.034+i*.007,-.108,.112],[.002,.006,.002],[.65,.08,.04,1],False)
    # Low cable loops stay behind y=-.08, clear of the base and moving arm.
    for i,col in enumerate([[.75,.76,.72,1],[.55,.025,.025,1],[.025,.025,.025,1]]):
        pts=[[.025+.018*np.sin(np.pi*t)+i*.004,-.18+.07*t,.115+.018*np.sin(np.pi*t)] for t in np.linspace(0,1,25)]
        for j,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
            rod(c,a,b,.002 if i==0 else .0012,col,name=f'cable_{i}_{j}',mass='0',contype='0',conaffinity='0')
    for side,x in [('left',-.10),('right',.10)]:
        for end,y in [('front',.04),('rear',-.16)]:
            name=f'wheel_{side}_{end}';w=add(c,'body',name=name,pos=v([x,y,.05]))
            add(w,'joint',name=name,type='hinge',axis='1 0 0',damping='.02',armature='.001')
            geom(w,'cylinder',[0,0,0],[.05,.0225],[0,0,0,0],group='3',euler=v([0,np.pi/2,0]),mass='.15',friction='1.2 .005 .0001',condim='4',name=name+'_contact')
            vis={'mass':'0','contype':'0','conaffinity':'0'}
            # Rim and spokes sit axially outside the rollers: no spokes through rubber.
            for xx in [-.020,.020]:
                geom(w,'cylinder',[xx,0,0],[.014,.0015],gray,euler=v([0,np.pi/2,0]),**vis)
                for a,b in zip(np.linspace(0,2*np.pi,49)[:-1],np.linspace(0,2*np.pi,49)[1:]):
                    rod(w,[xx,.032*np.cos(a),.032*np.sin(a)],[xx,.032*np.cos(b),.032*np.sin(b)],.0017,gray,**vis)
                for a in np.linspace(0,2*np.pi,8,endpoint=False):
                    rod(w,[xx,.012*np.cos(a+.3),.012*np.sin(a+.3)],[xx,.038*np.cos(a),.038*np.sin(a)],.0025,gray,**vis)
            sign=1 if (side,end) in [('left','front'),('right','rear')] else -1
            for a in np.linspace(0,2*np.pi,8,endpoint=False):
                p=np.array([0,.040*np.cos(a),.040*np.sin(a)])
                axis=np.array([1.,-sign*np.sin(a),sign*np.cos(a)])/np.sqrt(2)
                q=np.sqrt((1+axis[2])/2)
                geom(w,'ellipsoid',p,[.009,.009,.023],[.065,.07,.07,1],quat=v([q,-axis[1]/(2*q),axis[0]/(2*q),0]),**vis)
            add(root.find('actuator'),'velocity',name=name,joint=name,kv='20',ctrlrange='-8 8',forcerange='-3 3')
    return c
