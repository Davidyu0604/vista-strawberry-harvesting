"""Photo-reference visual fittings. Zero mass and no collision: do not silently alter dynamics."""
import numpy as np

def decorate(root,world,chassis,basket,crowns,seed,add,geom,rod,v):
    rng=np.random.default_rng(seed+604)
    visual={'contype':'0','conaffinity':'0','mass':'0'}
    def box(p,pos,size,color,**kw):return geom(p,'box',pos,size,color,**visual,**kw)
    def pipe(p,a,b,r=.004,color=(.22,.25,.25,1)):return rod(p,a,b,r,color,**visual)
    # Greenhouse end wall and a door frame establish an enclosed growing space.
    for x in [-1.64,1.04]:
        box(world,[x,.9,.27],[.003,2.2,.27],[.69,.80,.69,.20])
        pipe(world,[x,-1.29,.57],[x,3.1,.57],.025,(.52,.63,.53,1))
    for x in [-1.1,.5]:box(world,[x,3.115,.9],[.55,.003,.9],[.78,.88,.77,.23])
    for x in [-.70,.10]:pipe(world,[x,3.105,0],[x,3.105,1.85],.017,(.51,.57,.55,1))
    pipe(world,[-.70,3.105,1.85],[.10,3.105,1.85],.017,(.51,.57,.55,1))
    box(world,[-.3,3.115,2.0],[1.34,.003,.17],[.78,.88,.77,.14])
    pipe(world,[-1.64,3.10,.5],[1.04,3.10,.5],.008)
    # Irrigation barrel, feed line and small traps outside the arm's work area.
    geom(world,'cylinder',[.80,2.70,.22],[.14,.22],[.15,.29,.34,1],**visual)
    geom(world,'cylinder',[.80,2.70,.445],[.145,.008],[.10,.20,.23,1],**visual)
    for z in [.09,.34]:geom(world,'cylinder',[.80,2.70,z],[.143,.012],[.20,.35,.39,1],**visual)
    for x in [.42,-1.05]:
        pipe(world,[.80,2.70,.04],[x,2.70,.04],.008,(.035,.055,.04,1))
        pipe(world,[x,2.70,.04],[x,2.70,.135],.006,(.035,.055,.04,1))
    for x,y in [(.64,1.12),(-.86,1.70)]:
        pipe(world,[x,y,.13],[x,y,.55],.002,(.45,.38,.20,1))
        box(world,[x,y,.52],[.035,.001,.045],[.83,.68,.13,1])
    # Sparse five-petal flowers and buds; deterministic appearance-only RNG.
    for row,crown in crowns:
        if row==0 and -.30<crown[1]<.30:continue
        if rng.random()>.40:continue
        point=np.array(crown)+[rng.uniform(-.045,.045),rng.uniform(-.04,.04),rng.uniform(.065,.11)]
        pipe(world,crown,point,.0009,(.23,.39,.11,1))
        geom(world,'sphere',point,[.003],[.86,.66,.12,1],**visual)
        for a in np.linspace(0,2*np.pi,5,endpoint=False):
            geom(world,'ellipsoid',point+[.0045*np.cos(a),.0045*np.sin(a),0],[.004,.0025,.001],[.89,.89,.78,1],euler=v([0,0,a]),**visual)
