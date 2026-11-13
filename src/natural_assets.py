"""Code-native mesh/texture assets for soil, mulch and strawberry leaflets."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from project_paths import PROJECT_ROOT
ASSETS=PROJECT_ROOT/'assets/natural'

def obj(path, vertices, faces, uv=None):
    lines=['v '+' '.join(f'{a:.7g}' for a in p) for p in vertices]
    if uv is not None: lines += ['vt '+' '.join(f'{a:.7g}' for a in p) for p in uv]
    lines += ['f '+' '.join(f'{i+1}/{i+1}' if uv is not None else str(i+1) for i in f) for f in faces]
    path.write_text('\n'.join(lines)+'\n')
    return path

def textures():
    ASSETS.mkdir(exist_ok=True)
    rng=np.random.default_rng(180)
    n=512
    noise=np.zeros((n,n))
    for size,weight in [(8,9),(32,6),(128,4),(512,3)]:
        layer=Image.fromarray(np.uint8(rng.uniform(0,255,(size,size)))).resize((n,n),Image.Resampling.BILINEAR)
        noise+=weight*(np.asarray(layer,dtype=float)/127.5-1)
    soil=np.clip(np.array([105,82,54])+noise[:,:,None],0,255).astype('uint8')
    im=Image.fromarray(soil); d=ImageDraw.Draw(im)
    for _ in range(1800):
        x,y=rng.integers(0,n,2); r=int(rng.choice([1,1,1,2])); shade=int(rng.integers(-30,24))
        d.ellipse((x,y,x+r,y+r),fill=tuple(int(np.clip(c+shade,0,255)) for c in [105,82,54]))
    im.save(ASSETS/'soil.png')
    im=Image.new('RGB',(256,256),(120,145,92));d=ImageDraw.Draw(im)
    d.line([(0,128),(256,128)],fill=(178,188,121),width=3)
    for x in range(30,236,28):
        for side in [-1,1]: d.line([(x,128),(min(255,x+48),128+side*82)],fill=(151,170,106),width=2)
    im.filter(ImageFilter.GaussianBlur(.5)).save(ASSETS/'leaf.png')
    noise=rng.normal(0,2,(256,256,1))
    Image.fromarray(np.uint8(np.clip(np.array([30,34,29])+noise,0,255))).save(ASSETS/'mulch.png')
    return ASSETS

def leaf_mesh():
    # Closed thin serrated blade, raised central vein; separate visual and collision geometry.
    count=48; vertices=[[.025,0,.004]]; uv=[[.5,.5]]
    for i in range(count):
        a=2*np.pi*i/count
        tooth=1 if i%2==0 else .88
        x=.025+.025*np.cos(a)
        y=.016*np.sin(a)*tooth
        vertices.append([x,y,.0015*(abs(y)/.016)**1.5])
        uv.append([x/.05,.5+y/.034])
    vertices.append([.025,0,-.001]);uv.append([.5,.5])
    faces=[]
    for i in range(count):
        j=1+i;k=1+(i+1)%count
        faces += [(0,j,k),(count+1,k,j)]
    return obj(ASSETS/'leaf.obj',vertices,faces,uv)

def ridge_mesh(seed,row,centers,mulch=False):
    """Wavy visual ridge; mulch omits cells at real planting openings."""
    rng=np.random.default_rng(seed+row*137+912)
    xs=np.linspace(-.245,.245,45); ys=np.linspace(-1,3,241)
    vertices=[];uv=[]
    for y in ys:
        for x in xs:
            height=.13 if abs(x)<.16 else max(.008,.13*(.245-abs(x))/.085)
            wave=.0012*np.sin(110*x+12*y)+.0014*np.sin(50*y+30*x)
            z=height+wave+(0.001 if mulch else -.001)
            vertices.append([x,y,z]);uv.append([x*4,y*2])
    faces=[];nx=len(xs)
    for j in range(len(ys)-1):
        for i in range(nx-1):
            x=(xs[i]+xs[i+1])/2;y=(ys[j]+ys[j+1])/2
            if mulch:
                if abs(x)>.225:continue
                if any((x-cx)**2+(y-cy)**2<.027**2 for cx,cy in centers):continue
            a=j*nx+i;b=a+1;c=a+nx;d=c+1
            faces += [(a,b,c),(b,d,c)]
    return obj(ASSETS/f'{"mulch" if mulch else "soil"}_{seed}_{row}.obj',vertices,faces,uv)
