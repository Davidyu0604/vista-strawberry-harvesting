"""Bounded RGB-D stalk-point proposals for a visually measured fruit body."""
import numpy as np
from scipy import ndimage
from PIL import ImageDraw


def propose_stalk_points(rgb,depth,calibration,body,fitter,max_samples=32,max_proposals=4):
    h,w=depth.shape;v,u=np.indices(depth.shape)
    focal=h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    local=np.stack(((u+.5-w/2)*depth/focal,-(v+.5-h/2)*depth/focal,-depth),axis=-1)
    xyz=local@np.asarray(calibration['rotation']).T+calibration['position']
    delta=xyz-np.asarray(body);color=np.asarray(rgb,dtype=float)
    mask=(np.isfinite(depth)&(depth>.01)&(depth<2)&
          (np.linalg.norm(delta[:,:,:2],axis=-1)<.035)&(delta[:,:,2]>.005)&(delta[:,:,2]<.085)&
          (color[:,:,1]>1.15*color[:,:,0])&(color[:,:,1]>1.15*color[:,:,2]))
    yy,xx=np.where(mask)
    info={'green_pixels':len(xx),'sampled_pixels':0,'fit_passes':0,'proposals':[],
          'coverage_certified':False,'note':'Image-derived fitted segments, not semantic stalk identity. Empty proposals do not prove no target.'}
    if not len(xx):return info
    # Spread bounded samples across the observed region, preferring interior pixels.
    pixels=np.column_stack((xx,yy));interior=ndimage.distance_transform_edt(mask)[yy,xx]
    selected=[];nearest=np.full(len(xx),np.inf)
    for _ in range(min(max_samples,len(xx))):
        index=int(np.argmax(interior if not selected else nearest*(1+np.minimum(interior,3)*.1)))
        selected.append(index);nearest=np.minimum(nearest,np.sum((pixels-pixels[index])**2,axis=1));nearest[selected]=-1
    successful=[]
    for index in selected:
        x,y=map(int,pixels[index]);info['sampled_pixels']+=1
        try:target,fit=fitter(xyz[y,x],depth,calibration,rgb)
        except ValueError:continue
        offset=target-np.asarray(body)
        if np.linalg.norm(offset[:2])>.035 or not -.005<=offset[2]<=.085:continue
        info['fit_passes']+=1
        successful.append({'pixel':[x,y],'target_m':np.asarray(target).tolist(),'axis_world':fit['axis_world'],
                           'visible_length_m':fit['visible_length_m'],'samples':fit['samples'],
                           'interior_pixels':float(interior[index])})
    successful.sort(key=lambda p:(-p['samples'],-p['interior_pixels'],p['pixel']))
    for item in successful:
        # Merge nearby fits on the same thin axis; no simulator identity used.
        if any(np.linalg.norm(np.asarray(item['target_m'])-other['target_m'])<.008 and
               abs(np.dot(item['axis_world'],other['axis_world']))>.96 for other in info['proposals']):continue
        info['proposals'].append(item)
        if len(info['proposals'])==max_proposals:break
    return info


def mark_stalk_points(image,mapping,proposals):
    draw=ImageDraw.Draw(image);sx,sy,_,_=mapping['source_region_xywh']
    ox,oy=mapping['display_origin_xy'];scale=mapping['scale']
    for item in proposals:
        px,py=item['pixel'];x=ox+(px-sx)*scale;y=oy+(py-sy)*scale
        draw.ellipse((x-4,y-4,x+4,y+4),outline='yellow',width=2)
        label=item['proposal_id'];left=min(image.width-25,max(0,x+7));top=min(image.height-14,max(0,y-14))
        draw.line((x+4,y,left,top+7),fill='yellow',width=1)
        draw.rectangle((left,top,left+23,top+13),fill='black');draw.text((left+2,top),label,fill='yellow')
    return image
