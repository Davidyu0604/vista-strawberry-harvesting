"""Conservative visual association of a small body fragment, never scene IDs."""
import numpy as np
from visual_proposals import body_regions


def check_duplicate_links(links, claims, batch, wrist_view):
    """Model association must agree with RGB-D and a separate camera body region.

    This is evidence for association, not an identity oracle: touching/occluded
    fruit can still be ambiguous and the model must retain uncertainty then.
    """
    if not isinstance(links, list) or len(links) > 8:
        raise ValueError('duplicate_links must be a list of at most eight associations')
    if not links:return []
    if wrist_view is None:raise ValueError('Duplicate association requires fresh wrist RGB-D')
    rgb, sample = wrist_view
    h,s,v = np.asarray(rgb.convert('HSV')).transpose(2,0,1)
    mask = ((h<8)|(h>244))&(s>=115)&(v>=30)
    labels = np.zeros(mask.shape, dtype=int)
    for index,(box,region) in enumerate(body_regions(mask),1):
        labels[box][region] = index
    by_id = {p['proposal_id']:p for p in batch['proposals']}
    claim = {p['proposal_id']:p for p in claims}
    sources=set(); parents=set(); checks=[]
    for link in links:
        if (not isinstance(link,dict) or set(link)!={'fragment_id','parent_id','reason'}
                or not isinstance(link['reason'],str) or not 1<=len(link['reason'].strip())<=600):
            raise ValueError('Explain duplicate fragment and parent association using both views')
        src,dst=link['fragment_id'],link['parent_id']
        if (not isinstance(src,str) or not isinstance(dst,str) or src==dst
                or src not in by_id or dst not in by_id or src in sources):
            raise ValueError('Duplicate associations require unique existing fragment and parent IDs')
        sources.add(src);parents.add(dst)
        if (claim[src]['maturity'] not in ('uncertain','ripe')
                or claim[dst]['maturity']!='ripe'
                or any(claim[k]['visibility']=='occluded' for k in (src,dst))):
            raise ValueError('Retain fragment maturity; duplicate parent must remain visually ripe and visible')
        a,b=by_id[src],by_id[dst]
        area_a,area_b=a.get('red_area_pixels',0),b.get('red_area_pixels',0)
        if not (0<area_a<=.2*area_b and area_b>=120):
            raise ValueError('Only a small fragment may associate with a larger body, not two full fruit regions')
        if a['surface_m'] is None or b['surface_m'] is None:
            raise ValueError('Duplicate association requires both body depths')
        distance=float(np.linalg.norm(np.asarray(a['surface_m'])-b['surface_m']))
        if not np.isfinite(distance) or distance>.020:
            raise ValueError('Duplicate fragment and parent exceed 20 mm')
        ax,ay,ax1,ay1=a['bbox_xyxy'];bx,by,bx1,by1=b['bbox_xyxy']
        if max(0,ax-bx1,bx-ax1,ay-by1,by-ay1)>6:
            raise ValueError('Duplicate image regions are not adjacent')
        pixels=[];regions=[];errors=[]
        for p in (a,b):
            point=(np.asarray(p['surface_m'])-sample['position'])@sample['rotation']
            if not np.isfinite(point).all() or point[2]>=-.01:
                raise ValueError('Duplicate point behind wrist camera')
            focal=rgb.height/(2*np.tan(np.deg2rad(float(sample['fovy']))/2))
            x,y=np.rint([rgb.width/2-.5+focal*point[0]/-point[2],
                         rgb.height/2-.5-focal*point[1]/-point[2]]).astype(int)
            if not (0<=x<rgb.width and 0<=y<rgb.height):
                raise ValueError('Duplicate points must both be visible in wrist image')
            error=float(abs(sample['depth'][y,x]+point[2]))
            if not np.isfinite(error) or error>.004:
                raise ValueError('Wrist depth disagrees with duplicate surface; association unresolved')
            pixels.append([int(x),int(y)]);regions.append(int(labels[y,x]));errors.append(error)
        if regions[0]==0 or regions[0]!=regions[1]:
            raise ValueError('Duplicate points are not on the same wrist red body region')
        checks.append({**link,'distance_m':distance,'wrist_pixels':pixels,'wrist_depth_errors_m':errors})
    if sources & parents:raise ValueError('Duplicate associations cannot form chains or cycles')
    return checks
