"""Image-derived body proposals. No simulator state, segmentation IDs or maturity truth."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def body_color_masks(rgb):
    """Include orange body context, without treating orange as ripe evidence."""
    hsv=np.asarray(rgb.convert('HSV'));h,s,v=hsv.transpose(2,0,1)
    saturated=(s>=115)&(v>=30)
    return (((h<8)|(h>244))&saturated,
            ((h<23)|(h>244))&saturated)


def body_regions(mask):
    """Separate connected lobes across thin bridges, without claiming identity."""
    labels,_=ndimage.label(mask)
    for index,box in enumerate(ndimage.find_objects(labels),1):
        if box is None:continue
        region=labels[box]==index
        filled=np.pad(ndimage.binary_fill_holes(region),1)
        inside=ndimage.distance_transform_edt(filled)[1:-1,1:-1]
        cores,n=ndimage.label(inside>=max(2.5,.65*inside.max()))
        sizes=np.bincount(cores.ravel(),minlength=n+1)
        keep=np.flatnonzero(sizes>=9);keep=keep[keep!=0]
        if len(keep)<2:
            yield box,region
            continue
        cores[~np.isin(cores,keep)]=0
        nearest=ndimage.distance_transform_edt(cores==0,return_distances=False,return_indices=True)
        assigned=cores[tuple(nearest)]
        for key in keep:
            fragment=region&(assigned==key)
            yy,xx=np.where(fragment)
            y0,y1=int(yy.min()),int(yy.max())+1;x0,x1=int(xx.min()),int(xx.max())+1
            yield (slice(box[0].start+y0,box[0].start+y1),slice(box[1].start+x0,box[1].start+x1)),fragment[y0:y1,x0:x1]


def propose_bodies(rgb, depth, calibration, base_pose, unproject, limit=16):
    """Red/orange body regions with a locally coherent RGB-D sample.

    Color proposes regions only: the model must still inspect whole-fruit maturity.
    Small/occluded/mixed-color fruit may be missed; coverage is never certified here.
    """
    red_mask,mask=body_color_masks(rgb)
    proposals=[];outside=0;small=0
    for box,region in body_regions(mask):
        area=int(region.sum())
        red_area=int((red_mask[box]&region).sum())
        # Orange extends existing red evidence; it does not propose every
        # brown planter or yellow robot highlight as a new fruit.
        if red_area<9:small+=1;continue
        # Padding ensures image-border regions cannot select their exposed edge.
        inside=ndimage.distance_transform_edt(np.pad(region,1))[1:-1,1:-1]
        yy,xx=np.where(region)
        order=np.argsort(-inside[yy,xx],kind='stable')
        point=None;pixel=None
        for i in order:
            y=int(yy[i]+box[0].start);x=int(xx[i]+box[1].start)
            if inside[yy[i],xx[i]]<1.4:break
            patch=depth[max(0,y-1):y+2,max(0,x-1):x+2]
            red=mask[max(0,y-1):y+2,max(0,x-1):x+2]
            coherent=red&np.isfinite(patch)&(np.abs(patch-depth[y,x])<.01)
            if coherent.sum()<5:continue
            try:point=unproject(depth,calibration,x,y)
            except ValueError:continue
            pixel=[x,y];break
        if point is None:
            # A thin distant region may have no coherent interior. Exclude it
            # only if EVERY red pixel has finite depth and lies outside reach.
            # Mixed/unknown depth remains unresolved instead of claiming absence.
            samples=depth[box][region]
            all_outside=bool(np.isfinite(samples).all() and np.all(samples>.01))
            if all_outside:
                for ry,rx in zip(yy,xx):
                    y=int(ry+box[0].start);x=int(rx+box[1].start)
                    if depth[y,x]>=2:continue
                    measured=unproject(depth,calibration,x,y)
                    if .12<=measured[2]<=.46 and np.linalg.norm(measured[:2]-np.asarray(base_pose)[:2])<=.56:
                        all_outside=False;break
            if all_outside:outside+=1;continue
            i=order[0];pixel=[int(xx[i]+box[1].start),int(yy[i]+box[0].start)]
            distance=None
        else:
            distance=float(np.linalg.norm(point[:2]-np.asarray(base_pose)[:2]))
            if not .12<=point[2]<=.46 or distance>.56:outside+=1;continue
        proposals.append({'pixel':pixel,'bbox_xyxy':[box[1].start,box[0].start,box[1].stop-1,box[0].stop-1],
                          'red_area_pixels':red_area,
                          'body_area_pixels':area,'candidate_color_support':'red_or_orange',
                          'surface_m':point.tolist() if point is not None else None,
                          'distance_xy_m':distance,'depth_verified':point is not None})
    proposals.sort(key=lambda p:p['distance_xy_m'] if p['distance_xy_m'] is not None else -1)
    omitted=max(0,len(proposals)-limit)
    proposals=proposals[:limit]
    for i,p in enumerate(proposals):p['proposal_id']=f'P{i+1}'
    return {'proposals':proposals,'omitted_by_limit':omitted,'outside_measured_envelope':outside,
            'small_regions_ignored':small,'coverage_certified':False,
            'method':'RGB regions with >=9 red pixels, extended through orange body context; separate thin bridges; interior RGB-D sample; orange is NOT ripe evidence; model must review whole-body maturity and missing ripe fruit'}


def draw_proposals(rgb, batch):
    """Frame-local labels, never scene IDs; leave original camera image available."""
    image=rgb.convert('RGB').resize((rgb.width*2,rgb.height*2),Image.Resampling.NEAREST)
    draw=ImageDraw.Draw(image)
    for p in batch['proposals']:
        x0,y0,x1,y1=[v*2 for v in p['bbox_xyxy']]
        draw.rectangle((x0,y0,x1,y1),outline='yellow',width=2)
        x,y=[v*2 for v in p['pixel']]
        draw.ellipse((x-3,y-3,x+3,y+3),outline='cyan',width=1)
        label=p['proposal_id'];ty=max(0,y0-13)
        draw.rectangle((x0,ty,x0+len(label)*7+3,ty+12),fill='black')
        draw.text((x0+2,ty),label,fill='yellow')
    return image


def refresh_proposals(original, rgb, depth, calibration, base_pose, unproject):
    """Remeasure original image regions without reordering their labels.

    Caller must verify the original/current camera pose before accepting review.
    A depth hole is retained, never treated as proof that a fruit is absent.
    """
    from copy import deepcopy
    batch=deepcopy(original)
    red,body=body_color_masks(rgb)
    for p in batch['proposals']:
        support=body if p.get('candidate_color_support')=='red_or_orange' else red
        x0,y0,x1,y1=p['bbox_xyxy']
        pixels=[p['pixel']]
        if p['surface_m'] is None:
            mask=support[y0:y1+1,x0:x1+1]
            inside=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
            yy,xx=np.where(inside>=1.4)
            pixels += [[int(x+x0),int(y+y0)] for x,y in zip(xx,yy)]
        point=None
        for x,y in pixels:
            patch=depth[max(0,y-1):y+2,max(0,x-1):x+2]
            coherent=np.isfinite(patch)&(np.abs(patch-depth[y,x])<.01)
            if p['surface_m'] is None:coherent &= support[max(0,y-1):y+2,max(0,x-1):x+2]
            if coherent.sum()<5:continue
            try:measured=unproject(depth,calibration,x,y)
            except ValueError:continue
            distance=float(np.linalg.norm(measured[:2]-np.asarray(base_pose)[:2]))
            if not .12<=measured[2]<=.46 or distance>.56:continue
            point=measured;p['pixel']=[x,y];break
        p.update(surface_m=point.tolist() if point is not None else None,
                 depth_verified=point is not None,
                 distance_xy_m=float(np.linalg.norm(point[:2]-np.asarray(base_pose)[:2])) if point is not None else None)
    batch['method']='Current RGB-D remeasurement of original numbered image regions; review whole image for missed fruit'
    return batch


def normalize_review_frame(args, frame):
    """Accept the shared tool frame alias only when every supplied ID is current."""
    if 'basis_frame_id' not in args:return args
    if args['basis_frame_id']!=frame or ('frame_id' in args and args['frame_id']!=frame):
        raise ValueError(f'Review frame IDs must both equal current frame_id {frame}; stale or conflicting IDs are not accepted')
    return {**{k:v for k,v in args.items() if k!='basis_frame_id'},'frame_id':frame}


def reviewed_candidates(args, batch, frame, rgb):
    """Validate complete, frame-bound semantic review before changing the queue."""
    from local_targets import body_pixel_evidence
    if set(args)!={'frame_id','coverage','candidates','missing_visible_ripe'} or args['frame_id']!=frame:
        expected={'frame_id','coverage','candidates','missing_visible_ripe'}
        raise ValueError(f'Proposal review requires frame_id={frame} and exactly {sorted(expected)}; '
                         f'missing={sorted(expected-set(args))}, extra={sorted(set(args)-expected)}')
    if not isinstance(args['coverage'],str) or not 1<=len(args['coverage'])<=1500:
        raise ValueError('Explain visual coverage')
    if type(args['missing_visible_ripe']) is not bool or not isinstance(args['candidates'],list):
        raise ValueError('Invalid proposal review')
    by_id={p['proposal_id']:p for p in batch['proposals']};seen=set();measured=[]
    blocked=[]
    if args['missing_visible_ripe']:
        from perception_recovery import MISSING_BLOCK
        blocked.append(MISSING_BLOCK)
    if batch['omitted_by_limit']:blocked.append('Proposal limit omitted regions; finish, coverage unresolved')
    for item in args['candidates']:
        if (not isinstance(item,dict) or set(item)!={'proposal_id','maturity','visibility','reason'}
                or not isinstance(item['proposal_id'],str) or item['proposal_id'] not in by_id or item['proposal_id'] in seen
                or item['maturity'] not in ('ripe','turning','green','uncertain','not_fruit')
                or item['visibility'] not in ('clear','partial','occluded')
                or not isinstance(item['reason'],str) or not 1<=len(item['reason'])<=600):
            raise ValueError('Review each displayed proposal exactly once with visual evidence; do not send x/y. Allowed proposal_ids: '+str(list(by_id))+('. Submit candidates=[]; do not invent labels or localize before survey.' if not by_id else ''))
        seen.add(item['proposal_id']);p=by_id[item['proposal_id']]
        if p['surface_m'] is None:
            if item['maturity'] not in ('not_fruit','green','turning'):
                blocked.append(f"{p['proposal_id']} has unresolved body depth; finish")
            continue
        x,y=p['pixel'];evidence=body_pixel_evidence(rgb,x,y)
        measured.append({**item,'x':x,'y':y,'surface_m':p['surface_m'],'source_frame':frame,
                         'body_pixel_verified':evidence['passed'] if item['maturity'] in ('ripe','uncertain') else True,'body_pixel_check':evidence,
                         'semantic_unverified':item['maturity']=='uncertain'})
    if seen!=set(by_id):raise ValueError('Review ALL displayed proposals; omitted candidates cannot be skipped')
    return measured,'; '.join(blocked) or None
