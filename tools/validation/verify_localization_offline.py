"""Replay recorded RGB-D through tool dispatch; no simulator or network client."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vista import Session


def replay(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in sorted(root.glob('episode-*-seed-*')):
        events = [json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
        observations = {e['frame_id']:e for e in events if e['kind']=='observation'}
        receipts = {e['call']:e['result'] for e in events if e['kind']=='receipt'}
        for event in events:
            if event['kind'] != 'response': continue
            for call in event['message'].get('tool_calls', []):
                if call['function']['name'] != 'locate_pixel': continue
                args = json.loads(call['function']['arguments']); frame = args['frame_id']
                state = observations[frame]
                session = Session(SimpleNamespace(meta={'harvest_mechanism':True,'upright_stalk_prior':True}),
                                  SimpleNamespace(out=out), None)
                session.frames[frame] = {c:episode/'observations'/f'{frame}-{c}.jpg' for c in ('top','wrist','harvest_close')}
                session.depth_frames = {(frame,c):episode/'observations'/f'{frame}-{c}-depth.npz' for c in session.frames[frame]}
                direct, images = session.dispatch('locate_pixel', args, frame, state, 0)
                expected = receipts[event['call']]
                assert direct.get('status') == expected.get('status')
                if direct.get('status') == 'rejected':
                    assert direct['reason'] == expected['reason'] and len(images)==1
                    assert session.locked_target is None
                    encoded = images[0]['image_url']['url'].split(',',1)[1]
                    (out/f'{episode.name}-call-{event["call"]}-feedback.jpg').write_bytes(base64.b64decode(encoded))
                else:
                    np.testing.assert_allclose(direct['locked_target_m'],expected['locked_target_m'],atol=1e-12)
                    assert direct['target_confirmed'] == expected['target_confirmed']
                # Clear only local replay state, then select the same pixel via displayed crop coordinates.
                session.locked_target = None; session.target_camera = None; session.target_confirmed = False
                x,y=args['x'],args['y']; x0,y0=max(0,x-25),max(0,y-25)
                region=[x0,y0,min(60,480-x0),min(60,360-y0)]
                inspected,_=session.dispatch('inspect',{'frame_id':frame,'camera':args['camera'],'region':region},frame,state,0)
                m=inspected['crop_mapping']; scale=m['scale']; ox,oy=m['display_origin_xy']
                local={'frame_id':frame,'crop_id':inspected['crop_id'],
                       'x':ox+(x-x0)*scale+scale//2,'y':oy+(y-y0)*scale+scale//2}
                next_frame=frame+'-replay-next'
                session.frames[next_frame]=session.frames[frame]
                for camera in session.frames[frame]:
                    session.depth_frames[(next_frame,camera)]=session.depth_frames[(frame,camera)]
                local['frame_id']=next_frame
                session.calls += 1
                mapped,_=session.dispatch('locate_crop',local,next_frame,state,0)
                assert mapped.pop('crop_selection')['source_pixel'] == [x,y]
                mapped['frame_id']=frame
                assert mapped == direct
                session.calls += 1
                try: session.dispatch('locate_crop',local,next_frame,state,0)
                except ValueError as exc: assert 'expired' in str(exc)
                else: raise AssertionError('Expired crop accepted')
                rows.append({'episode':episode.name,'call':event['call'],'pixel':[x,y],
                             'status':direct.get('status','localized'),'fit':direct['stalk_fit'],
                             'same_as_recording':True,'crop_equivalent':True})
    assert len(rows)==17 and sum(r['status']=='localized' for r in rows)==4
    report={'episodes':5,'recorded_selections':len(rows),'accepted':4,'rejected':13,
            'all_original_outcomes_preserved':True,'all_crop_selections_equivalent':True,
            'network_calls':0,'limitations':'Recorded points are replayed, not new model choices. No new harvest or success-rate evaluation.',
            'results':rows}
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings',type=Path,default=PROJECT_ROOT/'reports/02_批量评估/greenhouse-stability-1794204520156356271')
    parser.add_argument('--out',type=Path,default=PROJECT_ROOT/'reports/01_控制验证/localization-feedback-offline')
    args=parser.parse_args(); result=replay(args.recordings,args.out)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False))
