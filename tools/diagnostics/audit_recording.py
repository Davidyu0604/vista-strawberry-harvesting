"""Read-only diagnostic recording audit, including every camera image."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image

def audit(path):
    if (path/'recording.json').exists():
        from recording_quality import audit_recording
        result=audit_recording(path);print(json.dumps(result,indent=2));return result
    report=json.loads((path/'report.json').read_text())
    rows=[json.loads(x) for x in (path/'trajectory.jsonl').read_text().splitlines()]
    assert rows and len(rows)==report['frames']
    assert report['formal_training_demo'] is False
    assert [r['frame'] for r in rows]==list(range(len(rows)))
    times=np.array([r['timestamp'] for r in rows])
    assert np.allclose(np.diff(times),.05,atol=1e-8)
    state=np.array([r['state'] for r in rows]); action=np.array([r['action'] for r in rows])
    assert state.shape==action.shape==(len(rows),6)
    assert np.isfinite(state).all() and np.isfinite(action).all()
    assert np.max(np.abs(np.diff(action,axis=0)))<=.04000001
    decoded=0
    for row in rows:
        assert set(row['images'])=={'top','wrist'}
        for image_path in row['images'].values():
            with Image.open(path/image_path) as im:
                im.load(); assert im.size==(320,240) and im.mode=='RGB'
            decoded+=1
    for event in report['events']:
        index=int(np.searchsorted(times,event['time'],side='right')-1)
        assert rows[index]['phase']=='pull',(event,rows[index]['phase'])
        assert event['bilateral_contact']
    assert not any(not row['attached'] for row in rows if row['phase'] in ['pregrasp','approach','grasp'])
    if report['success']:
        assert any(e['fruit']==report['target'] for e in report['events'])
        pos=np.array(report['terminal_fruit_position']); box=np.array(report.get('terminal_basket_position',report['basket']))
        assert abs(pos[0]-box[0])<.066 and abs(pos[1]-box[1])<.048
        assert box[2]+.012<pos[2]<box[2]+.05
    out={'status':'passed','frames':len(rows),'decoded_images':decoded,'duration_seconds':len(rows)*.05,'success':report['success'],'note':'Diagnostic data; no policy evaluation or formal training export.'}
    (path/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);audit(p.parse_args().path)
