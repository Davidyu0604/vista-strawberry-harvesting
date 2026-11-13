"""Bounded, sequential five-episode batches with fixed scene split; no retries."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from recording_quality import atomic_json

from project_paths import PROJECT_ROOT as ROOT, project_source, source_files

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,choices=range(4),required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    sources=['vista.py','scene.py','peduncle.py','env.py','run.py','recording_quality.py','visual_maturity.py','peduncle_prompt.txt','vista_prompt.txt','chassis.py']
    hashes={name:hashlib.sha256(project_source(name).read_bytes()).hexdigest() for name in sources}
    plan={'train_seeds':list(range(2,17)),'validation_seeds':list(range(17,22)),
          'held_out_test_seeds':None,'attempts_per_seed':1,'max_calls_per_episode':32,'total_call_limit':640,
          'viewer':True,'record_images':True,'source_sha256':hashes,'note':'Pilot, not final test set; failed attempts retained; no automatic retries.'}
    manifest=out/'plan.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=plan:raise RuntimeError('Plan/source changed; refuse mixing versions')
    else:atomic_json(manifest,plan)
    results=[]
    for seed in range(2+a.batch*5,7+a.batch*5):
        split='train' if seed<17 else 'validation';dest=out/f'{split}-seed-{seed}'
        if dest.exists():raise FileExistsError(dest)
        env=os.environ.copy();env.update(MUJOCO_GL='glfw',PYOPENGL_PLATFORM='glx',PYTHONPATH='',PYTHONNOUSERSITE='1')
        cmd=[sys.executable,'-u',str(project_source('vista.py')),'--viewer','--peduncle-test','--seed',str(seed),'--max-calls','32','--record-images','--out',str(dest)]
        print(f'START seed={seed} split={split}',flush=True);start=time.monotonic()
        with (out/f'seed-{seed}.log').open('w') as log:
            code=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        report=json.loads((dest/'report.json').read_text()) if (dest/'report.json').exists() else {}
        quality=json.loads((dest/'quality.json').read_text()) if (dest/'quality.json').exists() else {}
        row={'seed':seed,'split':split,'exit_code':code,'wall_seconds':time.monotonic()-start,
             'success':report.get('success',False),'calls':report.get('model_calls',0),'stop_reason':report.get('stop_reason'),
             'error':report.get('error'),'integrity_passed':quality.get('integrity_passed',False),
             'eligible':quality.get('eligible_for_training_export_review',False),'path':str(dest)}
        results.append(row);atomic_json(out/f'batch-{a.batch+1}.json',results);print(json.dumps(row),flush=True)
        if code or report.get('stop_reason') in ('interrupted','error') or not quality.get('integrity_passed'):
            print('STOP: runtime/interruption/data integrity failure; review before proceeding.',flush=True);break

if __name__=='__main__':main()
