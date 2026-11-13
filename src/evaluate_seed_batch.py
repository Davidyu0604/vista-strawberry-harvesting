"""Frozen full-scene diagnostic batch: no retries, at most 192 model calls."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from collections import Counter
from pathlib import Path
from recording_quality import atomic_json
from project_paths import PROJECT_ROOT as ROOT, project_source, source_files
SEEDS=[7,8,9,10,11]


def hashes():
    paths=source_files()
    if (ROOT/'vista-guide.md').exists():paths.append(ROOT/'vista-guide.md')
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def summarize(dest):
    report=json.loads((dest/'report.json').read_text()) if (dest/'report.json').exists() else {}
    quality=json.loads((dest/'quality.json').read_text()) if (dest/'quality.json').exists() else {}
    events=[json.loads(l) for l in (dest/'vista-events.jsonl').read_text().splitlines()] if (dest/'vista-events.jsonl').exists() else []
    rejected=Counter();searches=0;last_reason=None;actions=Counter()
    for event in events:
        if event['kind']!='receipt':continue
        result=event.get('result')
        for item in result if isinstance(result,list) else [result]:
            if not isinstance(item,dict):continue
            if item.get('status')=='rejected':
                reason=item.get('reason','unknown');last_reason=reason
                key=('maturity' if 'maturity' in reason.lower() else 'approach_blocked' if 'Approach blocked' in reason else 'station_limit' if 'Station perception' in reason else 'localization' if any(t in reason for t in ('stalk','segment','Stalk')) else 'other')
                rejected[key]+=1
            if item.get('status')=='executed':
                stage=item.get('stage') or item.get('action',{}).get('action','unknown');actions[stage]+=1
                if stage=='search':searches+=1
    return {'seed':report.get('seed'),'harvest_success':bool(report.get('success')),
            'episode_completed':report.get('episode_completed',False),'model_calls':report.get('model_calls',0),
            'stop_reason':report.get('stop_reason'),'error':report.get('error'),
            'integrity_passed':quality.get('integrity_passed',False),
            'eligible_for_export_review':quality.get('eligible_for_training_export_review',False),
            'searches':searches,'executed_actions':dict(actions),'rejections':dict(rejected),
            'last_rejection':last_reason,'parking_trial':report.get('parking_trial'),
            'parking_trial_applied':report.get('parking_trial',{}).get('status')=='applied' if report.get('parking_trial') else False,
            'target_aware_parking':report.get('target_aware_parking',False),'fixed_base':report.get('fixed_base',False),
            'preclose_visual_check':report.get('preclose_visual_check',False),
            'preclose_recoveries':report.get('preclose_recoveries',0),
            'preclose_recovery_failed':report.get('preclose_recovery_failed',False),
            'local_first_selection':report.get('local_selection') is not None,
            'target_parking_moves':len(report.get('parking_history',[])), 'path':str(dest)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--fixed-base',action='store_true')
    mode.add_argument('--allow-row-search',action='store_true')
    p.add_argument('--target-aware-parking',action='store_true')
    p.add_argument('--pregrasp-park-plus40',action='store_true')
    p.add_argument('--preclose-visual-check',action='store_true')
    a=p.parse_args()
    if a.target_aware_parking and a.pregrasp_park_plus40:p.error('Choose target-aware or fixed parking, not both')
    if a.fixed_base and (a.target_aware_parking or a.pregrasp_park_plus40):p.error('--fixed-base excludes parking experiments')
    a.fixed_base=not (a.allow_row_search or a.target_aware_parking or a.pregrasp_park_plus40)
    seeds=a.seeds
    if len(seeds)>6 or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        p.error('Choose 1–6 distinct nonnegative seeds (32 calls each, at most 192 total)')
    out=a.out.resolve();frozen=hashes()
    plan={'purpose':'diagnostic evaluation; not a held-out benchmark or automatic training export','seeds':seeds,
          'attempts_per_seed':1,'max_calls_per_episode':32,'total_call_limit':32*len(seeds),
          'pregrasp_park_plus40':a.pregrasp_park_plus40,'target_aware_parking':a.target_aware_parking,'fixed_base':a.fixed_base,'scene_mode':'full-greenhouse',
          'preclose_visual_check':a.preclose_visual_check,
          'local_first_selection':not a.pregrasp_park_plus40,
          'basket_mount':'chassis','policy_cameras':['harvest_close','wrist'],
          'viewer':True,'record_images':True,'source_sha256':frozen}
    if a.dry_run:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True,exist_ok=False);atomic_json(out/'plan.json',plan)
    (out/'source').mkdir()
    for name in frozen:shutil.copy2(project_source(name),out/'source'/name)
    results=[];stop=None
    for seed in seeds:
        if hashes()!=frozen:stop='Source changed; fixed-version batch stopped';break
        if shutil.disk_usage(out).free<2*1024**3:stop='Less than 2 GiB free; batch stopped before next episode';break
        dest=out/f'seed-{seed}'
        env=os.environ.copy();env.update(MUJOCO_GL='glfw',PYOPENGL_PLATFORM='glx',PYTHONPATH='',PYTHONNOUSERSITE='1')
        command=[sys.executable,'-u',str(project_source('vista.py')),'--viewer','--peduncle-test','--seed',str(seed),'--max-calls','32','--record-images','--out',str(dest)]
        if a.pregrasp_park_plus40:command.append('--pregrasp-park-plus40')
        if a.target_aware_parking:command.append('--target-aware-parking')
        if a.preclose_visual_check:command.append('--preclose-visual-check')
        if a.fixed_base:command.append('--fixed-base')
        if a.allow_row_search:command.append('--allow-row-search')
        print(f'START seed={seed} log={out / ("seed-"+str(seed)+".log")}',flush=True)
        start=time.monotonic()
        with (out/f'seed-{seed}.log').open('w') as log:
            code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        row=summarize(dest);row.update(seed=seed,exit_code=code,wall_seconds=time.monotonic()-start,source_unchanged=hashes()==frozen)
        results.append(row)
        if code or row['error'] or not row['integrity_passed'] or not row['source_unchanged']:
            stop='Runtime, interruption, integrity or source-version failure; no further episode started'
        summary={'planned_seeds':seeds,'completed_attempts':len(results),'harvest_successes':sum(r['harvest_success'] for r in results),
                 'parking_trial_applied_count':sum(r['parking_trial_applied'] for r in results),
                 'total_model_calls':sum(r['model_calls'] for r in results),'stop_reason':stop,'results':results}
        atomic_json(out/'summary.json',summary);print(json.dumps(row),flush=True)
        if stop:break
    if stop:print('STOP: '+stop,flush=True)
    atomic_json(out/'summary.json',{'planned_seeds':seeds,'completed_attempts':len(results),'harvest_successes':sum(r['harvest_success'] for r in results),
                                  'parking_trial_applied_count':sum(r['parking_trial_applied'] for r in results),
                                  'total_model_calls':sum(r['model_calls'] for r in results),'stop_reason':stop,'results':results})
    print('DONE '+str(out/'summary.json'),flush=True)

if __name__=='__main__':main()
