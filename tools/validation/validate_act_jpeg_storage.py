"""Compare PNG staging with original JPEG embedding; no harvest claims or model API.

Use run-act-offline.sh with --source RAW [--source RAW ...] --out REPORT.json.
Temporary datasets and caches are removed, and raw sources are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from export_act_dataset import JOINTS, TASK, jpeg_episode_data, verify_embedded_jpegs
from recording_quality import atomic_json


def benchmark(sources, out):
    import datasets
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(4)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    cameras = ['harvest_close', 'wrist']
    records = [{'source': Path(p).resolve(), 'rows': [json.loads(line) for line in
                (Path(p) / 'trajectory.jsonl').read_text().splitlines()]} for p in sources]
    originals = {str(record['source'] / row['images'][c]):
                 hashlib.sha256((record['source'] / row['images'][c]).read_bytes()).hexdigest()
                 for record in records for row in record['rows'] for c in cameras}
    features = {'observation.state': {'dtype':'float32','shape':(9,), 'names':JOINTS+['base_x_m','base_y_m','base_yaw_rad']},
                'action': {'dtype':'float32','shape':(8,), 'names':JOINTS+['base_linear_m_s','base_angular_rad_s']},
                **{f'observation.images.{c}': {'dtype':'image','shape':(360,480,3),'names':['height','width','channels']} for c in cameras}}
    report = {'purpose':'Storage/loader comparison only, not successful harvest data',
              'episodes':len(records), 'frames':sum(len(r['rows']) for r in records),
              'cameras':cameras, 'image_size':[480,360], 'fps':20}
    old_cache = datasets.config.HF_DATASETS_CACHE
    try:
        with tempfile.TemporaryDirectory(prefix='act-jpeg-comparison-') as temp:
            work = Path(temp); datasets.config.HF_DATASETS_CACHE = work / 'cache'
            for mode in ['png_staging', 'original_jpeg']:
                dest = work / mode
                ds = LeRobotDataset.create('local/jpeg_storage_probe',20,features,root=dest,
                                            robot_type='so101_mobile_sim',use_videos=False)
                for index, record in enumerate(records):
                    if mode == 'original_jpeg':
                        ds.save_episode(episode_data=jpeg_episode_data(record['source'],record['rows'],cameras,index))
                    else:
                        for row in record['rows']:
                            frame={'observation.state':np.asarray(row['state']+row['base_pose'],np.float32),
                                   'action':np.asarray(row['action']+row['base_command'],np.float32),'task':TASK}
                            for c in cameras:
                                with Image.open(record['source']/row['images'][c]) as im:
                                    frame[f'observation.images.{c}']=np.asarray(im.convert('RGB')).copy()
                            ds.add_frame(frame)
                        ds.save_episode()
                ds.finalize()
                report[mode]={'dataset_bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),
                              'parquet_data_bytes':sum(p.stat().st_size for p in (dest/'data').rglob('*.parquet'))}
                if mode == 'original_jpeg':
                    report['jpeg_check']=verify_embedded_jpegs(dest,records,cameras)
                loaded = LeRobotDataset('local/jpeg_storage_probe',root=dest)
                offset = 0
                for record in records:
                    for i,row in enumerate(record['rows']):
                        sample=loaded[offset+i]
                        np.testing.assert_array_equal(sample['action'].numpy(),np.asarray(row['action']+row['base_command'],np.float32))
                        for c in cameras:
                            with Image.open(record['source']/row['images'][c]) as im:
                                expected=np.asarray(im.convert('RGB'))
                            actual=np.rint(sample[f'observation.images.{c}'].permute(1,2,0).numpy()*255).astype(np.uint8)
                            np.testing.assert_array_equal(actual,expected)
                    offset+=len(record['rows'])
                report[mode]['all_image_pixels_and_actions_matched']=True
            report['jpeg_to_png_size_ratio']=report['original_jpeg']['dataset_bytes']/report['png_staging']['dataset_bytes']
            report['space_reduction_percent']=100*(1-report['jpeg_to_png_size_ratio'])
    finally:
        datasets.config.HF_DATASETS_CACHE=old_cache
    for name,h in originals.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    report['source_images_unchanged']=True
    report['temporary_datasets_and_cache_removed']=True
    out.parent.mkdir(parents=True,exist_ok=True);atomic_json(out,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();benchmark(args.source,args.out)
