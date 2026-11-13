"""Bounded ACT updates and optional seed-held-out checks. Offline only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType,PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from recording_quality import atomic_json


def smoke(root,out):
    if (root/'EXPORT_INCOMPLETE.json').exists():raise ValueError('Incomplete export')
    verification=json.loads((root/'export_verification.json').read_text())
    if not verification.get('passed'):raise ValueError('Export verification failed')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=json.loads((root/'export_manifest.json').read_text())
    ds=LeRobotDataset(manifest['repo_id'],root=root,delta_timestamps={'action':[j/20 for j in range(10)]})
    cameras=[key.removeprefix('observation.images.') for key in ds.features if key.startswith('observation.images.')]
    if not cameras:raise ValueError('Dataset has no visual inputs')
    alignment=json.loads((root/'source_alignment.json').read_text())
    labels=alignment['rows']
    split=manifest.get('split')
    if isinstance(split,dict):
        train_episodes=split['train'];validation_episodes=split['validation']
        assert not set(train_episodes)&set(validation_episodes)
        assert not set(split['train_seeds'])&set(split['validation_seeds'])
    else:train_episodes=[0];validation_episodes=[]
    training_indices=[i for i,r in enumerate(labels) if r.get('episode_index',0) in train_episodes]
    assert training_indices
    stats={}
    numeric=ds.hf_dataset.select_columns(['observation.state','action']).with_format('numpy')
    for k in ('observation.state','action'):
        data=np.array(numeric[k])[training_indices];stats[k]={'mean':data.mean(0).tolist(),'std':np.maximum(data.std(0),1e-3).tolist()}
    def batch(index):
        item=ds[index];b={}
        for k in ('observation.state','action'):
            b[k]=((item[k]-torch.tensor(stats[k]['mean']))/torch.tensor(stats[k]['std'])).unsqueeze(0).to(device)
        b['action_is_pad']=item['action_is_pad'].unsqueeze(0).to(device)
        for c in cameras:
            key=f'observation.images.{c}'
            img=F.interpolate(item[key].unsqueeze(0),size=(240,320),mode='bilinear',align_corners=False)
            b[key]=((img-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]).to(device)
        return b
    config=ACTConfig(input_features={'observation.state':PolicyFeature(FeatureType.STATE,(9,)),**{f'observation.images.{c}':PolicyFeature(FeatureType.VISUAL,(3,240,320)) for c in cameras}},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(8,))},chunk_size=10,n_action_steps=1,
        dim_model=128,n_heads=4,dim_feedforward=512,n_encoder_layers=2,n_decoder_layers=1,n_vae_encoder_layers=2,
        pretrained_backbone_weights=None,device=device,push_to_hub=False)
    policy=ACTPolicy(config).to(device);optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-5)
    config.save_pretrained(out/'policy_config');atomic_json(out/'normalization.json',stats)
    atomic_json(out/'normalization_provenance.json',{'train_episode_indices':train_episodes,
        'frames_used':len(training_indices),'held_out_frames_used':0})
    history=[];start=time.monotonic();before=next(policy.parameters()).detach().clone()
    def stage_indices(episodes):
        return [next(i for i,r in enumerate(labels) if r.get('episode_index',0)==episode and r['stage']==stage)
                for episode in episodes for stage in ('approach','detach','transport')]
    indices=stage_indices(train_episodes)
    for step,index in enumerate(indices,1):
        policy.train();optimizer.zero_grad();loss,metrics=policy(batch(index))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),10,error_if_nonfinite=True);optimizer.step()
        row={'step':step,'dataset_index':index,'episode_index':labels[index].get('episode_index',0),
             'stage':labels[index]['stage'],'loss':loss.item(),'grad_norm':norm.item(),**metrics};history.append(row);print(row,flush=True)
    changed=not torch.equal(before,next(policy.parameters()).detach())
    assert changed
    policy.eval();probe=batch(indices[-1])
    with torch.inference_mode():pred=policy.predict_action_chunk(probe).clone()
    assert pred.shape==(1,10,8) and torch.isfinite(pred).all()
    validation=[]
    with torch.inference_mode():
        for index in stage_indices(validation_episodes):
            held=batch(index);prediction=policy.predict_action_chunk(held)
            assert prediction.shape==(1,10,8) and torch.isfinite(prediction).all()
            mask=~held['action_is_pad'].unsqueeze(-1)
            l1=((prediction-held['action']).abs()*mask).sum()/(mask.sum()*8)
            assert torch.isfinite(l1)
            validation.append({'dataset_index':index,'episode_index':labels[index]['episode_index'],
                               'stage':labels[index]['stage'],'normalized_l1':l1.item()})
    torch.save({'model':policy.state_dict(),'optimizer':optimizer.state_dict(),'steps':len(indices),'diagnostic_only':True},out/'checkpoint.pt')
    reloaded=ACTPolicy(config).to(device);reloaded.load_state_dict(torch.load(out/'checkpoint.pt',map_location=device,weights_only=True)['model']);reloaded.eval()
    with torch.inference_mode():again=reloaded.predict_action_chunk(probe)
    torch.testing.assert_close(pred,again,atol=1e-6,rtol=1e-5)
    atomic_json(out/'smoke_report.json',{'passed':True,'steps':len(indices),'device':device,'history':history,'elapsed_seconds':time.monotonic()-start,
        'train_episode_indices':train_episodes,'validation_episode_indices':validation_episodes,
        'validation':validation,'normalization_from_training_only':True,
        'parameter_update_verified':changed,'checkpoint_reload_matches':True,'prediction_shape':list(pred.shape),
        'source_dataset':str(root.resolve()),'precision':'float32','backbone':'random initialization; no weights downloaded',
        'training_resize':[240,320],'export_resolution':[360,480],
        'limitations':'Reduced ACT configuration for pipeline validation only. Held-out samples check the loading/inference path, not generalization. No closed-loop rollout; not a trained picking policy.'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();smoke(a.dataset,a.out)
