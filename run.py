"""Portable release interface to the original WL-CDiff entry points."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train','infer','evaluate'])
    p.add_argument('--wells',type=int,choices=[4,9,18,36],default=36)
    p.add_argument('--noise',choices=['clean','0.03'],default='clean')
    p.add_argument('--tag',default=None,help='Simple output namespace; default sNN')
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--device',default=None)
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--num-workers',type=int,default=None)
    p.add_argument('--max-items',type=int,default=None)
    p.add_argument('--ddim-steps',type=int,default=None)
    p.add_argument('--num-samples',type=int,default=None)
    p.add_argument('--seed',type=int,default=42,help='Run RNG seed, also set for inference')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    os.chdir(ROOT)
    tag=a.tag or f's{a.wells:02d}'
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in tag):
        p.error('--tag must contain only letters, digits, underscores or hyphens')
    if a.noise!='clean' and (a.wells!=36 or a.action not in ['infer','evaluate']):
        p.error('--noise 0.03 is a 36-well inference/evaluation option')
    ob='obData' if a.wells==36 else f'obData_s{a.wells:02d}'
    test_ob=ob if a.noise=='clean' else 'obData_s36_noise_s003'
    out=Path('outputs')/f'public_{tag}'
    train_out=out/'train'
    infer_out=out/'infer'/('clean' if a.noise=='clean' else 'noise_003')
    ckpt=Path(a.checkpoint) if a.checkpoint else train_out/'checkpoints/best.pt'
    settings=dict(action=a.action,wells=a.wells,config=a.config,
                  train_obdata_dir=f'dataset/v1/train/{ob}',test_obdata_dir=f'dataset/v1/test/{test_ob}',
                  train_output_dir=str(train_out),infer_output_dir=str(infer_out),checkpoint=str(ckpt),seed=a.seed)
    for path in [Path(a.config),Path(settings['train_obdata_dir']),Path(settings['test_obdata_dir']),Path('dataset/v1/train/fields'),Path('dataset/v1/test/fields')]:
        if not path.exists():raise FileNotFoundError(path)
    print(json.dumps(settings,indent=2))
    if a.dry_run:
        print('Input paths OK. Checkpoint exists:',ckpt.exists(),'(weights are not included in this release).')
        return
    if a.action=='prepare':
        from src.data.build_sensor_dataset import build_train_val_cache
        print(build_train_val_cache(settings['train_obdata_dir'],'dataset/v1/train/fields',train_out/'cache/processed',field_kind='lnk',seed=42))
    elif a.action=='train':
        import src.train.train as tr
        tr.USER_CFG.update(train_obdata_dir=settings['train_obdata_dir'],train_field_truth_dir='dataset/v1/train/fields',
            train_output_dir=str(train_out),dataset_cache_dir=None,config_path=a.config,experiment_name=f'public_{tag}',
            field_kind='lnk',device=a.device,epochs=a.epochs,batch_size=a.batch_size,num_workers=a.num_workers,seed_override=a.seed)
        tr.main()
    elif a.action=='infer':
        if not ckpt.exists():raise FileNotFoundError(f'Train a model first: {ckpt}. Historical weights are intentionally not included.')
        import torch
        import numpy as np
        np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        import src.infer.infer_and_viz as inf
        inf.USER_CFG.update(checkpoint_path=str(ckpt),test_obdata_dir=settings['test_obdata_dir'],
            test_field_truth_dir='dataset/v1/test/fields',infer_output_dir=str(infer_out),config_path=a.config,
            field_kind='lnk',device=a.device,max_items=a.max_items,ddim_steps=a.ddim_steps,num_samples=a.num_samples)
        inf.main()
    else:
        import src.infer.evaluate_infer_results as ev
        ev.USER_CFG.update(infer_arrays_dir=str(infer_out/'arrays'),truth_field_dirs=['dataset/v1/test/fields'],
                           field_kind='lnk',out_dir=str(infer_out/'evaluation'),max_samples=a.max_items)
        ev.main()

if __name__=='__main__':main()
