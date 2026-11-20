"""Check model/data compatibility on CPU; no weights or predictions are saved."""
from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import torch
import numpy as np
from src.data.dataset import SensorDataset
from src.models.diffusion import build_model_from_config,DiffusionScheduler
from src.utils.io import load_yaml
from src.utils.metrics import ssim,rmse,mae

def main():
    os.chdir(Path(__file__).resolve().parent)
    torch.set_num_threads(1);torch.manual_seed(42)
    ds=SensorDataset('dataset/processed/v1/train.h5','dataset/processed/v1/stats.json')
    item=ds[0];cfg=load_yaml('configs/default.yaml')
    model=build_model_from_config(cfg).cpu().eval()
    obs=item['obs_feat'].unsqueeze(0)
    xy=torch.from_numpy(ds.sensor_xy_norm).float()
    x=item['lnk'].unsqueeze(0);t=torch.tensor([10],dtype=torch.long)
    scheduler=DiffusionScheduler(cfg['diffusion']['timesteps'])
    noise=torch.randn_like(x);noisy=scheduler.q_sample(x,t,noise)
    for s in [4,9,18,36]:
        with torch.no_grad():pred=model(noisy,t,obs[:,:s],xy[:s])
        assert pred.shape==x.shape and torch.isfinite(pred).all()
        print('Forward shape/finite check: sensors=',s,tuple(pred.shape))
    model.train();pred=model(noisy,t,obs[:,:4],xy[:4]);loss=(pred-scheduler.v_target(x,noise,t)).square().mean();loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    ref=item['lnk_raw'].numpy()[0]
    assert rmse(ref,ref)==0 and mae(ref,ref)==0 and abs(ssim(ref,ref)-1)<1e-6
    print('PASS: untrained forward/backward and metric identity checks. No accuracy evaluation performed.')

if __name__=='__main__':main()
