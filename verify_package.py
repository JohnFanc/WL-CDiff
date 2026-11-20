"""Verify release file hashes and the supplied datasets without training a model."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(check_hashes=True,deep=False):
    result={'hashes_checked':0,'processed':{},'raw_observation_files_checked':0,'raw_heads_checked':0}
    manifest=ROOT/'SHA256SUMS.csv'
    if check_hashes:
        if not manifest.exists():raise FileNotFoundError('SHA256SUMS.csv is missing')
        with manifest.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                p=ROOT/row['path']
                assert p.is_file() and p.stat().st_size==int(row['bytes']) and digest(p)==row['sha256'],row['path']
                result['hashes_checked']+=1
    for pair,sensors in [('v1',36),('v1_s18',18)]:
        ids={}
        for split,n in [('train',1350),('val',150),('test',100)]:
            p=ROOT/f'dataset/processed/{pair}/{split}.h5'
            with h5py.File(p) as f:
                assert f['lnk'].shape==(n,50,50) and f['obs_feat'].shape==(n,sensors,51,3)
                for key in ['lnk','obs_feat','sensor_xy','times']:assert np.isfinite(f[key][:]).all(),(p,key)
                ids[split]=f['sample_id'][:].tolist()
                assert np.all(np.diff(f['times'][:])>0)
        assert len(set(sum(ids.values(),[])))==1600
        assert set(ids['train']+ids['val'])==set(range(1,1501)) and set(ids['test'])==set(range(1501,1601))
        result['processed'][pair]={k:len(v) for k,v in ids.items()}
        if pair=='v1':baseline_ids=ids
        else:assert ids==baseline_ids
    with (ROOT/'dataset/split_indices.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):assert baseline_ids[row['split']][int(row['index_in_baseline_h5'])]==int(row['sample_id'])
    rng=np.random.default_rng(42);idx=np.arange(1500);rng.shuffle(idx)
    assert baseline_ids['val']==(idx[:150]+1).tolist() and baseline_ids['train']==(idx[150:]+1).tolist()
    if deep:
        for split,n,start in [('train',1500,1),('test',100,1501)]:
            base=ROOT/f'dataset/v1/{split}'
            for obdir in sorted(base.glob('obData*')):
                files=sorted(obdir.glob('Head*.npz'));assert len(files)==n
                meta=json.loads((obdir/'meta.json').read_text(encoding='utf-8'));s=int(meta['num_sensors']);seen=[]
                for p in files:
                    with np.load(p,allow_pickle=False) as f:
                        sid=int(f['sample_id']);seen.append(sid)
                        assert f['obs_feat'].shape==(s,51,3) and f['obs_abs'].shape==(s,51)
                        assert np.isfinite(f['obs_feat']).all() and np.isfinite(f['obs_abs']).all()
                        assert np.array_equal(f['obs_feat'][:,:,0],f['obs_abs'])
                        assert np.allclose(f['obs_feat'][:,:,1],f['obs_abs']-f['obs_abs'][:,:1])
                        assert np.allclose(f['obs_feat'][:,:,2],np.gradient(f['obs_abs'],f['times'].astype(np.float32),axis=1))
                        assert (f['sensor_xy']>=0).all() and (f['sensor_xy']<50).all()
                    result['raw_observation_files_checked']+=1
                assert set(seen)==set(range(start,start+n))
            for sid in range(start,start+n):
                lnk=np.loadtxt(base/f'fields/lnKfield_{sid:04d}.csv',delimiter=',')
                k=np.loadtxt(base/f'fields/Kfield_{sid:04d}.csv',delimiter=',')
                assert lnk.shape==(50,50) and np.isfinite(lnk).all() and np.isfinite(k).all() and (k>0).all()
                assert np.allclose(np.exp(lnk),k,rtol=2e-5,atol=2e-5)
                with h5py.File(base/f'heads/Head{sid:04d}.h5') as f:
                    arr=f['Datasets/Head/Values'][:];assert arr.shape==(51,2500) and np.isfinite(arr).all()
                    head=arr.reshape(51,50,50)
                    with np.load(base/f'obData/Head{sid:04d}.npz') as ob:
                        xy=ob['sensor_xy'];assert np.array_equal(head[:,xy[:,1],xy[:,0]].T,ob['obs_abs'])
                result['raw_heads_checked']+=1
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deep',action='store_true',help='Also read all raw field/head/observation arrays')
    args=p.parse_args();print(json.dumps(verify(deep=args.deep),indent=2));print('PASS')
