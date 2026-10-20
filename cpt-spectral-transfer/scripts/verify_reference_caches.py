"""Verify transferred .npy hashes and exact paired Old replay; never modify caches."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
old=None
count=0
for run in [2,3,4]:
    for cache in sorted((root/f'runs/real_task_eight_theme_cpt/run{run}/cache').glob('*')):
        if not cache.is_dir(): continue
        manifest=json.loads((cache/'manifest.json').read_text())
        for name,expected in manifest['file_sha256'].items():
            f=cache/name
            digest=hashlib.sha256()
            with f.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
            if digest.hexdigest()!=expected: raise RuntimeError(f'Hash mismatch: {f}')
        x=np.load(cache/'train_seed42_rho20_tokens.npy',mmap_mode='r')
        s=np.load(cache/'train_seed42_rho20_source_ids.npy')
        current=np.array(x[s==1])
        if old is None: old=current
        if not np.array_equal(current,old): raise RuntimeError(f'Unpaired Old replay: {cache}')
        if x.shape!=(19520,1025) or (s==0).sum()!=15616 or (s==1).sum()!=3904:
            raise RuntimeError(f'Unexpected reference geometry: {cache}')
        count+=1
        print(f'PASS {cache.name}')
if count!=16: raise RuntimeError(f'Expected 16 task caches, found {count}')
print('All 16 caches passed hashes, geometry, and exact Old replay checks.')
