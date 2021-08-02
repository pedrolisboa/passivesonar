import sample_generator

import numpy as np
from pathlib import Path
import sys
sys.path.append('../../')
from modified_lofar import lofar
from time import time

start = time()
for i in range(10000):
    if i % 1000 == 0 :
        print(i)
    sxx, mask, freq, _ = sample_generator.gen_sim_lofargram(seconds, sample_rate, return_axis=True)
    #np.save(Path('generated_samples') / f'{i:05d}_sample.npy', sxx)
    #np.save(Path('generated_samples') / f'{i:05d}_mask.npy', mask)
    np.savez_compressed(Path('generated_samples')/ 'train' / f'{i:05d}.npz', sxx, mask)
print(time() - start)    

start = time()
for i in range(5000):
    if i % 500 == 0 :
        print(i)
    sxx, mask, freq, _ = sample_generator.gen_sim_lofargram(seconds, sample_rate, return_axis=True)
    #np.save(Path('generated_samples') / f'{i:05d}_sample.npy', sxx)
    #np.save(Path('generated_samples') / f'{i:05d}_mask.npy', mask)
    np.savez_compressed(Path('generated_samples')/ 'validation' / f'{i:05d}.npz', sxx, mask)
print(time() - start)