import src.sample_generator as sample_generator

import numpy as np
from pathlib import Path
import sys
from time import time
import os 
import warnings
warnings.filterwarnings('error')

seconds=20
sample_rate=22050

n_train = 100
n_validation = 100

gen_samples_path = Path("generated_samples")

start = time()

pixel_modes = 0
acc_pixels  = 0
i = 0
while i < n_train:
    if i % 10 == 0 :
        print(i)
    try:
        sxx, mask, freq, _ = sample_generator.gen_sim_lofargram(seconds, sample_rate, return_axis=True)
        
        pixel_modes += mask[:, :, 0].sum()
        acc_pixels  += mask.shape[0]*mask.shape[1]
        np.savez_compressed(gen_samples_path / 'train' / f'{i:03d}.npz', sxx, mask)
        i+=1
    except RuntimeWarning:
        print("runtime warning")
print(time() - start)

ones_proportion = pixel_modes/acc_pixels
class_weights = [1 - ones_proportion, ones_proportion]
np.save(gen_samples_path / "class_weights.npy", class_weights)

start = time()
i = 0
while i < n_validation:
    if i % 10 == 0 :
        print(i)
    try:
        sxx, mask, freq, _ = sample_generator.gen_sim_lofargram(seconds, sample_rate, return_axis=True)
        np.savez_compressed(gen_samples_path / 'validation' / f'{i:05d}.npz', sxx, mask)
        i+=1
    except RuntimeWarning:
        print("runtime warning")
print(time() - start)