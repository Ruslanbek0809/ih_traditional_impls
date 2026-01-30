#!/usr/bin/env python3
# Quick test for Efficient Poisson Image Editing

import numpy as np
import cv2
from pathlib import Path
import sys
import time

# Forces fresh import to avoid cache issues
if 'poisson_editing' in sys.modules:
    del sys.modules['poisson_editing']

from poisson_editing import efficient_poisson_editing

DATASET_PATH = Path('test_images/GMSDataset')


# Loads image as RGB float32 in [0, 1]
def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# Loads mask as float32 in [0, 1]
def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return mask.astype(np.float32) / 255.0


# Computes PSNR in dB using MSE
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


# Loads test image
img_name = '000-001.png'
print("=" * 60)
print(f"Testing Efficient Poisson on: {img_name}")
print("=" * 60)

composite = load_rgb(DATASET_PATH / 'com' / img_name)
background = load_rgb(DATASET_PATH / 'bg' / img_name)
mask = load_mask(DATASET_PATH / 'mask' / img_name)
gt = load_rgb(DATASET_PATH / 'gt' / img_name)

# Runs Efficient Poisson Image Editing
print("Running Efficient Poisson Image Editing...")
print("(uses 2-level pyramid + block partitioning)")
start = time.time()
result = efficient_poisson_editing(composite, background, mask, levels=2)
elapsed = time.time() - start

# Computes metrics
psnr = compute_psnr(result, gt)
print(f"PSNR: {psnr:.2f} dB")
print(f"Time: {elapsed:.2f} seconds")
print(f"Range: [{result.min():.3f}, {result.max():.3f}]")

# Saves result
out = (np.clip(result, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
cv2.imwrite('test_efficient_result.png', out)
print(f"Saved: test_efficient_result.png")
