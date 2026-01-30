#!/usr/bin/env python3
# Quick test for Gray Mean Scale (GMS) harmonization

import numpy as np
import cv2
from pathlib import Path
import sys
import time

# Forces fresh import to avoid cache issues
if 'gray_mean_scale' in sys.modules:
    del sys.modules['gray_mean_scale']

from gray_mean_scale import gray_mean_scale_harmonize, gray_mean_scale_harmonize_with_info

DATASET_PATH = Path('test_images/GMSDataset')


# Loads image as RGB float32 in [0, 1]
def load_rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# Loads mask as float32 in [0, 1]
def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
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
print(f"Testing Gray Mean Scale on: {img_name}")
print("=" * 60)

# Loads data
composite = load_rgb(DATASET_PATH / 'com' / img_name)
background = load_rgb(DATASET_PATH / 'bg' / img_name)
mask = load_mask(DATASET_PATH / 'mask' / img_name)
ground_truth = load_rgb(DATASET_PATH / 'gt' / img_name)

H, W = composite.shape[:2]
mask_pixels = int(np.sum(mask > 0.5))

print(f"Image size: {H} × {W}")
print(f"Mask pixels: {mask_pixels:,}")
print()

# Runs Gray Mean Scale harmonization

print("Running Gray Mean Scale harmonization...")
start_time = time.time()

result, info = gray_mean_scale_harmonize_with_info(
    I_c=composite,
    I_t=background,
    M=mask,
    gray_percent=0.001  # 0.1% as recommended in paper
)

elapsed = time.time() - start_time

# Computes metrics
psnr = compute_psnr(result, ground_truth)
mse = np.mean((result - ground_truth) ** 2)

# Prints results
print()
print("=" * 60)
print("Results")
print("=" * 60)
print(f"Time:  {elapsed:.3f} seconds")
print(f"PSNR:  {psnr:.2f} dB")
print(f"MSE:   {mse:.6f}")
print(f"Range: [{result.min():.3f}, {result.max():.3f}]")
print()
print("Scale factors (k = g_target / g_source):")
print(f"  k_R = {info['k'][0]:.4f}")
print(f"  k_G = {info['k'][1]:.4f}")
print(f"  k_B = {info['k'][2]:.4f}")
print()
print("Gray pixel statistics:")
print(f"  Source gray pixels: {info['num_gray_source']:,} "
      f"({100 * info['num_gray_source'] / (H * W):.3f}%)")
print(f"  Target gray pixels: {info['num_gray_target']:,} "
      f"({100 * info['num_gray_target'] / (H * W):.3f}%)")
print()
print("Mean illumination estimates:")
print(f"  Source (g_s): R={info['g_s'][0]:.4f}, G={info['g_s'][1]:.4f}, B={info['g_s'][2]:.4f}")
print(f"  Target (g_t): R={info['g_t'][0]:.4f}, G={info['g_t'][1]:.4f}, B={info['g_t'][2]:.4f}")
print("=" * 60)

# Saves result

out = (np.clip(result, 0, 1) * 255).astype(np.uint8)
out_bgr = out[:, :, ::-1]
cv2.imwrite('test_gms_result.png', out_bgr)

print()
print(f"Saved to: test_gms_result.png")
