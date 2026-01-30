import numpy as np
from typing import Tuple, Dict, Optional

EPS = 1e-6  # Small epsilon to avoid division by zero


# GRAYNESS LIKELIHOOD COMPUTATION

# Computes how "gray" each pixel is using chromatic difference metric. A pixel is gray when R ≈ G ≈ B. We measure grayness by computing how similar the RGB channels are relative to the total intensity.
# Formula (from paper and validated implementations): chroma_diff = |R - G| + |R - B| + |G - B| intensity = R + G + B + ε grayness = 1 - (chroma_diff / intensity)
# Higher values indicate more gray (achromatic) pixels.
# Args: rgb: RGB image, shape (H, W, 3), values in [0, 1]
# Returns: grayness: Array (H, W), higher = more gray
# Example: Pure gray (0.5, 0.5, 0.5): chroma_diff = 0, grayness = 1.0
# Pure red (1.0, 0, 0): chroma_diff = 2.0, intensity = 1.0, grayness = -1.0

def compute_grayness_likelihood(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Input must be RGB image with shape (H, W, 3)")
    
    # Extracts channels
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Computes chromatic difference (sum of channel differences)
    chroma_diff = np.abs(R - G) + np.abs(R - B) + np.abs(G - B)
    
    # Computes total intensity (add epsilon to avoid division by zero)
    intensity = R + G + B + EPS
    
    # Computes grayness likelihood: higher = more gray
    # Range approximately [0, 1] for typical images
    grayness = 1.0 - (chroma_diff / intensity)
    
    return grayness


# GRAY PIXEL SELECTION

# Selects the grayest pixels from a specified region. The paper (Song et al. 2020) recommends using top 0.1% (0.001) of grayest pixels for robustness. We also filter by intensity to exclude very dark or saturated pixels which are unreliable.
# Selection criteria:
# 1. Must be within region_mask (if provided)
# 2. Must have mean intensity in [min_intensity, max_intensity]
# 3. Must be in top gray_percent of pixels by grayness likelihood
# Args: image: RGB image (H, W, 3), values in [0, 1]
# region_mask: Binary mask (H, W), True/1 = include pixel
# gray_percent: Fraction of grayest pixels to select (default: 0.001 = 0.1%)
# min_intensity: Minimum mean brightness to consider (excludes very dark pixels)
# max_intensity: Maximum mean brightness to consider (excludes saturated pixels)
# Returns: gray_mask: Boolean array (H, W), True for selected gray pixels
def select_top_gray_pixels(
    image: np.ndarray,
    region_mask: Optional[np.ndarray] = None,
    gray_percent: float = 0.001,
    min_intensity: float = 0.05,
    max_intensity: float = 0.95
) -> np.ndarray:
    # Computes grayness likelihood for all pixels
    grayness_score = compute_grayness_likelihood(image)
    
    # Computes mean intensity per pixel
    mean_intensity = np.mean(image, axis=2)
    
    # Creates validity mask based on intensity thresholds
    intensity_valid = (mean_intensity >= min_intensity) & (mean_intensity <= max_intensity)
    
    # Applies region mask if provided
    if region_mask is not None:
        region_binary = region_mask.astype(bool) if region_mask.dtype != bool else region_mask
        valid_pixels = intensity_valid & region_binary
    else:
        valid_pixels = intensity_valid
    
    # Handles edge case: no valid pixels
    if not np.any(valid_pixels):
        print("Warning: No valid pixels found in region, using all intensity-valid pixels")
        return intensity_valid
    
    # Gets grayness scores within valid region
    valid_grayness = grayness_score[valid_pixels]
    
    # Finds threshold: top gray_percent of pixels (highest grayness scores)
    # We want the HIGHEST grayness values (most gray)
    k = max(1, int(valid_grayness.size * gray_percent))
    threshold = np.partition(valid_grayness, -k)[-k]
    
    # Selects pixels that are:
    # 1. Valid (in region and intensity bounds)
    # 2. Have grayness >= threshold (among top gray_percent)
    gray_mask = valid_pixels & (grayness_score >= threshold)
    
    return gray_mask


# GRAY MEAN COMPUTATION
# Computes the mean RGB values of selected gray pixels. This gives us g = [g_R, g_G, g_B], which under the Gray World assumption represents the illumination color and intensity.
# Args: image: RGB image (H, W, 3), values in [0, 1]
# gray_mask: Boolean mask (H, W), True for selected gray pixels
# Returns: g: Mean gray values array [g_R, g_G, g_B], shape (3,)
# Raises: Warning if no gray pixels found, using global image mean as fallback
def compute_gray_mean(
    image: np.ndarray,
    gray_mask: np.ndarray
) -> np.ndarray:
    if not np.any(gray_mask):
        print("NO GRAY PIXELS FOUND, using global image mean as fallback")
        return np.mean(image, axis=(0, 1))
    
    # Extracts gray pixels: shape (N, 3) where N is number of gray pixels
    gray_pixels = image[gray_mask]
    
    # Computes mean across all gray pixels for each channel
    g = np.mean(gray_pixels, axis=0)
    
    return g


# Computes the scale factors k = g_t / g_s for each channel. This gives us the ratio of target illumination to source illumination for each color channel.
# Args: g_s: Mean gray values from source background [g_R, g_G, g_B]
# g_t: Mean gray values from target background [g_R, g_G, g_B]
# min_scale: Minimum allowed scale (prevents extreme darkening)
# max_scale: Maximum allowed scale (prevents extreme brightening)
# Returns: k: Scale factors [k_R, k_G, k_B], shape (3,)
# Raises: Warning if no gray pixels found, using global image mean as fallback
def compute_scale_factors(
    g_s: np.ndarray,
    g_t: np.ndarray,
    min_scale: float = 0.1,
    max_scale: float = 10.0
) -> np.ndarray:
    # Computes scale factors k = g_t / g_s with epsilon to prevent division by zero
    k = g_t / (g_s + EPS)
    
    # Clips to reasonable range to prevent extreme artifacts
    k = np.clip(k, min_scale, max_scale)
    
    return k


# Main harmonization function:
# This function expects I_c to be the COMPOSITE image where the source foreground is already pasted onto a source background. The source background region (I_c[~M]) is used to estimate source lighting.
# Algorithm (Song et al. 2020):
# 1. Find gray pixels in source BACKGROUND (composite outside mask)
# 2. Find gray pixels in target background
# 3. Compute scale k = g_target / g_source for each channel
# 4. Apply scale to foreground: I_h[M] = k * I_c[M]
# Args: I_c: Composite image (H, W, 3), values in [0, 1]
# I_t: Target image (H, W, 3), values in [0, 1]
# M: Binary mask (H, W), values in [0, 1]
# gray_percent: Fraction of grayest pixels to use (default: 0.001 = 0.1%)
# Returns: I_h: Harmonized image (H, W, 3), values in [0, 1]
# Raises: ValueError if input shapes are invalid
# Example:
# >>> composite = load_image("source_composite.jpg")  # Has source background
# >>> target = load_image("target.jpg")
# >>> mask = load_mask("mask.png")
# >>> harmonized = gray_mean_scale_harmonize(composite, target, mask)
def gray_mean_scale_harmonize(
    I_c: np.ndarray,
    I_t: np.ndarray,
    M: np.ndarray,
    gray_percent: float = 0.001
) -> np.ndarray:
    # Input Validation
    if I_c.ndim != 3 or I_c.shape[2] != 3:
        raise ValueError("Composite image must have shape (H, W, 3)")
    if I_t.ndim != 3 or I_t.shape[2] != 3:
        raise ValueError("Target image must have shape (H, W, 3)")
    if M.ndim != 2:
        raise ValueError("Mask must have shape (H, W)")
    if I_c.shape[:2] != I_t.shape[:2]:
        raise ValueError("Composite and target must have same height and width")
    if I_c.shape[:2] != M.shape:
        raise ValueError("Images and mask must have same height and width")
    
    # Step 1: Creates binary masks
    # NOTE: Using np.logical_not() instead of ~ operator due to a numpy bug
    # where ~ can unexpectedly modify the original array in certain contexts   
    M_foreground = M > 0.5  # Foreground region (object)
    M_src_background = np.logical_not(M_foreground)  # Source background region
    
    # Step 2: Selects gray pixels in source BACKGROUND
    # Key is to use source background to estimate source illumination, not the foreground object itself!   
    gray_source = select_top_gray_pixels(
        I_c,
        region_mask=M_src_background,
        gray_percent=gray_percent
    )
    
    # Step 3: Selects gray pixels in target background
    gray_target = select_top_gray_pixels(
        I_t,
        region_mask=None,  # Use entire target image
        gray_percent=gray_percent
    )
    
    # Step 4: Computes mean gray values (illumination estimates)
    g_s = compute_gray_mean(I_c, gray_source)  # Source illumination
    g_t = compute_gray_mean(I_t, gray_target)  # Target illumination
    
    # Step 5: Computes per-channel scale factors
    k = compute_scale_factors(g_s, g_t)
    
    # Step 6: Applies harmonization
    # Start with target as base
    I_h = I_t.copy()
    
    # Scale the foreground region
    # k has shape (3,), need to broadcast for pixel-wise multiplication
    foreground_scaled = I_c[M_foreground] * k[np.newaxis, :]
    
    # Clip to valid range
    foreground_scaled = np.clip(foreground_scaled, 0.0, 1.0)
    
    # Paste scaled foreground onto target
    I_h[M_foreground] = foreground_scaled
    
    return I_h.astype(np.float32)


# All in one function with diagnostic info
def gray_mean_scale_harmonize_with_info(
    I_c: np.ndarray,
    I_t: np.ndarray,
    M: np.ndarray,
    gray_percent: float = 0.001
) -> Tuple[np.ndarray, Dict]:
    # Creates masks
    M_foreground = M > 0.5
    M_src_background = np.logical_not(M_foreground)
    
    # Selects gray pixels
    gray_source = select_top_gray_pixels(I_c, region_mask=M_src_background, gray_percent=gray_percent)
    gray_target = select_top_gray_pixels(I_t, region_mask=None, gray_percent=gray_percent)
    
    # Computes means and scales
    g_s = compute_gray_mean(I_c, gray_source)
    g_t = compute_gray_mean(I_t, gray_target)
    k = compute_scale_factors(g_s, g_t)
    
    # Harmonizes
    I_h = gray_mean_scale_harmonize(I_c, I_t, M, gray_percent)
    
    # Collects diagnostic info
    info = {
        'k': k,
        'g_s': g_s,
        'g_t': g_t,
        'num_gray_source': int(np.sum(gray_source)),
        'num_gray_target': int(np.sum(gray_target)),
        'gray_mask_source': gray_source,
        'gray_mask_target': gray_target
    }
    
    return I_h, info

