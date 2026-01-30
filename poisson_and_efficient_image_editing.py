
# Poisson Image Editing - Seamless image compositing by solving the Poisson equation.

# Mechanism: This method copies gradients from the source and solves for pixel values that match the gradients of the source inside the region (Δf = Δg) and the pixel values of the target at the boundary (f|∂Ω = f*|∂Ω).
# The result is smooth transitions where foreground blends naturally with background while preserving internal texture and detail.

import numpy as np
import cv2
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, List, Literal


# Core Laplacian Matrix Construction

# Generates the discrete Laplacian matrix for a grid of size rows × cols.
# The matrix has: Diagonal: 4 (center pixel coefficient), Off-diagonals: -1 (neighbor coefficients)
def laplacian_matrix(rows: int, cols: int) -> sparse.lil_matrix:
    # Creates ablock for one row (handles left-right neighbors)
    mat_D = sparse.lil_matrix((cols, cols))
    mat_D.setdiag(-1, -1)   # Left neighbor
    mat_D.setdiag(4)        # Center pixel
    mat_D.setdiag(-1, 1)    # Right neighbor
    
    # Replicates the block for all rows
    mat_A = sparse.block_diag([mat_D] * rows).tolil()
    
    # Adds up-down neighbors across rows
    mat_A.setdiag(-1, cols)    # Down neighbor (next row)
    mat_A.setdiag(-1, -cols)   # Up neighbor (previous row)
    
    return mat_A


# STANDARD POISSON IMAGE EDITING

# Performs Poisson image editing on a single color channel of the source and target images.
# Algorithm (Pérez et al. 2003):
# 1. Zero out mask edges to prevent boundary artifacts
# 2. Find bounding box around mask for efficiency
# 3. Build Laplacian matrix for the bounding box
# 4. Compute guidance field (Laplacian of source, optionally mixed with target)
# 5. Set boundary conditions: non-masked pixels use target values
# 6. Solve the sparse linear system Ax = b
# Mixed gradients (Equation 12 from paper):
# At each pixel, use the gradient with larger magnitude: v(x) = ∇source if |∇source| > |∇target| else ∇target
def poisson_edit_channel(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    offset: Tuple[int, int] = (0, 0),
    mixing: bool = False,
    zero_edges: bool = True
) -> np.ndarray:
    H_tgt, W_tgt = target.shape
    
    # Applies offset to source if needed
    offset_y, offset_x = offset
    if offset_y != 0 or offset_x != 0:
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        source = cv2.warpAffine(source, M, (W_tgt, H_tgt))
    
    # Thresholds the mask to binary
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Zeros out edges to prevent boundary artifacts. Critical part to prevent boundary artifacts.
    if zero_edges and mask_binary.shape[0] > 2 and mask_binary.shape[1] > 2:
        mask_binary[0, :] = 0    # Top edge
        mask_binary[-1, :] = 0   # Bottom edge
        mask_binary[:, 0] = 0    # Left edge
        mask_binary[:, -1] = 0   # Right edge
    
    # Finds bounding box around mask
    y_indices, x_indices = np.where(mask_binary > 0)
    
    if len(y_indices) == 0:
        return target  # Empty mask
    
    # Gets bounding box with 1-pixel padding
    y_min = max(0, y_indices.min() - 1)
    y_max = min(H_tgt, y_indices.max() + 2)
    x_min = max(0, x_indices.min() - 1)
    x_max = min(W_tgt, x_indices.max() + 2)
    
    # Crops to bounding box
    height = y_max - y_min
    width = x_max - x_min
    
    source_crop = source[y_min:y_max, x_min:x_max]
    target_crop = target[y_min:y_max, x_min:x_max]
    mask_crop = mask_binary[y_min:y_max, x_min:x_max]
    
    # Builds Laplacian matrix
    N = height * width
    mat_A = laplacian_matrix(height, width)
    
    # Computes Laplacian of source (guidance field)
    laplacian_op = mat_A.tocsc()
    source_flat = source_crop.flatten().astype(np.float64)
    laplacian_source = laplacian_op.dot(source_flat)
    
    # Mixed gradients: use gradient with larger magnitude (Equation 12)
    if mixing:
        target_flat = target_crop.flatten().astype(np.float64)
        laplacian_target = laplacian_op.dot(target_flat)
        
        # Uses whichever gradient has larger absolute value
        use_target = np.abs(laplacian_target) > np.abs(laplacian_source)
        laplacian_source[use_target] = laplacian_target[use_target]
    
    # Modifies matrix for boundary conditions
    # For pixels OUTSIDE the mask: f = target (identity operation)
    mask_flat = mask_crop.flatten()
    target_flat = target_crop.flatten().astype(np.float64)
    
    for i in range(N):
        if mask_flat[i] == 0:
            mat_A[i, :] = 0
            mat_A[i, i] = 1
    
    # Builds right-hand side
    vec_b = laplacian_source.copy()
    vec_b[mask_flat == 0] = target_flat[mask_flat == 0]
    
    # Solves the linear system
    mat_A = mat_A.tocsc()
    solution = spsolve(mat_A, vec_b)
    
    # Reshapes and clips
    result_crop = solution.reshape((height, width))
    result_crop = np.clip(result_crop, 0, 255)
    
    # Copies back to full image
    result = target.copy()
    result[y_min:y_max, x_min:x_max] = result_crop
    
    return result



# Performs Standard Poisson Image Editing on the entire image.
def poisson_image_editing(
    I_c: np.ndarray,
    I_t: np.ndarray,
    M: np.ndarray,
    offset: Tuple[int, int] = (0, 0),
    mixing: bool = False,
    zero_edges: bool = True
) -> np.ndarray:
    if I_c.ndim != 3 or I_c.shape[2] != 3:
        raise ValueError("Composite image must have shape (H, W, 3)")
    if I_t.ndim != 3 or I_t.shape[2] != 3:
        raise ValueError("Target image must have shape (H, W, 3)")
    if M.ndim != 2:
        raise ValueError("Mask must have shape (H, W)")
    if I_c.shape != I_t.shape:
        raise ValueError("Composite and target must have same shape")
    if I_c.shape[:2] != M.shape:
        raise ValueError("Images and mask must have same height and width")
    
    # Convert to [0, 255] range for processing
    composite_255 = (I_c * 255).astype(np.float64)
    target_255 = (I_t * 255).astype(np.float64)
    
    # Process each channel independently
    result_255 = np.zeros_like(target_255)
    
    for c in range(3):
        result_255[:, :, c] = poisson_edit_channel(
            source=composite_255[:, :, c],
            target=target_255[:, :, c],
            mask=M,
            offset=offset,
            mixing=mixing,
            zero_edges=zero_edges
        )
    
    # Convert back to [0, 1] range
    result = result_255 / 255.0
    return result.astype(np.float32)


# EFFICIENT POISSON IMAGE EDITING

# Builds a Gaussian pyramid from the finest (original) to the coarsest level.
def build_gaussian_pyramid(
    image: np.ndarray, # Input image (H, W) or (H, W, C)
    levels: int, # Number of pyramid levels
    is_mask: bool = False # If True, apply conservative mask thresholding
) -> List[np.ndarray]:
    pyramid = [image]
    current = image
    
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        
        # For masks, uses conservative threshold to prevent expansion at coarse levels, which causes blur.
        if is_mask:
            current = (current > 0.6).astype(np.float32)
        
        pyramid.append(current)
    
    return pyramid


# Upsamples the image to the specified shape using Lanczos interpolation to provide better sharpness.
def upsample_to_shape(
    image: np.ndarray, # Input image (H, W) or (H, W, C)
    target_shape: Tuple[int, int] # Target shape (height, width)
) -> np.ndarray:
    return cv2.resize(
        image,
        (target_shape[1], target_shape[0]),  # cv2 uses (width, height)
        interpolation=cv2.INTER_LANCZOS4  # Better than cubic for sharpness
    )


# Partitions the mask into square blocks of the specified size.
# By solving multiple small linear systems instead of one large one, we reduce computational complexity:
# One system of size N: O(N^1.5)
# r systems of size N/r: O(N^1.5 / √r)
# This is the PRSSB method from Hussain & Kamel.
def partition_into_blocks(
    mask: np.ndarray, # Binary mask (H, W)
    block_size: int = 32 # Size of each square block
) -> List[np.ndarray]:
    H, W = mask.shape
    blocks = []
    
    for start_y in range(0, H, block_size):
        for start_x in range(0, W, block_size):
            end_y = min(start_y + block_size, H)
            end_x = min(start_x + block_size, W)
            
            # Create mask for just this block
            block_mask = np.zeros_like(mask)
            block_mask[start_y:end_y, start_x:end_x] = mask[start_y:end_y, start_x:end_x]
            
            # Only add if block contains masked pixels
            if np.any(block_mask > 0.5):
                blocks.append(block_mask)
    
    return blocks


# Performs Efficient Poisson Image Editing on the entire image.
# Uses coarse-to-fine pyramid approach with block partitioning (PRSSB):
# 1. Build Gaussian pyramids for composite, target, mask
# 2. Solve at coarsest level (small problem)
# 3. For each finer level:
#    a. Upsample result from previous level
#    b. Partition mask into small blocks
#    c. Solve each block using upsampled result as boundary
# Key insight: upsampled coarse solution provides good boundary conditions, so we only need to refine details locally at each level.
# To reduce blur:
# - Use only 2 pyramid levels instead of 3
# - Use conservative mask thresholding (>0.6) at coarse levels
# - Use Lanczos interpolation for upsampling
# - Use smaller partition size (16-32)
def efficient_poisson_editing(
    I_c: np.ndarray, # Composite image (H, W, 3), float32 in [0, 1]
    I_t: np.ndarray, # Target image (H, W, 3), float32 in [0, 1]
    M: np.ndarray, # Binary mask (H, W), float in [0, 1]
    levels: int = 2, # Number of pyramid levels (2 recommended for quality)
    partition_size: int = 32, # Block size for partitioning
    mixing: bool = False, # If True, use mixed gradients
    zero_edges: bool = True # If True, zero mask edges
) -> np.ndarray:
    if I_c.ndim != 3 or I_t.ndim != 3:
        raise ValueError("Images must have shape (H, W, 3)")
    if levels < 2:
        raise ValueError("Need at least 2 pyramid levels")
    
    # Builds Gaussian pyramids
    pyr_c = build_gaussian_pyramid(I_c, levels, is_mask=False)
    pyr_t = build_gaussian_pyramid(I_t, levels, is_mask=False)
    pyr_m = build_gaussian_pyramid(M.astype(np.float32), levels, is_mask=True)
    
    # Solves at coarsest level (standard Poisson)
    I_h = poisson_image_editing(
        pyr_c[-1],
        pyr_t[-1],
        pyr_m[-1],
        mixing=mixing,
        zero_edges=zero_edges
    )
    
    # Refines at each finer level
    for level in range(levels - 2, -1, -1):
        # Upsamples result to current level's size
        target_shape = pyr_c[level].shape[:2]
        I_h = upsample_to_shape(I_h, target_shape)
        
        # Gets data for current level
        composite_level = pyr_c[level]
        mask_level = pyr_m[level]
        
        # Partitions mask into blocks
        partitions = partition_into_blocks(mask_level, partition_size)
        
        # Solves each partition independently
        # I_h (upsampled) provides boundaries
        for partition_mask in partitions:
            I_h = poisson_image_editing(
                composite_level,
                I_h,  # Uses current result as target (boundary)
                partition_mask,
                mixing=mixing,
                zero_edges=zero_edges
            )
    
    return I_h.astype(np.float32)


# All in one function

# Performs Seamless Cloning using Poisson Image Editing.
def seamless_clone(
    I_c: np.ndarray, # Composite image (H, W, 3), float32 in [0, 1]
    I_t: np.ndarray, # Target image (H, W, 3), float32 in [0, 1]
    M: np.ndarray, # Binary mask (H, W), float in [0, 1]
    method: Literal["standard", "efficient"] = "standard", # Method to use
    mixing: bool = False, # If True, use mixed gradients
    offset: Tuple[int, int] = (0, 0), # (offset_y, offset_x) for placing source
    zero_edges: bool = True # If True, zero mask edges
) -> np.ndarray:
    if method == "efficient":
        return efficient_poisson_editing(
            I_c, I_t, M, mixing=mixing, zero_edges=zero_edges
        )
    else:  # standard
        return poisson_image_editing(
            I_c, I_t, M, offset=offset, mixing=mixing, zero_edges=zero_edges
        )
