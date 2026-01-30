Python implementations of traditional image harmonization methods

# Installation

```bash
cd implementations
pip install -r requirements.txt
```

Don't forget to always clear Python cache before running tests to ensure you're using the latest code:

```bash
cd implementations
rm -rf __pycache__
find . -name "*.pyc" -delete
```

# Quick Testing of implementations

```bash
# Tests Gray Mean Scale
python3 quick_test_gms.py

# Tests Standard Poisson Image Editing
python3 quick_test_poisson.py

# Tests Efficient Poisson Image Editing. Faster than standard PIE
python3 quick_test_efficient_poisson.py
```

# Quick method details

# 1. Gray Mean Scale

Core idea is to uise "gray pixels" (achromatic colors) to estimate illumination scale factors. 
Gray pixels are special because they reflect light without adding their own color. Under the Dichromatic Reflection Model, their appearance depends only on lighting, not on material's inherent color.

# Algorithm:
```python
# Pseudocode
def gray_mean_scale(I_c, I_t, M):
    # 1. Identify gray pixels in source BACKGROUND, NOT foreground object
    M_src_bg = ~M  # Source background region
    gray_mask_s = select_top_gray_pixels(I_c, M_src_bg, percent=0.001)
    
    # 2. Identify gray pixels in target background. `gray_percent` is a top percentile for gray pixel selection Default is 0.001 = 0.1% as paper recommends 0.1% for robustness
    gray_mask_t = select_top_gray_pixels(I_t, None, percent=0.001)
    
    # 3. Compute mean gray values. Estimates of illumination estimates
    g_s = mean(I_c[gray_mask_s])  # Source illumination
    g_t = mean(I_t[gray_mask_t])  # Target illumination
    
    # 4. Compute scale factors k = g_t / g_s
    k = g_t / g_s  # Per-channel
    
    # 5. Scale foreground and composite
    I_h = I_t.copy()
    I_h[M] = clip(k * I_c[M], 0, 1)
    
    return I_h
```

# 2. Poisson Image Editing

Core idea is to to seamlessly blend by solving the Poisson equation. Instead of copying pixels directly, copy GRADIENTS from source and solve for pixel values that:
1. Match gradients of source INSIDE the region (Δf = Δg)
2. Match pixel values of target AT THE BOUNDARY (f|∂Ω = f*|∂Ω)

**Algorithm:**
```python
# Pseudocode - Standard Poisson
def poisson_image_editing(I_c, I_t, M):
    # For each RGB channel independently:
    for channel in [R, G, B]:
        # 1. Zero out mask edges to prevent artifacts
        M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = 0
        
        # 2. Find bounding box around mask
        y_min, y_max, x_min, x_max = bounding_box(M)
        
        # 3. Build Laplacian matrix (5-point stencil)
        A = laplacian_matrix(height, width)
        
        # 4. Compute guidance field (Laplacian of source)
        b = A · source_crop
        
        # 5. Set boundary conditions (non-masked pixels = target)
        for i in pixels_outside_mask:
            A[i, :] = 0
            A[i, i] = 1
            b[i] = target[i]
        
        # 6. Solve sparse linear system Ax = b
        result_crop = sparse_solve(A, b)
        
        # 7. Clip and copy back to full image
        result[y_min:y_max, x_min:x_max] = clip(result_crop, 0, 255)
    
    return result / 255.0
```

# 3. Efficient Poisson Image Editing (Hussain & Kamel, 2015)

Core idea is to accelerate Poisson solving using pyramid and block partitioning (PRSSB method).

It is faster because: - it solves one system of size N: O(N^1.5), - it also solve r systems of size N/r: O(N^1.5 / √r), - it speeds up factor: √r

**Algorithm:**
```python
# Pseudocode - Efficient Poisson (PRSSB)
def efficient_poisson_editing(I_c, I_t, M):
    # 1. Build Gaussian pyramids
    pyr_c = gaussian_pyramid(I_c, levels=2)
    pyr_t = gaussian_pyramid(I_t, levels=2)
    pyr_m = gaussian_pyramid(M, levels=2, threshold=0.6)  # Conservative
    
    # 2. Solve at coarsest level (standard Poisson on small image)
    I_h = poisson_image_editing(pyr_c[-1], pyr_t[-1], pyr_m[-1])
    
    # 3. Refine at each finer level
    for level in range(levels-2, -1, -1):
        # a. Upsample result to current level's size
        I_h = upsample(I_h, target_shape, interpolation=LANCZOS)
        
        # b. Partition mask into small square blocks (32×32)
        partitions = partition_into_blocks(pyr_m[level], block_size=32)
        
        # c. Solve each block independently
        #    (I_h from previous level provides boundary conditions)
        for partition_mask in partitions:
            I_h = poisson_image_editing(pyr_c[level], I_h, partition_mask)
    
    return I_h
```

# Common Notation

 `f` => Foreground region 
 `b` => Background region 
 `I_s` => Source image (foreground only) 
 `I_t` => Target image (background) 
 `I_c` => Composite image (source pasted on target) 
 `I_h` => Harmonized result 
 `Ω` => Harmonization region (mask area) 
 `M` => Binary mask (1 = foreground, 0 = background) 
 `∂Ω` => Boundary of Ω 
 `∇g` => Gradient of source 
 `Δf` => Laplacian of result 
 `k` => Illumination scale factor 
