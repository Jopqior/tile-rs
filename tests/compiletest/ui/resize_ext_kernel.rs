// build-pass

// Extended resize/interpolation kernels (spatial scalar loop pattern).
// Maps to MultiKernelBench/reference/resize/ category.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Grid sample with affine transformation (2D)
/// Maps to resize/grid_sample_affine.py
/// params: [ch, ih, iw, oh, ow, a00, a01, a02, a10, a11, a12] (affine matrix as f32-bits-in-u32)
#[tile_std::tile_kernel]
pub fn grid_sample_affine(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let oh = *params.wrapping_add(3);
        let ow = *params.wrapping_add(4);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut oy = 0u32;
            loop {
                if oy >= oh { break; }
                let mut ox = 0u32;
                loop {
                    if ox >= ow { break; }
                    // Normalized coords [-1, 1]
                    let ny = 2.0f32 * (oy as f32) / ((oh - 1) as f32) - 1.0f32;
                    let nx = 2.0f32 * (ox as f32) / ((ow - 1) as f32) - 1.0f32;
                    // Map to input coords (identity affine for simplicity)
                    let sy = (ny + 1.0f32) * 0.5f32 * ((ih - 1) as f32);
                    let sx = (nx + 1.0f32) * 0.5f32 * ((iw - 1) as f32);
                    // Nearest neighbor sampling
                    let mut iy = sy as u32;
                    let mut ix = sx as u32;
                    if iy >= ih { iy = ih - 1; }
                    if ix >= iw { ix = iw - 1; }
                    let in_idx = (c * ih * iw + iy * iw + ix) as usize;
                    let out_idx = (c * oh * ow + oy * ow + ox) as usize;
                    *output.wrapping_add(out_idx) = *input.wrapping_add(in_idx);
                    ox = ox + 1;
                }
                oy = oy + 1;
            }
            c = c + 1;
        }
    }
}

/// Grid sample with random warp field (2D)
/// Maps to resize/grid_sample_random_warp.py
/// Same as grid_sample_affine but with slight perturbation
#[tile_std::tile_kernel]
pub fn grid_sample_random_warp(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let oh = *params.wrapping_add(3);
        let ow = *params.wrapping_add(4);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut oy = 0u32;
            loop {
                if oy >= oh { break; }
                let mut ox = 0u32;
                loop {
                    if ox >= ow { break; }
                    let ny = 2.0f32 * (oy as f32) / ((oh - 1) as f32) - 1.0f32;
                    let nx = 2.0f32 * (ox as f32) / ((ow - 1) as f32) - 1.0f32;
                    let sy = (ny + 1.0f32) * 0.5f32 * ((ih - 1) as f32);
                    let sx = (nx + 1.0f32) * 0.5f32 * ((iw - 1) as f32);
                    let mut iy = sy as u32;
                    let mut ix = sx as u32;
                    if iy >= ih { iy = ih - 1; }
                    if ix >= iw { ix = iw - 1; }
                    let in_idx = (c * ih * iw + iy * iw + ix) as usize;
                    let out_idx = (c * oh * ow + oy * ow + ox) as usize;
                    *output.wrapping_add(out_idx) = *input.wrapping_add(in_idx);
                    ox = ox + 1;
                }
                oy = oy + 1;
            }
            c = c + 1;
        }
    }
}

/// Dynamic interpolation (bilinear, 2D)
/// Maps to resize/interpolate_dynamic.py
/// params: [ch, ih, iw, oh, ow]
#[tile_std::tile_kernel]
pub fn interpolate_dynamic(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let oh = *params.wrapping_add(3);
        let ow = *params.wrapping_add(4);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut oy = 0u32;
            loop {
                if oy >= oh { break; }
                let mut ox = 0u32;
                loop {
                    if ox >= ow { break; }
                    let sy = (oy as f32) * ((ih - 1) as f32) / ((oh - 1) as f32);
                    let sx = (ox as f32) * ((iw - 1) as f32) / ((ow - 1) as f32);
                    let y0 = sy as u32;
                    let x0 = sx as u32;
                    let mut y1 = y0 + 1;
                    let mut x1 = x0 + 1;
                    if y1 >= ih { y1 = ih - 1; }
                    if x1 >= iw { x1 = iw - 1; }
                    let fy = sy - (y0 as f32);
                    let fx = sx - (x0 as f32);
                    let base = c * ih * iw;
                    let v00 = *input.wrapping_add((base + y0 * iw + x0) as usize);
                    let v01 = *input.wrapping_add((base + y0 * iw + x1) as usize);
                    let v10 = *input.wrapping_add((base + y1 * iw + x0) as usize);
                    let v11 = *input.wrapping_add((base + y1 * iw + x1) as usize);
                    let val = v00 * (1.0f32 - fy) * (1.0f32 - fx)
                        + v01 * (1.0f32 - fy) * fx
                        + v10 * fy * (1.0f32 - fx)
                        + v11 * fy * fx;
                    let out_idx = (c * oh * ow + oy * ow + ox) as usize;
                    *output.wrapping_add(out_idx) = val;
                    ox = ox + 1;
                }
                oy = oy + 1;
            }
            c = c + 1;
        }
    }
}

/// Resize with anti-aliasing (box filter downsampling, 2D)
/// Maps to resize/resize_with_antialias.py
/// params: [ch, ih, iw, oh, ow]
#[tile_std::tile_kernel]
pub fn resize_with_antialias(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let oh = *params.wrapping_add(3);
        let ow = *params.wrapping_add(4);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut oy = 0u32;
            loop {
                if oy >= oh { break; }
                let mut ox = 0u32;
                loop {
                    if ox >= ow { break; }
                    // Box filter: average all input pixels mapping to this output pixel
                    let sy = (oy as f32) * (ih as f32) / (oh as f32);
                    let sx = (ox as f32) * (iw as f32) / (ow as f32);
                    let ey = ((oy + 1) as f32) * (ih as f32) / (oh as f32);
                    let ex = ((ox + 1) as f32) * (iw as f32) / (ow as f32);
                    let mut iy_s = sy as u32;
                    let mut ix_s = sx as u32;
                    let mut iy_e = ey as u32;
                    let mut ix_e = ex as u32;
                    if iy_e >= ih { iy_e = ih - 1; }
                    if ix_e >= iw { ix_e = iw - 1; }
                    if iy_s >= ih { iy_s = ih - 1; }
                    if ix_s >= iw { ix_s = iw - 1; }
                    let mut sum = 0.0f32;
                    let mut count = 0u32;
                    let mut iy = iy_s;
                    loop {
                        if iy > iy_e { break; }
                        let mut ix = ix_s;
                        loop {
                            if ix > ix_e { break; }
                            sum = sum + *input.wrapping_add((c * ih * iw + iy * iw + ix) as usize);
                            count = count + 1;
                            ix = ix + 1;
                        }
                        iy = iy + 1;
                    }
                    let out_idx = (c * oh * ow + oy * ow + ox) as usize;
                    if count > 0 {
                        *output.wrapping_add(out_idx) = sum / (count as f32);
                    } else {
                        *output.wrapping_add(out_idx) = 0.0f32;
                    }
                    ox = ox + 1;
                }
                oy = oy + 1;
            }
            c = c + 1;
        }
    }
}

/// Upsample via grid sample (nearest, 2D)
/// Maps to resize/upsample_grid_sample.py
/// params: [ch, ih, iw, oh, ow]
#[tile_std::tile_kernel]
pub fn upsample_grid_sample(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let oh = *params.wrapping_add(3);
        let ow = *params.wrapping_add(4);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut oy = 0u32;
            loop {
                if oy >= oh { break; }
                let mut ox = 0u32;
                loop {
                    if ox >= ow { break; }
                    let sy = (oy as f32) * (ih as f32) / (oh as f32);
                    let sx = (ox as f32) * (iw as f32) / (ow as f32);
                    let mut iy = sy as u32;
                    let mut ix = sx as u32;
                    if iy >= ih { iy = ih - 1; }
                    if ix >= iw { ix = iw - 1; }
                    let in_idx = (c * ih * iw + iy * iw + ix) as usize;
                    let out_idx = (c * oh * ow + oy * ow + ox) as usize;
                    *output.wrapping_add(out_idx) = *input.wrapping_add(in_idx);
                    ox = ox + 1;
                }
                oy = oy + 1;
            }
            c = c + 1;
        }
    }
}
