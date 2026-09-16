// build-pass

// Windowed pooling kernels (1D, 2D, 3D) with explicit sliding window.
// Maps to MultiKernelBench/reference/pooling/ category.
// All use scalar nested loops on GM pointers.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Max pooling 1D: output[i] = max(input[i*stride .. i*stride+k])
/// Maps to pooling/max_pool_1d.py
#[tile_std::tile_kernel]
pub fn max_pooling_1d(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_len = *params;
        let k_size = *params.wrapping_add(1);
        let stride = *params.wrapping_add(2);
        let out_len = (in_len - k_size) / stride + 1;

        let mut i = 0u32;
        loop {
            if i >= out_len { break; }
            let base = i * stride;
            let mut max_val = *input.wrapping_add(base as usize);
            let mut k = 1u32;
            loop {
                if k >= k_size { break; }
                let val = *input.wrapping_add((base + k) as usize);
                if val > max_val { max_val = val; }
                k = k + 1;
            }
            *output.wrapping_add(i as usize) = max_val;
            i = i + 1;
        }
    }
}

/// Max pooling 2D: sliding window max over HxW spatial dims
/// Maps to pooling/max_pool_2d.py
#[tile_std::tile_kernel]
pub fn max_pooling_2d(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let kw = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kw) / stride + 1;

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let base_h = ohi * stride;
                    let base_w = owi * stride;
                    let mut max_val = *input.wrapping_add((c * ih * iw + base_h * iw + base_w) as usize);
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kw { break; }
                            let val = *input.wrapping_add((c * ih * iw + (base_h + ki) * iw + base_w + kj) as usize);
                            if val > max_val { max_val = val; }
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *output.wrapping_add((c * oh * ow + ohi * ow + owi) as usize) = max_val;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }
    }
}

/// Max pooling 3D: sliding window max over DxHxW spatial dims
/// Maps to pooling/max_pool_3d.py
#[tile_std::tile_kernel]
pub fn max_pooling_3d(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let id = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kd = *params.wrapping_add(4);
        let kh = *params.wrapping_add(5);
        let kw = *params.wrapping_add(6);
        let stride = *params.wrapping_add(7);
        let od = (id - kd) / stride + 1;
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kw) / stride + 1;

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut odi = 0u32;
            loop {
                if odi >= od { break; }
                let mut ohi = 0u32;
                loop {
                    if ohi >= oh { break; }
                    let mut owi = 0u32;
                    loop {
                        if owi >= ow { break; }
                        let bd = odi * stride;
                        let bh = ohi * stride;
                        let bw = owi * stride;
                        let mut max_val = *input.wrapping_add((c * id * ih * iw + bd * ih * iw + bh * iw + bw) as usize);
                        let mut di = 0u32;
                        loop {
                            if di >= kd { break; }
                            let mut hi = 0u32;
                            loop {
                                if hi >= kh { break; }
                                let mut wi = 0u32;
                                loop {
                                    if wi >= kw { break; }
                                    let val = *input.wrapping_add((c * id * ih * iw + (bd + di) * ih * iw + (bh + hi) * iw + bw + wi) as usize);
                                    if val > max_val { max_val = val; }
                                    wi = wi + 1;
                                }
                                hi = hi + 1;
                            }
                            di = di + 1;
                        }
                        *output.wrapping_add((c * od * oh * ow + odi * oh * ow + ohi * ow + owi) as usize) = max_val;
                        owi = owi + 1;
                    }
                    ohi = ohi + 1;
                }
                odi = odi + 1;
            }
            c = c + 1;
        }
    }
}

/// Average pooling 1D: output[i] = mean(input[i*stride .. i*stride+k])
/// Maps to pooling/avg_pool_1d.py
#[tile_std::tile_kernel]
pub fn average_pooling_1d(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_len = *params;
        let k_size = *params.wrapping_add(1);
        let stride = *params.wrapping_add(2);
        let out_len = (in_len - k_size) / stride + 1;
        let inv_k = 1.0f32 / (k_size as f32);

        let mut i = 0u32;
        loop {
            if i >= out_len { break; }
            let base = i * stride;
            let mut sum = 0.0f32;
            let mut k = 0u32;
            loop {
                if k >= k_size { break; }
                sum = sum + *input.wrapping_add((base + k) as usize);
                k = k + 1;
            }
            *output.wrapping_add(i as usize) = sum * inv_k;
            i = i + 1;
        }
    }
}

/// Average pooling 2D: sliding window mean over HxW spatial dims
/// Maps to pooling/avg_pool_2d.py
#[tile_std::tile_kernel]
pub fn average_pooling_2d(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let kw = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kw) / stride + 1;
        let inv_k = 1.0f32 / ((kh * kw) as f32);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let base_h = ohi * stride;
                    let base_w = owi * stride;
                    let mut sum = 0.0f32;
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kw { break; }
                            sum = sum + *input.wrapping_add((c * ih * iw + (base_h + ki) * iw + base_w + kj) as usize);
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *output.wrapping_add((c * oh * ow + ohi * ow + owi) as usize) = sum * inv_k;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }
    }
}

/// Average pooling 3D: sliding window mean over DxHxW spatial dims
/// Maps to pooling/avg_pool_3d.py
#[tile_std::tile_kernel]
pub fn average_pooling_3d(
    input: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let id = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kd = *params.wrapping_add(4);
        let kh = *params.wrapping_add(5);
        let kw = *params.wrapping_add(6);
        let stride = *params.wrapping_add(7);
        let od = (id - kd) / stride + 1;
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kw) / stride + 1;
        let inv_k = 1.0f32 / ((kd * kh * kw) as f32);

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut odi = 0u32;
            loop {
                if odi >= od { break; }
                let mut ohi = 0u32;
                loop {
                    if ohi >= oh { break; }
                    let mut owi = 0u32;
                    loop {
                        if owi >= ow { break; }
                        let bd = odi * stride;
                        let bh = ohi * stride;
                        let bw = owi * stride;
                        let mut sum = 0.0f32;
                        let mut di = 0u32;
                        loop {
                            if di >= kd { break; }
                            let mut hi = 0u32;
                            loop {
                                if hi >= kh { break; }
                                let mut wi = 0u32;
                                loop {
                                    if wi >= kw { break; }
                                    sum = sum + *input.wrapping_add((c * id * ih * iw + (bd + di) * ih * iw + (bh + hi) * iw + bw + wi) as usize);
                                    wi = wi + 1;
                                }
                                hi = hi + 1;
                            }
                            di = di + 1;
                        }
                        *output.wrapping_add((c * od * oh * ow + odi * oh * ow + ohi * ow + owi) as usize) = sum * inv_k;
                        owi = owi + 1;
                    }
                    ohi = ohi + 1;
                }
                odi = odi + 1;
            }
            c = c + 1;
        }
    }
}
