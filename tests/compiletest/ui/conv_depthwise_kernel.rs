// build-pass

// Depthwise and pointwise convolution kernels.
// Maps to MultiKernelBench/reference/conv/ depthwise category.
// Depthwise: groups == in_channels == out_channels (each channel convolved independently).
// Pointwise: 1x1 convolution (kh=kw=1).

#![feature(no_core)]

#![no_std]
#![no_core]

/// Depthwise 2D convolution with square input and square kernel
/// Maps to conv/conv_depthwise_2d_square_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_depthwise_2d_sq_sq(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params; // in_ch == out_ch == groups
        let h = *params.wrapping_add(1); // square: h == w
        let kh = *params.wrapping_add(2); // square: kh == kw
        let stride = *params.wrapping_add(3);
        let oh = (h - kh) / stride + 1;

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= oh { break; }
                    let mut sum = 0.0f32;
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kh { break; }
                            let r = ohi * stride + ki;
                            let col = owi * stride + kj;
                            let in_idx = (c * h * h + r * h + col) as usize;
                            let w_idx = (c * kh * kh + ki * kh + kj) as usize;
                            sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *output.wrapping_add((c * oh * oh + ohi * oh + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }
    }
}

/// Depthwise 2D convolution with asymmetric input and square kernel
/// Maps to conv/conv_depthwise_2d_asymmetric_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_depthwise_2d_asym_sq(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let ih = *params.wrapping_add(1);
        let iw = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kh) / stride + 1;

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let mut sum = 0.0f32;
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kh { break; }
                            let r = ohi * stride + ki;
                            let col = owi * stride + kj;
                            let in_idx = (c * ih * iw + r * iw + col) as usize;
                            let w_idx = (c * kh * kh + ki * kh + kj) as usize;
                            sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *output.wrapping_add((c * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }
    }
}

/// Depthwise 2D convolution with square input and asymmetric kernel
/// Maps to conv/conv_depthwise_2d_square_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_depthwise_2d_sq_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let ch = *params;
        let h = *params.wrapping_add(1);
        let kh = *params.wrapping_add(2);
        let kw = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let oh = (h - kh) / stride + 1;
        let ow = (h - kw) / stride + 1;

        let mut c = 0u32;
        loop {
            if c >= ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let mut sum = 0.0f32;
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kw { break; }
                            let r = ohi * stride + ki;
                            let col = owi * stride + kj;
                            let in_idx = (c * h * h + r * h + col) as usize;
                            let w_idx = (c * kh * kw + ki * kw + kj) as usize;
                            sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *output.wrapping_add((c * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }
    }
}

/// Depthwise 2D convolution with asymmetric input and asymmetric kernel
/// Maps to conv/conv_depthwise_2d_asymmetric_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_depthwise_2d_asym_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
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
                    let mut sum = 0.0f32;
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kw { break; }
                            let r = ohi * stride + ki;
                            let col = owi * stride + kj;
                            let in_idx = (c * ih * iw + r * iw + col) as usize;
                            let w_idx = (c * kh * kw + ki * kw + kj) as usize;
                            sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *output.wrapping_add((c * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }
    }
}

/// Depthwise separable 2D convolution: depthwise conv + pointwise conv
/// Maps to conv/conv_depthwise_separable_2d.py
#[tile_std::tile_kernel]
pub fn conv_depthwise_separable_2d(
    input: *const f32, dw_weight: *const f32, pw_weight: *const f32,
    output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let h = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let oh = (h - kh) / stride + 1;

        // Step 1: Depthwise — intermediate[c][ohi][owi]
        // We write intermediate results to output first, then overwrite with pointwise.
        // Use output buffer as intermediate storage (large enough: out_ch * oh * oh >= in_ch * oh * oh when out_ch >= in_ch).
        let inter = output; // reuse output as intermediate
        let mut c = 0u32;
        loop {
            if c >= in_ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= oh { break; }
                    let mut sum = 0.0f32;
                    let mut ki = 0u32;
                    loop {
                        if ki >= kh { break; }
                        let mut kj = 0u32;
                        loop {
                            if kj >= kh { break; }
                            let r = ohi * stride + ki;
                            let col = owi * stride + kj;
                            let in_idx = (c * h * h + r * h + col) as usize;
                            let w_idx = (c * kh * kh + ki * kh + kj) as usize;
                            sum = sum + *input.wrapping_add(in_idx) * *dw_weight.wrapping_add(w_idx);
                            kj = kj + 1;
                        }
                        ki = ki + 1;
                    }
                    *inter.wrapping_add((c * oh * oh + ohi * oh + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            c = c + 1;
        }

        // Step 2: Pointwise (1x1 conv across channels)
        // Read from intermediate, pointwise weight: out_ch x in_ch
        // Write final output offset by in_ch*oh*oh to avoid clobbering intermediate
        let final_off = (in_ch * oh * oh) as usize;
        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= oh { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let inter_idx = (ic * oh * oh + ohi * oh + owi) as usize;
                        let pw_idx = (oc * in_ch + ic) as usize;
                        sum = sum + *inter.wrapping_add(inter_idx) * *pw_weight.wrapping_add(pw_idx);
                        ic = ic + 1;
                    }
                    *output.wrapping_add(final_off + (oc * oh * oh + ohi * oh + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// Pointwise 2D convolution (1x1 kernel): output[oc][h][w] = sum_{ic} input[ic][h][w] * weight[oc][ic]
/// Maps to conv/conv_pointwise_2d.py
#[tile_std::tile_kernel]
pub fn conv_pointwise_2d(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let h = *params.wrapping_add(2);
        let w = *params.wrapping_add(3);

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= h { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= w { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let in_idx = (ic * h * w + hi * w + wi) as usize;
                        let w_idx = (oc * in_ch + ic) as usize;
                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                        ic = ic + 1;
                    }
                    *output.wrapping_add((oc * h * w + hi * w + wi) as usize) = sum;
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            oc = oc + 1;
        }
    }
}
