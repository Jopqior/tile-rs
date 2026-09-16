// build-pass

// Transposed convolution kernels (1D, 2D, 3D).
// Maps to MultiKernelBench/reference/conv/ transposed category.
// Transposed conv uses scatter-add: for each input element, scatter-add to output.

#![feature(no_core)]

#![no_std]
#![no_core]

/// Transposed 1D convolution
/// Maps to conv/conv_transposed_1d.py
#[tile_std::tile_kernel]
pub fn conv_transposed_1d(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let in_len = *params.wrapping_add(2);
        let k_size = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let out_len = (in_len - 1) * stride + k_size;

        // Zero output
        let mut i = 0u32;
        loop {
            if i >= out_ch * out_len { break; }
            *output.wrapping_add(i as usize) = 0.0f32;
            i = i + 1;
        }

        // Scatter-add: for each input[ic][p], add weight[ic][oc][k] * input[ic][p] to output[oc][p*stride+k]
        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut p = 0u32;
            loop {
                if p >= in_len { break; }
                let in_val = *input.wrapping_add((ic * in_len + p) as usize);
                let mut oc = 0u32;
                loop {
                    if oc >= out_ch { break; }
                    let mut k = 0u32;
                    loop {
                        if k >= k_size { break; }
                        let out_pos = p * stride + k;
                        let w_idx = (ic * out_ch * k_size + oc * k_size + k) as usize;
                        let o_idx = (oc * out_len + out_pos) as usize;
                        let cur = *output.wrapping_add(o_idx);
                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                        k = k + 1;
                    }
                    oc = oc + 1;
                }
                p = p + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 1D convolution with dilation
/// Maps to conv/conv_transposed_1d_dilated.py
#[tile_std::tile_kernel]
pub fn conv_transposed_1d_dilated(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let in_len = *params.wrapping_add(2);
        let k_size = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let dilation = *params.wrapping_add(5);
        let eff_k = (k_size - 1) * dilation + 1;
        let out_len = (in_len - 1) * stride + eff_k;

        let mut i = 0u32;
        loop {
            if i >= out_ch * out_len { break; }
            *output.wrapping_add(i as usize) = 0.0f32;
            i = i + 1;
        }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut p = 0u32;
            loop {
                if p >= in_len { break; }
                let in_val = *input.wrapping_add((ic * in_len + p) as usize);
                let mut oc = 0u32;
                loop {
                    if oc >= out_ch { break; }
                    let mut k = 0u32;
                    loop {
                        if k >= k_size { break; }
                        let out_pos = p * stride + k * dilation;
                        let w_idx = (ic * out_ch * k_size + oc * k_size + k) as usize;
                        let o_idx = (oc * out_len + out_pos) as usize;
                        let cur = *output.wrapping_add(o_idx);
                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                        k = k + 1;
                    }
                    oc = oc + 1;
                }
                p = p + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 1D convolution with asymmetric input, padding, stride, dilation
/// Maps to conv/conv_transposed_1d_asymmetric_input_square_kernel_padded_strided_dilated.py
#[tile_std::tile_kernel]
pub fn conv_transposed_1d_asym_padded_strided_dilated(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let in_len = *params.wrapping_add(2);
        let k_size = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let padding = *params.wrapping_add(5);
        let dilation = *params.wrapping_add(6);
        let eff_k = (k_size - 1) * dilation + 1;
        let out_len = (in_len - 1) * stride + eff_k - 2 * padding;

        let mut i = 0u32;
        loop {
            if i >= out_ch * out_len { break; }
            *output.wrapping_add(i as usize) = 0.0f32;
            i = i + 1;
        }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut p = 0u32;
            loop {
                if p >= in_len { break; }
                let in_val = *input.wrapping_add((ic * in_len + p) as usize);
                let mut oc = 0u32;
                loop {
                    if oc >= out_ch { break; }
                    let mut k = 0u32;
                    loop {
                        if k >= k_size { break; }
                        let raw_pos = p * stride + k * dilation;
                        if raw_pos >= padding {
                            let out_pos = raw_pos - padding;
                            if out_pos < out_len {
                                let w_idx = (ic * out_ch * k_size + oc * k_size + k) as usize;
                                let o_idx = (oc * out_len + out_pos) as usize;
                                let cur = *output.wrapping_add(o_idx);
                                *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                            }
                        }
                        k = k + 1;
                    }
                    oc = oc + 1;
                }
                p = p + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with square input and square kernel
/// Maps to conv/conv_transposed_2d_square_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_sq_sq(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let h = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let oh = (h - 1) * stride + kh;

        let total = out_ch * oh * oh;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= h { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= h { break; }
                    let in_val = *input.wrapping_add((ic * h * h + hi * h + wi) as usize);
                    let mut oc = 0u32;
                    loop {
                        if oc >= out_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kh { break; }
                                let or = hi * stride + ki;
                                let oc2 = wi * stride + kj;
                                let w_idx = (ic * out_ch * kh * kh + oc * kh * kh + ki * kh + kj) as usize;
                                let o_idx = (oc * oh * oh + or * oh + oc2) as usize;
                                let cur = *output.wrapping_add(o_idx);
                                *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        oc = oc + 1;
                    }
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with square input and asymmetric kernel
/// Maps to conv/conv_transposed_2d_square_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_sq_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let h = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let kw = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let oh = (h - 1) * stride + kh;
        let ow = (h - 1) * stride + kw;

        let total = out_ch * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= h { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= h { break; }
                    let in_val = *input.wrapping_add((ic * h * h + hi * h + wi) as usize);
                    let mut oc = 0u32;
                    loop {
                        if oc >= out_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kw { break; }
                                let or = hi * stride + ki;
                                let ocol = wi * stride + kj;
                                let w_idx = (ic * out_ch * kh * kw + oc * kh * kw + ki * kw + kj) as usize;
                                let o_idx = (oc * oh * ow + or * ow + ocol) as usize;
                                let cur = *output.wrapping_add(o_idx);
                                *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        oc = oc + 1;
                    }
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with asymmetric input and square kernel
/// Maps to conv/conv_transposed_2d_asymmetric_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_asym_sq(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kh;

        let total = out_ch * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= ih { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= iw { break; }
                    let in_val = *input.wrapping_add((ic * ih * iw + hi * iw + wi) as usize);
                    let mut oc = 0u32;
                    loop {
                        if oc >= out_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kh { break; }
                                let or = hi * stride + ki;
                                let ocol = wi * stride + kj;
                                let w_idx = (ic * out_ch * kh * kh + oc * kh * kh + ki * kh + kj) as usize;
                                let o_idx = (oc * oh * ow + or * ow + ocol) as usize;
                                let cur = *output.wrapping_add(o_idx);
                                *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        oc = oc + 1;
                    }
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with asymmetric input and asymmetric kernel
/// Maps to conv/conv_transposed_2d_asymmetric_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_asym_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let kw = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;

        let total = out_ch * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= ih { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= iw { break; }
                    let in_val = *input.wrapping_add((ic * ih * iw + hi * iw + wi) as usize);
                    let mut oc = 0u32;
                    loop {
                        if oc >= out_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kw { break; }
                                let or = hi * stride + ki;
                                let ocol = wi * stride + kj;
                                let w_idx = (ic * out_ch * kh * kw + oc * kh * kw + ki * kw + kj) as usize;
                                let o_idx = (oc * oh * ow + or * ow + ocol) as usize;
                                let cur = *output.wrapping_add(o_idx);
                                *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        oc = oc + 1;
                    }
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with asymmetric input, asymmetric kernel, and padding
/// Maps to conv/conv_transposed_2d_asymmetric_input_asymmetric_kernel_padded.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_asym_asym_padded(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let kw = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let padding = *params.wrapping_add(7);
        let oh = (ih - 1) * stride + kh - 2 * padding;
        let ow = (iw - 1) * stride + kw - 2 * padding;

        let total = out_ch * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= ih { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= iw { break; }
                    let in_val = *input.wrapping_add((ic * ih * iw + hi * iw + wi) as usize);
                    let mut oc = 0u32;
                    loop {
                        if oc >= out_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kw { break; }
                                let raw_r = hi * stride + ki;
                                let raw_c = wi * stride + kj;
                                if raw_r >= padding && raw_c >= padding {
                                    let or = raw_r - padding;
                                    let ocol = raw_c - padding;
                                    if or < oh && ocol < ow {
                                        let w_idx = (ic * out_ch * kh * kw + oc * kh * kw + ki * kw + kj) as usize;
                                        let o_idx = (oc * oh * ow + or * ow + ocol) as usize;
                                        let cur = *output.wrapping_add(o_idx);
                                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                    }
                                }
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        oc = oc + 1;
                    }
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with dilation, padding, and stride
/// Maps to conv/conv_transposed_2d_asymmetric_input_square_kernel_dilated_padded_strided.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_dilated_padded_strided(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let padding = *params.wrapping_add(6);
        let dilation = *params.wrapping_add(7);
        let eff_kh = (kh - 1) * dilation + 1;
        let oh = (ih - 1) * stride + eff_kh - 2 * padding;
        let ow = (iw - 1) * stride + eff_kh - 2 * padding;

        let total = out_ch * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut hi = 0u32;
            loop {
                if hi >= ih { break; }
                let mut wi = 0u32;
                loop {
                    if wi >= iw { break; }
                    let in_val = *input.wrapping_add((ic * ih * iw + hi * iw + wi) as usize);
                    let mut oc = 0u32;
                    loop {
                        if oc >= out_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kh { break; }
                                let raw_r = hi * stride + ki * dilation;
                                let raw_c = wi * stride + kj * dilation;
                                if raw_r >= padding && raw_c >= padding {
                                    let or = raw_r - padding;
                                    let ocol = raw_c - padding;
                                    if or < oh && ocol < ow {
                                        let w_idx = (ic * out_ch * kh * kh + oc * kh * kh + ki * kh + kj) as usize;
                                        let o_idx = (oc * oh * ow + or * ow + ocol) as usize;
                                        let cur = *output.wrapping_add(o_idx);
                                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                    }
                                }
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        oc = oc + 1;
                    }
                    wi = wi + 1;
                }
                hi = hi + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 2D convolution with groups, stride, padding, dilation
/// Maps to conv/conv_transposed_2d_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated.py
#[tile_std::tile_kernel]
pub fn conv_transposed_2d_grouped(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let kw = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let padding = *params.wrapping_add(7);
        let groups = *params.wrapping_add(8);
        let oh = (ih - 1) * stride + kh - 2 * padding;
        let ow = (iw - 1) * stride + kw - 2 * padding;
        let ic_per_g = in_ch / groups;
        let oc_per_g = out_ch / groups;

        let total = out_ch * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut g = 0u32;
        loop {
            if g >= groups { break; }
            let mut ic = 0u32;
            loop {
                if ic >= ic_per_g { break; }
                let abs_ic = g * ic_per_g + ic;
                let mut hi = 0u32;
                loop {
                    if hi >= ih { break; }
                    let mut wi = 0u32;
                    loop {
                        if wi >= iw { break; }
                        let in_val = *input.wrapping_add((abs_ic * ih * iw + hi * iw + wi) as usize);
                        let mut oc = 0u32;
                        loop {
                            if oc >= oc_per_g { break; }
                            let abs_oc = g * oc_per_g + oc;
                            let mut ki = 0u32;
                            loop {
                                if ki >= kh { break; }
                                let mut kj = 0u32;
                                loop {
                                    if kj >= kw { break; }
                                    let raw_r = hi * stride + ki;
                                    let raw_c = wi * stride + kj;
                                    if raw_r >= padding && raw_c >= padding {
                                        let or = raw_r - padding;
                                        let ocol = raw_c - padding;
                                        if or < oh && ocol < ow {
                                            let w_idx = (abs_ic * oc_per_g * kh * kw + oc * kh * kw + ki * kw + kj) as usize;
                                            let o_idx = (abs_oc * oh * ow + or * ow + ocol) as usize;
                                            let cur = *output.wrapping_add(o_idx);
                                            *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                        }
                                    }
                                    kj = kj + 1;
                                }
                                ki = ki + 1;
                            }
                            oc = oc + 1;
                        }
                        wi = wi + 1;
                    }
                    hi = hi + 1;
                }
                ic = ic + 1;
            }
            g = g + 1;
        }
    }
}

/// Transposed 3D convolution with square input and square kernel
/// Maps to conv/conv_transposed_3d_square_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_sq_sq(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let s = *params.wrapping_add(2);
        let kk = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let os = (s - 1) * stride + kk;

        let total = out_ch * os * os * os;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut di = 0u32;
            loop {
                if di >= s { break; }
                let mut hi = 0u32;
                loop {
                    if hi >= s { break; }
                    let mut wi = 0u32;
                    loop {
                        if wi >= s { break; }
                        let in_val = *input.wrapping_add((ic * s * s * s + di * s * s + hi * s + wi) as usize);
                        let mut oc = 0u32;
                        loop {
                            if oc >= out_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kk { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kk { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kk { break; }
                                        let od = di * stride + kdi;
                                        let oh = hi * stride + khi;
                                        let ow = wi * stride + kwi;
                                        let w_idx = (ic * out_ch * kk * kk * kk + oc * kk * kk * kk + kdi * kk * kk + khi * kk + kwi) as usize;
                                        let o_idx = (oc * os * os * os + od * os * os + oh * os + ow) as usize;
                                        let cur = *output.wrapping_add(o_idx);
                                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            oc = oc + 1;
                        }
                        wi = wi + 1;
                    }
                    hi = hi + 1;
                }
                di = di + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 3D convolution with square input and asymmetric kernel
/// Maps to conv/conv_transposed_3d_square_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_sq_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let s = *params.wrapping_add(2);
        let kd = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let kw = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let od = (s - 1) * stride + kd;
        let oh = (s - 1) * stride + kh;
        let ow = (s - 1) * stride + kw;

        let total = out_ch * od * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut di = 0u32;
            loop {
                if di >= s { break; }
                let mut hi = 0u32;
                loop {
                    if hi >= s { break; }
                    let mut wi = 0u32;
                    loop {
                        if wi >= s { break; }
                        let in_val = *input.wrapping_add((ic * s * s * s + di * s * s + hi * s + wi) as usize);
                        let mut oc = 0u32;
                        loop {
                            if oc >= out_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kd { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kh { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kw { break; }
                                        let p_od = di * stride + kdi;
                                        let p_oh = hi * stride + khi;
                                        let p_ow = wi * stride + kwi;
                                        let w_idx = (ic * out_ch * kd * kh * kw + oc * kd * kh * kw + kdi * kh * kw + khi * kw + kwi) as usize;
                                        let o_idx = (oc * od * oh * ow + p_od * oh * ow + p_oh * ow + p_ow) as usize;
                                        let cur = *output.wrapping_add(o_idx);
                                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            oc = oc + 1;
                        }
                        wi = wi + 1;
                    }
                    hi = hi + 1;
                }
                di = di + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 3D convolution with asymmetric input and square kernel
/// Maps to conv/conv_transposed_3d_asymmetric_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_asym_sq(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let id = *params.wrapping_add(2);
        let ih = *params.wrapping_add(3);
        let iw = *params.wrapping_add(4);
        let kk = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let od = (id - 1) * stride + kk;
        let oh = (ih - 1) * stride + kk;
        let ow = (iw - 1) * stride + kk;

        let total = out_ch * od * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut di = 0u32;
            loop {
                if di >= id { break; }
                let mut hi = 0u32;
                loop {
                    if hi >= ih { break; }
                    let mut wi = 0u32;
                    loop {
                        if wi >= iw { break; }
                        let in_val = *input.wrapping_add((ic * id * ih * iw + di * ih * iw + hi * iw + wi) as usize);
                        let mut oc = 0u32;
                        loop {
                            if oc >= out_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kk { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kk { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kk { break; }
                                        let p_od = di * stride + kdi;
                                        let p_oh = hi * stride + khi;
                                        let p_ow = wi * stride + kwi;
                                        let w_idx = (ic * out_ch * kk * kk * kk + oc * kk * kk * kk + kdi * kk * kk + khi * kk + kwi) as usize;
                                        let o_idx = (oc * od * oh * ow + p_od * oh * ow + p_oh * ow + p_ow) as usize;
                                        let cur = *output.wrapping_add(o_idx);
                                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            oc = oc + 1;
                        }
                        wi = wi + 1;
                    }
                    hi = hi + 1;
                }
                di = di + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 3D convolution with asymmetric input and asymmetric kernel
/// Maps to conv/conv_transposed_3d_asymmetric_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_asym_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let id = *params.wrapping_add(2);
        let ih = *params.wrapping_add(3);
        let iw = *params.wrapping_add(4);
        let kd = *params.wrapping_add(5);
        let kh = *params.wrapping_add(6);
        let kw = *params.wrapping_add(7);
        let stride = *params.wrapping_add(8);
        let od = (id - 1) * stride + kd;
        let oh = (ih - 1) * stride + kh;
        let ow = (iw - 1) * stride + kw;

        let total = out_ch * od * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut di = 0u32;
            loop {
                if di >= id { break; }
                let mut hi = 0u32;
                loop {
                    if hi >= ih { break; }
                    let mut wi = 0u32;
                    loop {
                        if wi >= iw { break; }
                        let in_val = *input.wrapping_add((ic * id * ih * iw + di * ih * iw + hi * iw + wi) as usize);
                        let mut oc = 0u32;
                        loop {
                            if oc >= out_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kd { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kh { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kw { break; }
                                        let p_od = di * stride + kdi;
                                        let p_oh = hi * stride + khi;
                                        let p_ow = wi * stride + kwi;
                                        let w_idx = (ic * out_ch * kd * kh * kw + oc * kd * kh * kw + kdi * kh * kw + khi * kw + kwi) as usize;
                                        let o_idx = (oc * od * oh * ow + p_od * oh * ow + p_oh * ow + p_ow) as usize;
                                        let cur = *output.wrapping_add(o_idx);
                                        *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            oc = oc + 1;
                        }
                        wi = wi + 1;
                    }
                    hi = hi + 1;
                }
                di = di + 1;
            }
            ic = ic + 1;
        }
    }
}

/// Transposed 3D convolution with groups, stride, and padding (asym input, square kernel)
/// Maps to conv/conv_transposed_3d_asymmetric_input_square_kernel_strided_padded_grouped.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_asym_sq_grouped(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let id = *params.wrapping_add(2);
        let ih = *params.wrapping_add(3);
        let iw = *params.wrapping_add(4);
        let kk = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let padding = *params.wrapping_add(7);
        let groups = *params.wrapping_add(8);
        let od = (id - 1) * stride + kk - 2 * padding;
        let oh = (ih - 1) * stride + kk - 2 * padding;
        let ow = (iw - 1) * stride + kk - 2 * padding;
        let ic_per_g = in_ch / groups;
        let oc_per_g = out_ch / groups;

        let total = out_ch * od * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut g = 0u32;
        loop {
            if g >= groups { break; }
            let mut ic = 0u32;
            loop {
                if ic >= ic_per_g { break; }
                let abs_ic = g * ic_per_g + ic;
                let mut di = 0u32;
                loop {
                    if di >= id { break; }
                    let mut hi = 0u32;
                    loop {
                        if hi >= ih { break; }
                        let mut wi = 0u32;
                        loop {
                            if wi >= iw { break; }
                            let in_val = *input.wrapping_add((abs_ic * id * ih * iw + di * ih * iw + hi * iw + wi) as usize);
                            let mut oc = 0u32;
                            loop {
                                if oc >= oc_per_g { break; }
                                let abs_oc = g * oc_per_g + oc;
                                let mut kdi = 0u32;
                                loop {
                                    if kdi >= kk { break; }
                                    let mut khi = 0u32;
                                    loop {
                                        if khi >= kk { break; }
                                        let mut kwi = 0u32;
                                        loop {
                                            if kwi >= kk { break; }
                                            let raw_d = di * stride + kdi;
                                            let raw_h = hi * stride + khi;
                                            let raw_w = wi * stride + kwi;
                                            if raw_d >= padding && raw_h >= padding && raw_w >= padding {
                                                let p_od = raw_d - padding;
                                                let p_oh = raw_h - padding;
                                                let p_ow = raw_w - padding;
                                                if p_od < od && p_oh < oh && p_ow < ow {
                                                    let w_idx = (abs_ic * oc_per_g * kk * kk * kk + oc * kk * kk * kk + kdi * kk * kk + khi * kk + kwi) as usize;
                                                    let o_idx = (abs_oc * od * oh * ow + p_od * oh * ow + p_oh * ow + p_ow) as usize;
                                                    let cur = *output.wrapping_add(o_idx);
                                                    *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                                }
                                            }
                                            kwi = kwi + 1;
                                        }
                                        khi = khi + 1;
                                    }
                                    kdi = kdi + 1;
                                }
                                oc = oc + 1;
                            }
                            wi = wi + 1;
                        }
                        hi = hi + 1;
                    }
                    di = di + 1;
                }
                ic = ic + 1;
            }
            g = g + 1;
        }
    }
}

/// Transposed 3D convolution with groups, stride, and padding (asym input, asym kernel)
/// Maps to conv/conv_transposed_3d_asymmetric_input_asymmetric_kernel_strided_padded_grouped.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_asym_asym_grouped(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let id = *params.wrapping_add(2);
        let ih = *params.wrapping_add(3);
        let iw = *params.wrapping_add(4);
        let kd = *params.wrapping_add(5);
        let kh = *params.wrapping_add(6);
        let kw = *params.wrapping_add(7);
        let stride = *params.wrapping_add(8);
        let padding = *params.wrapping_add(9);
        let groups = *params.wrapping_add(10);
        let od = (id - 1) * stride + kd - 2 * padding;
        let oh = (ih - 1) * stride + kh - 2 * padding;
        let ow = (iw - 1) * stride + kw - 2 * padding;
        let ic_per_g = in_ch / groups;
        let oc_per_g = out_ch / groups;

        let total = out_ch * od * oh * ow;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut g = 0u32;
        loop {
            if g >= groups { break; }
            let mut ic = 0u32;
            loop {
                if ic >= ic_per_g { break; }
                let abs_ic = g * ic_per_g + ic;
                let mut di = 0u32;
                loop {
                    if di >= id { break; }
                    let mut hi = 0u32;
                    loop {
                        if hi >= ih { break; }
                        let mut wi = 0u32;
                        loop {
                            if wi >= iw { break; }
                            let in_val = *input.wrapping_add((abs_ic * id * ih * iw + di * ih * iw + hi * iw + wi) as usize);
                            let mut oc = 0u32;
                            loop {
                                if oc >= oc_per_g { break; }
                                let abs_oc = g * oc_per_g + oc;
                                let mut kdi = 0u32;
                                loop {
                                    if kdi >= kd { break; }
                                    let mut khi = 0u32;
                                    loop {
                                        if khi >= kh { break; }
                                        let mut kwi = 0u32;
                                        loop {
                                            if kwi >= kw { break; }
                                            let raw_d = di * stride + kdi;
                                            let raw_h = hi * stride + khi;
                                            let raw_w = wi * stride + kwi;
                                            if raw_d >= padding && raw_h >= padding && raw_w >= padding {
                                                let p_od = raw_d - padding;
                                                let p_oh = raw_h - padding;
                                                let p_ow = raw_w - padding;
                                                if p_od < od && p_oh < oh && p_ow < ow {
                                                    let w_idx = (abs_ic * oc_per_g * kd * kh * kw + oc * kd * kh * kw + kdi * kh * kw + khi * kw + kwi) as usize;
                                                    let o_idx = (abs_oc * od * oh * ow + p_od * oh * ow + p_oh * ow + p_ow) as usize;
                                                    let cur = *output.wrapping_add(o_idx);
                                                    *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                                }
                                            }
                                            kwi = kwi + 1;
                                        }
                                        khi = khi + 1;
                                    }
                                    kdi = kdi + 1;
                                }
                                oc = oc + 1;
                            }
                            wi = wi + 1;
                        }
                        hi = hi + 1;
                    }
                    di = di + 1;
                }
                ic = ic + 1;
            }
            g = g + 1;
        }
    }
}

/// Transposed 3D convolution with dilation, padding, and stride (square input, square kernel)
/// Maps to conv/conv_transposed_3d_square_input_square_kernel_padded_dilated_strided.py
#[tile_std::tile_kernel]
pub fn conv_transposed_3d_sq_sq_dilated(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let s = *params.wrapping_add(2);
        let kk = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let padding = *params.wrapping_add(5);
        let dilation = *params.wrapping_add(6);
        let eff_k = (kk - 1) * dilation + 1;
        let os = (s - 1) * stride + eff_k - 2 * padding;

        let total = out_ch * os * os * os;
        let mut i = 0u32;
        loop { if i >= total { break; } *output.wrapping_add(i as usize) = 0.0f32; i = i + 1; }

        let mut ic = 0u32;
        loop {
            if ic >= in_ch { break; }
            let mut di = 0u32;
            loop {
                if di >= s { break; }
                let mut hi = 0u32;
                loop {
                    if hi >= s { break; }
                    let mut wi = 0u32;
                    loop {
                        if wi >= s { break; }
                        let in_val = *input.wrapping_add((ic * s * s * s + di * s * s + hi * s + wi) as usize);
                        let mut oc = 0u32;
                        loop {
                            if oc >= out_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kk { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kk { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kk { break; }
                                        let raw_d = di * stride + kdi * dilation;
                                        let raw_h = hi * stride + khi * dilation;
                                        let raw_w = wi * stride + kwi * dilation;
                                        if raw_d >= padding && raw_h >= padding && raw_w >= padding {
                                            let p_od = raw_d - padding;
                                            let p_oh = raw_h - padding;
                                            let p_ow = raw_w - padding;
                                            if p_od < os && p_oh < os && p_ow < os {
                                                let w_idx = (ic * out_ch * kk * kk * kk + oc * kk * kk * kk + kdi * kk * kk + khi * kk + kwi) as usize;
                                                let o_idx = (oc * os * os * os + p_od * os * os + p_oh * os + p_ow) as usize;
                                                let cur = *output.wrapping_add(o_idx);
                                                *output.wrapping_add(o_idx) = cur + in_val * *weight.wrapping_add(w_idx);
                                            }
                                        }
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            oc = oc + 1;
                        }
                        wi = wi + 1;
                    }
                    hi = hi + 1;
                }
                di = di + 1;
            }
            ic = ic + 1;
        }
    }
}
