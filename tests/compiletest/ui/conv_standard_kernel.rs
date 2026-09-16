// build-pass

// Standard convolution kernels (1D, 2D, 3D).
// Maps to MultiKernelBench/reference/conv/ category.
// All use scalar nested-loop multiply-accumulate on GM pointers.

#![feature(no_core)]

#![no_std]
#![no_core]

/// 1D convolution: output[oc][p] = sum_{ic,k} input[ic][p*stride+k] * weight[oc][ic][k]
/// Maps to conv/conv_standard_1d.py
#[tile_std::tile_kernel]
pub fn conv_standard_1d(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let in_len = *params.wrapping_add(2);
        let k_size = *params.wrapping_add(3);
        let stride = *params.wrapping_add(4);
        let out_len = (in_len - k_size) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut p = 0u32;
            loop {
                if p >= out_len { break; }
                let mut sum = 0.0f32;
                let mut ic = 0u32;
                loop {
                    if ic >= in_ch { break; }
                    let mut k = 0u32;
                    loop {
                        if k >= k_size { break; }
                        let in_idx = (ic * in_len + p * stride + k) as usize;
                        let w_idx = (oc * in_ch * k_size + ic * k_size + k) as usize;
                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                        k = k + 1;
                    }
                    ic = ic + 1;
                }
                *output.wrapping_add((oc * out_len + p) as usize) = sum;
                p = p + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 1D convolution with dilation and stride > 1
/// Maps to conv/conv_standard_1d_dilated_strided.py
#[tile_std::tile_kernel]
pub fn conv_standard_1d_dilated_strided(
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
        let out_len = (in_len - eff_k) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut p = 0u32;
            loop {
                if p >= out_len { break; }
                let mut sum = 0.0f32;
                let mut ic = 0u32;
                loop {
                    if ic >= in_ch { break; }
                    let mut k = 0u32;
                    loop {
                        if k >= k_size { break; }
                        let in_pos = p * stride + k * dilation;
                        let in_idx = (ic * in_len + in_pos) as usize;
                        let w_idx = (oc * in_ch * k_size + ic * k_size + k) as usize;
                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                        k = k + 1;
                    }
                    ic = ic + 1;
                }
                *output.wrapping_add((oc * out_len + p) as usize) = sum;
                p = p + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 2D convolution with square input and square kernel
/// Maps to conv/conv_standard_2d_square_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_2d_square_square(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let h = *params.wrapping_add(2); // square: h == w
        let kh = *params.wrapping_add(3); // square: kh == kw
        let stride = *params.wrapping_add(4);
        let oh = (h - kh) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut oh_i = 0u32;
            loop {
                if oh_i >= oh { break; }
                let mut ow_i = 0u32;
                loop {
                    if ow_i >= oh { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kh { break; }
                                let ih = oh_i * stride + ki;
                                let iw = ow_i * stride + kj;
                                let in_idx = (ic * h * h + ih * h + iw) as usize;
                                let w_idx = (oc * in_ch * kh * kh + ic * kh * kh + ki * kh + kj) as usize;
                                sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        ic = ic + 1;
                    }
                    *output.wrapping_add((oc * oh * oh + oh_i * oh + ow_i) as usize) = sum;
                    ow_i = ow_i + 1;
                }
                oh_i = oh_i + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 2D convolution with asymmetric input and square kernel
/// Maps to conv/conv_standard_2d_asymmetric_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_2d_asym_square(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let ih = *params.wrapping_add(2);
        let iw = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kh) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kh { break; }
                                let r = ohi * stride + ki;
                                let c = owi * stride + kj;
                                let in_idx = (ic * ih * iw + r * iw + c) as usize;
                                let w_idx = (oc * in_ch * kh * kh + ic * kh * kh + ki * kh + kj) as usize;
                                sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        ic = ic + 1;
                    }
                    *output.wrapping_add((oc * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 2D convolution with square input and asymmetric kernel
/// Maps to conv/conv_standard_2d_square_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_2d_square_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let h = *params.wrapping_add(2);
        let kh = *params.wrapping_add(3);
        let kw = *params.wrapping_add(4);
        let stride = *params.wrapping_add(5);
        let oh = (h - kh) / stride + 1;
        let ow = (h - kw) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kw { break; }
                                let r = ohi * stride + ki;
                                let c = owi * stride + kj;
                                let in_idx = (ic * h * h + r * h + c) as usize;
                                let w_idx = (oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj) as usize;
                                sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        ic = ic + 1;
                    }
                    *output.wrapping_add((oc * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 2D convolution with asymmetric input and asymmetric kernel
/// Maps to conv/conv_standard_2d_asymmetric_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_2d_asym_asym(
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
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kw) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kw { break; }
                                let r = ohi * stride + ki;
                                let c = owi * stride + kj;
                                let in_idx = (ic * ih * iw + r * iw + c) as usize;
                                let w_idx = (oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj) as usize;
                                sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        ic = ic + 1;
                    }
                    *output.wrapping_add((oc * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 2D convolution with dilation and padding
/// Maps to conv/conv_standard_2d_square_input_asymmetric_kernel_dilated_padded.py
#[tile_std::tile_kernel]
pub fn conv_standard_2d_dilated_padded(
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
        let dilation = *params.wrapping_add(8);
        let eff_kh = (kh - 1) * dilation + 1;
        let eff_kw = (kw - 1) * dilation + 1;
        let oh = (ih + 2 * padding - eff_kh) / stride + 1;
        let ow = (iw + 2 * padding - eff_kw) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut ohi = 0u32;
            loop {
                if ohi >= oh { break; }
                let mut owi = 0u32;
                loop {
                    if owi >= ow { break; }
                    let mut sum = 0.0f32;
                    let mut ic = 0u32;
                    loop {
                        if ic >= in_ch { break; }
                        let mut ki = 0u32;
                        loop {
                            if ki >= kh { break; }
                            let mut kj = 0u32;
                            loop {
                                if kj >= kw { break; }
                                let r = ohi * stride + ki * dilation;
                                let c = owi * stride + kj * dilation;
                                if r >= padding && c >= padding {
                                    let ri = r - padding;
                                    let ci = c - padding;
                                    if ri < ih && ci < iw {
                                        let in_idx = (ic * ih * iw + ri * iw + ci) as usize;
                                        let w_idx = (oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj) as usize;
                                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                    }
                                }
                                kj = kj + 1;
                            }
                            ki = ki + 1;
                        }
                        ic = ic + 1;
                    }
                    *output.wrapping_add((oc * oh * ow + ohi * ow + owi) as usize) = sum;
                    owi = owi + 1;
                }
                ohi = ohi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 3D convolution with square input and square kernel
/// Maps to conv/conv_standard_3d_square_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_3d_square_square(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let d = *params.wrapping_add(2); // square: d == h == w
        let kd = *params.wrapping_add(3); // square: kd == kh == kw
        let stride = *params.wrapping_add(4);
        let od = (d - kd) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut odi = 0u32;
            loop {
                if odi >= od { break; }
                let mut ohi = 0u32;
                loop {
                    if ohi >= od { break; }
                    let mut owi = 0u32;
                    loop {
                        if owi >= od { break; }
                        let mut sum = 0.0f32;
                        let mut ic = 0u32;
                        loop {
                            if ic >= in_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kd { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kd { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kd { break; }
                                        let id = odi * stride + kdi;
                                        let ih = ohi * stride + khi;
                                        let iw = owi * stride + kwi;
                                        let in_idx = (ic * d * d * d + id * d * d + ih * d + iw) as usize;
                                        let w_idx = (oc * in_ch * kd * kd * kd + ic * kd * kd * kd + kdi * kd * kd + khi * kd + kwi) as usize;
                                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            ic = ic + 1;
                        }
                        *output.wrapping_add((oc * od * od * od + odi * od * od + ohi * od + owi) as usize) = sum;
                        owi = owi + 1;
                    }
                    ohi = ohi + 1;
                }
                odi = odi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 3D convolution with asymmetric input and square kernel
/// Maps to conv/conv_standard_3d_asymmetric_input_square_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_3d_asym_square(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let id = *params.wrapping_add(2);
        let ih = *params.wrapping_add(3);
        let iw = *params.wrapping_add(4);
        let kk = *params.wrapping_add(5); // square kernel
        let stride = *params.wrapping_add(6);
        let od = (id - kk) / stride + 1;
        let oh = (ih - kk) / stride + 1;
        let ow = (iw - kk) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut odi = 0u32;
            loop {
                if odi >= od { break; }
                let mut ohi = 0u32;
                loop {
                    if ohi >= oh { break; }
                    let mut owi = 0u32;
                    loop {
                        if owi >= ow { break; }
                        let mut sum = 0.0f32;
                        let mut ic = 0u32;
                        loop {
                            if ic >= in_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kk { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kk { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kk { break; }
                                        let pd = odi * stride + kdi;
                                        let ph = ohi * stride + khi;
                                        let pw = owi * stride + kwi;
                                        let in_idx = (ic * id * ih * iw + pd * ih * iw + ph * iw + pw) as usize;
                                        let w_idx = (oc * in_ch * kk * kk * kk + ic * kk * kk * kk + kdi * kk * kk + khi * kk + kwi) as usize;
                                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            ic = ic + 1;
                        }
                        *output.wrapping_add((oc * od * oh * ow + odi * oh * ow + ohi * ow + owi) as usize) = sum;
                        owi = owi + 1;
                    }
                    ohi = ohi + 1;
                }
                odi = odi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 3D convolution with square input and asymmetric kernel
/// Maps to conv/conv_standard_3d_square_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_3d_square_asym(
    input: *const f32, weight: *const f32, output: *mut f32, params: *const u32,
) {
    unsafe {
        let in_ch = *params;
        let out_ch = *params.wrapping_add(1);
        let s = *params.wrapping_add(2); // square input: d == h == w == s
        let kd = *params.wrapping_add(3);
        let kh = *params.wrapping_add(4);
        let kw = *params.wrapping_add(5);
        let stride = *params.wrapping_add(6);
        let od = (s - kd) / stride + 1;
        let oh = (s - kh) / stride + 1;
        let ow = (s - kw) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut odi = 0u32;
            loop {
                if odi >= od { break; }
                let mut ohi = 0u32;
                loop {
                    if ohi >= oh { break; }
                    let mut owi = 0u32;
                    loop {
                        if owi >= ow { break; }
                        let mut sum = 0.0f32;
                        let mut ic = 0u32;
                        loop {
                            if ic >= in_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kd { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kh { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kw { break; }
                                        let pd = odi * stride + kdi;
                                        let ph = ohi * stride + khi;
                                        let pw = owi * stride + kwi;
                                        let in_idx = (ic * s * s * s + pd * s * s + ph * s + pw) as usize;
                                        let w_idx = (oc * in_ch * kd * kh * kw + ic * kd * kh * kw + kdi * kh * kw + khi * kw + kwi) as usize;
                                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            ic = ic + 1;
                        }
                        *output.wrapping_add((oc * od * oh * ow + odi * oh * ow + ohi * ow + owi) as usize) = sum;
                        owi = owi + 1;
                    }
                    ohi = ohi + 1;
                }
                odi = odi + 1;
            }
            oc = oc + 1;
        }
    }
}

/// 3D convolution with asymmetric input and asymmetric kernel
/// Maps to conv/conv_standard_3d_asymmetric_input_asymmetric_kernel.py
#[tile_std::tile_kernel]
pub fn conv_standard_3d_asym_asym(
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
        let od = (id - kd) / stride + 1;
        let oh = (ih - kh) / stride + 1;
        let ow = (iw - kw) / stride + 1;

        let mut oc = 0u32;
        loop {
            if oc >= out_ch { break; }
            let mut odi = 0u32;
            loop {
                if odi >= od { break; }
                let mut ohi = 0u32;
                loop {
                    if ohi >= oh { break; }
                    let mut owi = 0u32;
                    loop {
                        if owi >= ow { break; }
                        let mut sum = 0.0f32;
                        let mut ic = 0u32;
                        loop {
                            if ic >= in_ch { break; }
                            let mut kdi = 0u32;
                            loop {
                                if kdi >= kd { break; }
                                let mut khi = 0u32;
                                loop {
                                    if khi >= kh { break; }
                                    let mut kwi = 0u32;
                                    loop {
                                        if kwi >= kw { break; }
                                        let pd = odi * stride + kdi;
                                        let ph = ohi * stride + khi;
                                        let pw = owi * stride + kwi;
                                        let in_idx = (ic * id * ih * iw + pd * ih * iw + ph * iw + pw) as usize;
                                        let w_idx = (oc * in_ch * kd * kh * kw + ic * kd * kh * kw + kdi * kh * kw + khi * kw + kwi) as usize;
                                        sum = sum + *input.wrapping_add(in_idx) * *weight.wrapping_add(w_idx);
                                        kwi = kwi + 1;
                                    }
                                    khi = khi + 1;
                                }
                                kdi = kdi + 1;
                            }
                            ic = ic + 1;
                        }
                        *output.wrapping_add((oc * od * oh * ow + odi * oh * ow + ohi * ow + owi) as usize) = sum;
                        owi = owi + 1;
                    }
                    ohi = ohi + 1;
                }
                odi = odi + 1;
            }
            oc = oc + 1;
        }
    }
}
