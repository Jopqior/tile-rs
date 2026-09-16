//! CPU reference implementations of convolution kernels.
//! Mirrors: conv_standard_kernel.rs, conv_depthwise_kernel.rs, conv_transpose_kernel.rs

// ============================================================
// Standard convolutions
// ============================================================

/// 1D convolution: output[oc][p] = sum_{ic,k} input[ic][p*stride+k] * weight[oc][ic][k]
pub fn conv_standard_1d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
) {
    let out_len = (in_len - k_size) / stride + 1;
    for oc in 0..out_ch {
        for p in 0..out_len {
            let mut sum = 0.0f32;
            for ic in 0..in_ch {
                for k in 0..k_size {
                    let in_idx = ic * in_len + p * stride + k;
                    let w_idx = oc * in_ch * k_size + ic * k_size + k;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
            output[oc * out_len + p] = sum;
        }
    }
}

/// 1D convolution with dilation and stride > 1
pub fn conv_standard_1d_dilated_strided(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
    dilation: usize,
) {
    let eff_k = (k_size - 1) * dilation + 1;
    let out_len = (in_len - eff_k) / stride + 1;
    for oc in 0..out_ch {
        for p in 0..out_len {
            let mut sum = 0.0f32;
            for ic in 0..in_ch {
                for k in 0..k_size {
                    let in_pos = p * stride + k * dilation;
                    let in_idx = ic * in_len + in_pos;
                    let w_idx = oc * in_ch * k_size + ic * k_size + k;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
            output[oc * out_len + p] = sum;
        }
    }
}

/// 2D convolution (general: asymmetric input, asymmetric kernel)
pub fn conv_standard_2d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let oh = (ih - kh) / stride + 1;
    let ow = (iw - kw) / stride + 1;
    for oc in 0..out_ch {
        for ohi in 0..oh {
            for owi in 0..ow {
                let mut sum = 0.0f32;
                for ic in 0..in_ch {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let r = ohi * stride + ki;
                            let c = owi * stride + kj;
                            let in_idx = ic * ih * iw + r * iw + c;
                            let w_idx = oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj;
                            sum += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                output[oc * oh * ow + ohi * ow + owi] = sum;
            }
        }
    }
}

/// 2D convolution with dilation and padding
pub fn conv_standard_2d_dilated_padded(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    let eff_kh = (kh - 1) * dilation + 1;
    let eff_kw = (kw - 1) * dilation + 1;
    let oh = (ih + 2 * padding - eff_kh) / stride + 1;
    let ow = (iw + 2 * padding - eff_kw) / stride + 1;
    for oc in 0..out_ch {
        for ohi in 0..oh {
            for owi in 0..ow {
                let mut sum = 0.0f32;
                for ic in 0..in_ch {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let r = ohi * stride + ki * dilation;
                            let c = owi * stride + kj * dilation;
                            if r >= padding && c >= padding {
                                let ri = r - padding;
                                let ci = c - padding;
                                if ri < ih && ci < iw {
                                    let in_idx = ic * ih * iw + ri * iw + ci;
                                    let w_idx = oc * in_ch * kh * kw + ic * kh * kw + ki * kw + kj;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                }
                output[oc * oh * ow + ohi * ow + owi] = sum;
            }
        }
    }
}

/// 3D convolution (general: asymmetric input, asymmetric kernel)
pub fn conv_standard_3d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    id: usize,
    ih: usize,
    iw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let od = (id - kd) / stride + 1;
    let oh = (ih - kh) / stride + 1;
    let ow = (iw - kw) / stride + 1;
    for oc in 0..out_ch {
        for odi in 0..od {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let mut sum = 0.0f32;
                    for ic in 0..in_ch {
                        for kdi in 0..kd {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let pd = odi * stride + kdi;
                                    let ph = ohi * stride + khi;
                                    let pw = owi * stride + kwi;
                                    let in_idx = ic * id * ih * iw + pd * ih * iw + ph * iw + pw;
                                    let w_idx = oc * in_ch * kd * kh * kw
                                        + ic * kd * kh * kw
                                        + kdi * kh * kw
                                        + khi * kw
                                        + kwi;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    output[oc * od * oh * ow + odi * oh * ow + ohi * ow + owi] = sum;
                }
            }
        }
    }
}

// ============================================================
// Depthwise convolutions
// ============================================================

/// Depthwise 2D convolution (general: asymmetric input, asymmetric kernel)
pub fn conv_depthwise_2d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let oh = (ih - kh) / stride + 1;
    let ow = (iw - kw) / stride + 1;
    for c in 0..ch {
        for ohi in 0..oh {
            for owi in 0..ow {
                let mut sum = 0.0f32;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let r = ohi * stride + ki;
                        let col = owi * stride + kj;
                        let in_idx = c * ih * iw + r * iw + col;
                        let w_idx = c * kh * kw + ki * kw + kj;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
                output[c * oh * ow + ohi * ow + owi] = sum;
            }
        }
    }
}

/// Pointwise 2D convolution (1x1 kernel)
pub fn conv_pointwise_2d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
) {
    for oc in 0..out_ch {
        for hi in 0..h {
            for wi in 0..w {
                let mut sum = 0.0f32;
                for ic in 0..in_ch {
                    let in_idx = ic * h * w + hi * w + wi;
                    let w_idx = oc * in_ch + ic;
                    sum += input[in_idx] * weight[w_idx];
                }
                output[oc * h * w + hi * w + wi] = sum;
            }
        }
    }
}

// ============================================================
// Transposed convolutions
// ============================================================

/// Transposed 1D convolution (scatter-add pattern)
pub fn conv_transposed_1d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
) {
    let out_len = (in_len - 1) * stride + k_size;
    // Zero output
    for v in output[..out_ch * out_len].iter_mut() {
        *v = 0.0;
    }
    for ic in 0..in_ch {
        for p in 0..in_len {
            let in_val = input[ic * in_len + p];
            for oc in 0..out_ch {
                for k in 0..k_size {
                    let out_pos = p * stride + k;
                    let w_idx = ic * out_ch * k_size + oc * k_size + k;
                    output[oc * out_len + out_pos] += in_val * weight[w_idx];
                }
            }
        }
    }
}

/// Transposed 1D convolution with dilation
pub fn conv_transposed_1d_dilated(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
    dilation: usize,
) {
    let eff_k = (k_size - 1) * dilation + 1;
    let out_len = (in_len - 1) * stride + eff_k;
    for v in output[..out_ch * out_len].iter_mut() {
        *v = 0.0;
    }
    for ic in 0..in_ch {
        for p in 0..in_len {
            let in_val = input[ic * in_len + p];
            for oc in 0..out_ch {
                for k in 0..k_size {
                    let out_pos = p * stride + k * dilation;
                    let w_idx = ic * out_ch * k_size + oc * k_size + k;
                    output[oc * out_len + out_pos] += in_val * weight[w_idx];
                }
            }
        }
    }
}

/// Transposed 1D convolution with padding, stride, dilation
pub fn conv_transposed_1d_padded(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    in_len: usize,
    k_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) {
    let eff_k = (k_size - 1) * dilation + 1;
    let out_len = (in_len - 1) * stride + eff_k - 2 * padding;
    for v in output[..out_ch * out_len].iter_mut() {
        *v = 0.0;
    }
    for ic in 0..in_ch {
        for p in 0..in_len {
            let in_val = input[ic * in_len + p];
            for oc in 0..out_ch {
                for k in 0..k_size {
                    let raw_pos = p * stride + k * dilation;
                    if raw_pos >= padding {
                        let out_pos = raw_pos - padding;
                        if out_pos < out_len {
                            let w_idx = ic * out_ch * k_size + oc * k_size + k;
                            output[oc * out_len + out_pos] += in_val * weight[w_idx];
                        }
                    }
                }
            }
        }
    }
}

/// Transposed 2D convolution (general)
pub fn conv_transposed_2d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let oh = (ih - 1) * stride + kh;
    let ow = (iw - 1) * stride + kw;
    for v in output[..out_ch * oh * ow].iter_mut() {
        *v = 0.0;
    }
    for ic in 0..in_ch {
        for hi in 0..ih {
            for wi in 0..iw {
                let in_val = input[ic * ih * iw + hi * iw + wi];
                for oc in 0..out_ch {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let or = hi * stride + ki;
                            let ocol = wi * stride + kj;
                            let w_idx = ic * out_ch * kh * kw + oc * kh * kw + ki * kw + kj;
                            output[oc * oh * ow + or * ow + ocol] += in_val * weight[w_idx];
                        }
                    }
                }
            }
        }
    }
}

/// Transposed 2D convolution with padding
pub fn conv_transposed_2d_padded(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
) {
    let oh = (ih - 1) * stride + kh - 2 * padding;
    let ow = (iw - 1) * stride + kw - 2 * padding;
    for v in output[..out_ch * oh * ow].iter_mut() {
        *v = 0.0;
    }
    for ic in 0..in_ch {
        for hi in 0..ih {
            for wi in 0..iw {
                let in_val = input[ic * ih * iw + hi * iw + wi];
                for oc in 0..out_ch {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let raw_r = hi * stride + ki;
                            let raw_c = wi * stride + kj;
                            if raw_r >= padding && raw_c >= padding {
                                let or = raw_r - padding;
                                let ocol = raw_c - padding;
                                if or < oh && ocol < ow {
                                    let w_idx = ic * out_ch * kh * kw + oc * kh * kw + ki * kw + kj;
                                    output[oc * oh * ow + or * ow + ocol] += in_val * weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Transposed 2D convolution with groups
pub fn conv_transposed_2d_grouped(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    groups: usize,
) {
    let oh = (ih - 1) * stride + kh - 2 * padding;
    let ow = (iw - 1) * stride + kw - 2 * padding;
    let ic_per_g = in_ch / groups;
    let oc_per_g = out_ch / groups;
    for v in output[..out_ch * oh * ow].iter_mut() {
        *v = 0.0;
    }
    for g in 0..groups {
        for ic in 0..ic_per_g {
            let abs_ic = g * ic_per_g + ic;
            for hi in 0..ih {
                for wi in 0..iw {
                    let in_val = input[abs_ic * ih * iw + hi * iw + wi];
                    for oc in 0..oc_per_g {
                        let abs_oc = g * oc_per_g + oc;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let raw_r = hi * stride + ki;
                                let raw_c = wi * stride + kj;
                                if raw_r >= padding && raw_c >= padding {
                                    let or = raw_r - padding;
                                    let ocol = raw_c - padding;
                                    if or < oh && ocol < ow {
                                        let w_idx = abs_ic * oc_per_g * kh * kw
                                            + oc * kh * kw
                                            + ki * kw
                                            + kj;
                                        output[abs_oc * oh * ow + or * ow + ocol] +=
                                            in_val * weight[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Transposed 3D convolution (general)
pub fn conv_transposed_3d(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    id: usize,
    ih: usize,
    iw: usize,
    kd: usize,
    kh: usize,
    kw: usize,
    stride: usize,
) {
    let od = (id - 1) * stride + kd;
    let oh = (ih - 1) * stride + kh;
    let ow = (iw - 1) * stride + kw;
    for v in output[..out_ch * od * oh * ow].iter_mut() {
        *v = 0.0;
    }
    for ic in 0..in_ch {
        for di in 0..id {
            for hi in 0..ih {
                for wi in 0..iw {
                    let in_val = input[ic * id * ih * iw + di * ih * iw + hi * iw + wi];
                    for oc in 0..out_ch {
                        for kdi in 0..kd {
                            for khi in 0..kh {
                                for kwi in 0..kw {
                                    let p_od = di * stride + kdi;
                                    let p_oh = hi * stride + khi;
                                    let p_ow = wi * stride + kwi;
                                    let w_idx = ic * out_ch * kd * kh * kw
                                        + oc * kd * kh * kw
                                        + kdi * kh * kw
                                        + khi * kw
                                        + kwi;
                                    output
                                        [oc * od * oh * ow + p_od * oh * ow + p_oh * ow + p_ow] +=
                                        in_val * weight[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
