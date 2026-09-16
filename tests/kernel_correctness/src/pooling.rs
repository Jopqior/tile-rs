//! CPU reference implementations of windowed pooling kernels.
//! Mirrors: pooling_windowed_kernel.rs

/// Max pooling 1D
pub fn max_pooling_1d(
    input: &[f32],
    output: &mut [f32],
    in_len: usize,
    k_size: usize,
    stride: usize,
) {
    let out_len = (in_len - k_size) / stride + 1;
    for i in 0..out_len {
        let base = i * stride;
        let mut max_val = input[base];
        for k in 1..k_size {
            let val = input[base + k];
            if val > max_val {
                max_val = val;
            }
        }
        output[i] = max_val;
    }
}

/// Max pooling 2D
pub fn max_pooling_2d(
    input: &[f32],
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
                let base_h = ohi * stride;
                let base_w = owi * stride;
                let mut max_val = input[c * ih * iw + base_h * iw + base_w];
                for ki in 0..kh {
                    for kj in 0..kw {
                        let val = input[c * ih * iw + (base_h + ki) * iw + base_w + kj];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                output[c * oh * ow + ohi * ow + owi] = max_val;
            }
        }
    }
}

/// Max pooling 3D
pub fn max_pooling_3d(
    input: &[f32],
    output: &mut [f32],
    ch: usize,
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
    for c in 0..ch {
        for odi in 0..od {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let bd = odi * stride;
                    let bh = ohi * stride;
                    let bw = owi * stride;
                    let mut max_val = input[c * id * ih * iw + bd * ih * iw + bh * iw + bw];
                    for di in 0..kd {
                        for hi in 0..kh {
                            for wi in 0..kw {
                                let val = input[c * id * ih * iw
                                    + (bd + di) * ih * iw
                                    + (bh + hi) * iw
                                    + bw
                                    + wi];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    output[c * od * oh * ow + odi * oh * ow + ohi * ow + owi] = max_val;
                }
            }
        }
    }
}

/// Average pooling 1D
pub fn average_pooling_1d(
    input: &[f32],
    output: &mut [f32],
    in_len: usize,
    k_size: usize,
    stride: usize,
) {
    let out_len = (in_len - k_size) / stride + 1;
    let inv_k = 1.0 / k_size as f32;
    for i in 0..out_len {
        let base = i * stride;
        let mut sum = 0.0f32;
        for k in 0..k_size {
            sum += input[base + k];
        }
        output[i] = sum * inv_k;
    }
}

/// Average pooling 2D
pub fn average_pooling_2d(
    input: &[f32],
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
    let inv_k = 1.0 / (kh * kw) as f32;
    for c in 0..ch {
        for ohi in 0..oh {
            for owi in 0..ow {
                let base_h = ohi * stride;
                let base_w = owi * stride;
                let mut sum = 0.0f32;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += input[c * ih * iw + (base_h + ki) * iw + base_w + kj];
                    }
                }
                output[c * oh * ow + ohi * ow + owi] = sum * inv_k;
            }
        }
    }
}

/// Average pooling 3D
pub fn average_pooling_3d(
    input: &[f32],
    output: &mut [f32],
    ch: usize,
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
    let inv_k = 1.0 / (kd * kh * kw) as f32;
    for c in 0..ch {
        for odi in 0..od {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let bd = odi * stride;
                    let bh = ohi * stride;
                    let bw = owi * stride;
                    let mut sum = 0.0f32;
                    for di in 0..kd {
                        for hi in 0..kh {
                            for wi in 0..kw {
                                sum += input[c * id * ih * iw
                                    + (bd + di) * ih * iw
                                    + (bh + hi) * iw
                                    + bw
                                    + wi];
                            }
                        }
                    }
                    output[c * od * oh * ow + odi * oh * ow + ohi * ow + owi] = sum * inv_k;
                }
            }
        }
    }
}
