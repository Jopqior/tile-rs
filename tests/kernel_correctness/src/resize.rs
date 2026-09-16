//! CPU reference implementations of resize/interpolation kernels.
//! Mirrors: resize_spatial_kernel.rs

/// Bilinear upsample 2D (align_corners style)
pub fn bilinear_upsample_2d(
    input: &[f32],
    output: &mut [f32],
    ch: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
) {
    let denom_h = if oh > 1 { oh - 1 } else { 1 };
    let denom_w = if ow > 1 { ow - 1 } else { 1 };
    for c in 0..ch {
        for ohi in 0..oh {
            for owi in 0..ow {
                let src_h_num = ohi * (ih - 1);
                let src_w_num = owi * (iw - 1);
                let h0 = src_h_num / denom_h;
                let w0 = src_w_num / denom_w;
                let h1 = if h0 + 1 < ih { h0 + 1 } else { h0 };
                let w1 = if w0 + 1 < iw { w0 + 1 } else { w0 };
                let fh = (src_h_num - h0 * denom_h) as f32 / denom_h as f32;
                let fw = (src_w_num - w0 * denom_w) as f32 / denom_w as f32;

                let base = c * ih * iw;
                let v00 = input[base + h0 * iw + w0];
                let v01 = input[base + h0 * iw + w1];
                let v10 = input[base + h1 * iw + w0];
                let v11 = input[base + h1 * iw + w1];
                let val = v00 * (1.0 - fh) * (1.0 - fw)
                    + v01 * (1.0 - fh) * fw
                    + v10 * fh * (1.0 - fw)
                    + v11 * fh * fw;
                output[c * oh * ow + ohi * ow + owi] = val;
            }
        }
    }
}

/// Nearest-neighbor upsample 2D
pub fn nearest_upsample_2d(
    input: &[f32],
    output: &mut [f32],
    ch: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
) {
    for c in 0..ch {
        for ohi in 0..oh {
            for owi in 0..ow {
                let sh = ohi * ih / oh;
                let sw = owi * iw / ow;
                output[c * oh * ow + ohi * ow + owi] = input[c * ih * iw + sh * iw + sw];
            }
        }
    }
}

/// Trilinear upsample 3D (align_corners style)
pub fn trilinear_upsample_3d(
    input: &[f32],
    output: &mut [f32],
    ch: usize,
    id: usize,
    ih: usize,
    iw: usize,
    od: usize,
    oh: usize,
    ow: usize,
) {
    let dd = if od > 1 { od - 1 } else { 1 };
    let dh = if oh > 1 { oh - 1 } else { 1 };
    let dw = if ow > 1 { ow - 1 } else { 1 };
    for c in 0..ch {
        for odi in 0..od {
            for ohi in 0..oh {
                for owi in 0..ow {
                    let sd_num = odi * (id - 1);
                    let sh_num = ohi * (ih - 1);
                    let sw_num = owi * (iw - 1);
                    let d0 = sd_num / dd;
                    let h0 = sh_num / dh;
                    let w0 = sw_num / dw;
                    let d1 = if d0 + 1 < id { d0 + 1 } else { d0 };
                    let h1 = if h0 + 1 < ih { h0 + 1 } else { h0 };
                    let w1 = if w0 + 1 < iw { w0 + 1 } else { w0 };
                    let fd = (sd_num - d0 * dd) as f32 / dd as f32;
                    let fh = (sh_num - h0 * dh) as f32 / dh as f32;
                    let fw = (sw_num - w0 * dw) as f32 / dw as f32;

                    let base = c * id * ih * iw;
                    let v000 = input[base + d0 * ih * iw + h0 * iw + w0];
                    let v001 = input[base + d0 * ih * iw + h0 * iw + w1];
                    let v010 = input[base + d0 * ih * iw + h1 * iw + w0];
                    let v011 = input[base + d0 * ih * iw + h1 * iw + w1];
                    let v100 = input[base + d1 * ih * iw + h0 * iw + w0];
                    let v101 = input[base + d1 * ih * iw + h0 * iw + w1];
                    let v110 = input[base + d1 * ih * iw + h1 * iw + w0];
                    let v111 = input[base + d1 * ih * iw + h1 * iw + w1];

                    let val = v000 * (1.0 - fd) * (1.0 - fh) * (1.0 - fw)
                        + v001 * (1.0 - fd) * (1.0 - fh) * fw
                        + v010 * (1.0 - fd) * fh * (1.0 - fw)
                        + v011 * (1.0 - fd) * fh * fw
                        + v100 * fd * (1.0 - fh) * (1.0 - fw)
                        + v101 * fd * (1.0 - fh) * fw
                        + v110 * fd * fh * (1.0 - fw)
                        + v111 * fd * fh * fw;
                    output[c * od * oh * ow + odi * oh * ow + ohi * ow + owi] = val;
                }
            }
        }
    }
}

/// Downsample bilinear 2D (same algorithm as upsample, just oh < ih)
pub fn downsample_bilinear_2d(
    input: &[f32],
    output: &mut [f32],
    ch: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
) {
    bilinear_upsample_2d(input, output, ch, ih, iw, oh, ow);
}
