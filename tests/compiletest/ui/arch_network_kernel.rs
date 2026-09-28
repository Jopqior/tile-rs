// build-pass

// Network architecture building blocks (simplified forward passes).
// Maps to MultiKernelBench/reference/arch/ category.
// Full networks use conv2d (not in tile_std), so these implement
// the FC/attention/norm layers as representative patterns.

#![feature(no_core)]

#![no_std]
#![no_core]

/// AlexNet-style: FC + ReLU + dropout (dropout = identity at inference)
#[tile_std::tile_kernel]
pub fn alexnet_fc(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// VGG-style: FC + ReLU + bias
#[tile_std::tile_kernel]
pub fn vgg_fc(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// ResNet residual block: x + relu(norm(matmul(x, W)))
#[tile_std::tile_kernel]
pub fn resnet_residual(x: *const u16, w: *const u16, residual: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut res = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(res, residual, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, total);
        tile_std::__tile_pipe_barrier();
        // res dead after add
        tile_std::__tile_v_add_f32(res, work, res, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, res, total);
    }
}

/// DenseNet: concat = add (simplified), then norm + relu + FC
#[tile_std::tile_kernel]
pub fn densenet_block(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// MobileNet depthwise-separable (pointwise FC part): FC + relu6
#[tile_std::tile_kernel]
pub fn mobilenet_pointwise(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        // relu6 = min(max(x, 0), 6)
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_mins_f32(buf, buf, 6.0f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// EfficientNet: FC + swish (SiLU)
#[tile_std::tile_kernel]
pub fn efficientnet_fc(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::swish_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// GoogLeNet inception: parallel FCs merged (simplified as weighted sum)
#[tile_std::tile_kernel]
pub fn inception_merge(a: *const f32, b: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();
        // bb dead after add
        tile_std::__tile_v_add_f32(bb, ba, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(bb, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}

/// SqueezeNet: squeeze (FC) + expand (FC) with relu
#[tile_std::tile_kernel]
pub fn squeezenet_fire(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        // Squeeze: scale down
        tile_std::__tile_muls_f32(buf, buf, 0.25f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        // Expand: scale up
        tile_std::__tile_muls_f32(buf, buf, 4.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// ShuffleNet: channel shuffle = rearrange + FC
#[tile_std::tile_kernel]
pub fn shufflenet_fc(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// RegNet: stem block (norm + relu + scale)
#[tile_std::tile_kernel]
pub fn regnet_stem(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 0.1f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// LeNet-5 FC layer: matmul + tanh (original uses tanh, not relu)
#[tile_std::tile_kernel]
pub fn lenet_fc(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::tanh_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// UNet skip connection: add + norm
#[tile_std::tile_kernel]
pub fn unet_skip(encoder: *const f32, decoder: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut be = tile_std::__tile_buf_alloc(n);
        let mut bd = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(be, encoder, n);
        tile_std::__tile_buf_load_f32(bd, decoder, n);
        tile_std::__tile_pipe_barrier();
        // bd dead after add
        tile_std::__tile_v_add_f32(bd, be, bd, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &bd, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Vision Transformer: norm + matmul + gelu (MLP block)
#[tile_std::tile_kernel]
pub fn vit_mlp(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::gelu_f32(&mut tmp, &work, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// Swin Transformer: window attention (simplified: softmax + scale)
#[tile_std::tile_kernel]
pub fn swin_attention(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// MinGPT: LayerNorm + attention + residual
#[tile_std::tile_kernel]
pub fn mingpt_block(input: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut res = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_buf_load_f32(res, residual, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut extra, &mut work, &mut buf, n);
        tile_std::__tile_pipe_barrier();
        // res dead after add
        tile_std::__tile_v_add_f32(res, extra, res, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, res, n);
    }
}

/// MLP Mixer: transpose-like mixing via FC
#[tile_std::tile_kernel]
pub fn mlp_mixer(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::gelu_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// Mamba selective scan (simplified: sigmoid gate * linear)
#[tile_std::tile_kernel]
pub fn mamba_ssm(x: *const f32, gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bg, bg, n);
        tile_std::__tile_pipe_barrier();
        // bg dead after
        tile_std::__tile_v_mul_f32(bg, bx, bg, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bg, n);
    }
}

// === Split variants for 1:1 MKB kernel mapping ===

/// DenseNet-121: norm + relu + scale (maps to arch/densenet121.py)
#[tile_std::tile_kernel]
pub fn densenet121(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// DenseNet-121 dense block: norm + relu + scale (same as densenet121)
#[tile_std::tile_kernel]
pub fn densenet121_dense_block(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 0.5f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// DenseNet-121 transition layer: norm + relu + scale + avgpool (scale=0.25)
#[tile_std::tile_kernel]
pub fn densenet121_transition_layer(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 0.25f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// DenseNet-201: norm + relu + scale (deeper variant, scale=0.3)
#[tile_std::tile_kernel]
pub fn densenet201(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, n, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(work, work, 0.3f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// EfficientNet-B0: FC + swish (same as efficientnet_fc)
#[tile_std::tile_kernel]
pub fn efficientnet_b0(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::swish_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// EfficientNet-B1: FC + swish (wider variant)
#[tile_std::tile_kernel]
pub fn efficientnet_b1(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::swish_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// EfficientNet-B2: FC + swish (deeper variant)
#[tile_std::tile_kernel]
pub fn efficientnet_b2(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::swish_f32(&mut tmp, &buf, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// ResNet-18: residual block with residual add
#[tile_std::tile_kernel]
pub fn resnet18(x: *const u16, w: *const u16, residual: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut res = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(res, residual, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(res, work, res, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, res, total);
    }
}

/// ResNet-101: residual block (deeper variant)
#[tile_std::tile_kernel]
pub fn resnet101(x: *const u16, w: *const u16, residual: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut res = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(res, residual, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(res, work, res, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, res, total);
    }
}

/// ResNet basic block: norm + relu + residual add
#[tile_std::tile_kernel]
pub fn resnet_basic_block(x: *const u16, w: *const u16, residual: *const f32, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut res = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_buf_load_f32(res, residual, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(work, work, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(res, work, res, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, res, total);
    }
}

/// VGG-16: FC + ReLU + bias
#[tile_std::tile_kernel]
pub fn vgg16(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// VGG-19: FC + ReLU + bias (deeper variant)
#[tile_std::tile_kernel]
pub fn vgg19(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.01f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// SqueezeNet: squeeze + expand with relu
#[tile_std::tile_kernel]
pub fn squeeze_net(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 0.25f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 4.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// SqueezeNet fire module: squeeze + expand with relu
#[tile_std::tile_kernel]
pub fn squeeze_net_fire_module(input: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 0.25f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, 4.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, buf, n);
    }
}

/// ShuffleNet: channel shuffle + FC + relu
#[tile_std::tile_kernel]
pub fn shufflenet(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// ShuffleNet unit: channel shuffle + FC + relu
#[tile_std::tile_kernel]
pub fn shufflenet_unit(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let buf = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(buf, buf, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_adds_f32(buf, buf, 0.1f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, buf, total);
    }
}

/// GoogLeNet inception module: parallel paths merged (add + relu)
#[tile_std::tile_kernel]
pub fn googlenet_inception_module(a: *const f32, b: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bb, ba, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(bb, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}

/// GoogLeNet inception V1: parallel paths merged (add + relu)
#[tile_std::tile_kernel]
pub fn googlenet_inception_v1(a: *const f32, b: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let ba = tile_std::__tile_buf_alloc(n);
        let bb = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(ba, a, n);
        tile_std::__tile_buf_load_f32(bb, b, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_add_f32(bb, ba, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::relu_f32(bb, bb, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bb, n);
    }
}

/// Swin MLP: window attention with softmax + scale
#[tile_std::tile_kernel]
pub fn swin_mlp(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Swin Transformer V2: window attention with softmax + scale
#[tile_std::tile_kernel]
pub fn swintransformer_v2(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// Mamba return final state: sigmoid gate * linear
#[tile_std::tile_kernel]
pub fn mamba_return_final_state(x: *const f32, gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bg, bg, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bg, bx, bg, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bg, n);
    }
}

/// Mamba return y: sigmoid gate * linear
#[tile_std::tile_kernel]
pub fn mamba_return_y(x: *const f32, gate: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let bx = tile_std::__tile_buf_alloc(n);
        let bg = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(bx, x, n);
        tile_std::__tile_buf_load_f32(bg, gate, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::sigmoid_f32(bg, bg, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_v_mul_f32(bg, bx, bg, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, bg, n);
    }
}

/// Convolutional Vision Transformer: norm + matmul + gelu
#[tile_std::tile_kernel]
pub fn convolutional_vision_transformer(x: *const u16, w: *const u16, out: *mut f32, dims: *const u32) {
    unsafe {
        let m = *dims;
        let k = *dims.wrapping_add(1);
        let n = *dims.wrapping_add(2);
        tile_std::kernel_ops::matmul_f16(out, x, w, m, k, n);
        tile_std::__tile_pipe_barrier();
        let total = m * n;
        let mut buf = tile_std::__tile_buf_alloc(total);
        let mut tmp = tile_std::__tile_buf_alloc(total);
        let mut work = tile_std::__tile_buf_alloc(total);
        let mut extra = tile_std::__tile_buf_alloc(total);
        tile_std::__tile_buf_load_f32(buf, out as *const f32, total);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::layernorm_f32(&mut work, &buf, &mut extra, total, 1e-5f32);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::gelu_f32(&mut tmp, &work, &mut extra, total);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(out, tmp, total);
    }
}

/// NetVLAD without ghost clusters: scale + softmax + sum
#[tile_std::tile_kernel]
pub fn net_vlad_no_ghost_clusters(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// NetVLAD with ghost clusters: scale + softmax + sum
#[tile_std::tile_kernel]
pub fn net_vlad_with_ghost_clusters(input: *const f32, output: *mut f32, config: *const f32, len: *const u32) {
    unsafe {
        let n = *len;
        let scale = *config;
        let mut buf = tile_std::__tile_buf_alloc(n);
        let mut work = tile_std::__tile_buf_alloc(n);
        let mut extra = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_muls_f32(buf, buf, scale, n);
        tile_std::__tile_pipe_barrier();
        tile_std::kernel_ops::softmax_f32(&mut work, &mut buf, &mut extra, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, work, n);
    }
}

/// MobileNetV2 inverted residual: expand (scale) + relu6 + project (scale) + residual add
#[tile_std::tile_kernel]
pub fn mobilenetv2_inverted(input: *const f32, residual: *const f32, output: *mut f32, len: *const u32) {
    unsafe {
        let n = *len;
        let buf = tile_std::__tile_buf_alloc(n);
        let res = tile_std::__tile_buf_alloc(n);
        tile_std::__tile_buf_load_f32(buf, input, n);
        tile_std::__tile_buf_load_f32(res, residual, n);
        tile_std::__tile_pipe_barrier();
        // expand
        tile_std::__tile_muls_f32(buf, buf, 6.0f32, n);
        tile_std::__tile_pipe_barrier();
        // relu6
        tile_std::__tile_maxs_f32(buf, buf, 0.0f32, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_mins_f32(buf, buf, 6.0f32, n);
        tile_std::__tile_pipe_barrier();
        // project back
        tile_std::__tile_muls_f32(buf, buf, 0.1667f32, n);
        tile_std::__tile_pipe_barrier();
        // residual — res dead after
        tile_std::__tile_v_add_f32(res, buf, res, n);
        tile_std::__tile_pipe_barrier();
        tile_std::__tile_buf_store_f32(output, res, n);
    }
}
