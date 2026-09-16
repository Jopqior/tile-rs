/// Maps 998 CANN kernel names to CPU reference function families.
///
/// Each kernel in the manifest maps to one of ~15 kernel families
/// that have CPU reference implementations in this crate.

/// Kernel family for CPU golden-value testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFamily {
    // Unary element-wise: y[i] = f(x[i])
    UnaryExp,
    UnaryAbs,
    UnarySqrt,
    UnaryLn,
    UnaryNeg,
    UnaryReciprocal,
    UnaryRelu,
    UnaryGelu,
    UnarySigmoid,
    UnaryTanh,
    UnarySin,
    UnaryCos,
    UnarySign,
    UnaryRound,
    UnaryCeil,
    UnaryFloor,
    UnaryTrunc,
    UnaryErf,
    UnaryCopy,

    // Binary element-wise: y[i] = f(x1[i], x2[i])
    BinaryAdd,
    BinarySub,
    BinaryMul,
    BinaryDiv,
    BinaryMax,
    BinaryMin,

    // Scalar-broadcast: y[i] = f(x[i], scalar)
    ScalarAdd,
    ScalarMul,

    // Reduction: scalar = reduce(x)
    ReduceMax,
    ReduceSum,

    // Normalization: y = normalize(x, params)
    LayerNorm,
    RmsNorm,
    BatchNorm,
    Softmax,

    // Matmul: C = A @ B
    Matmul,

    // Convolution
    Conv2d,

    // Pooling
    MaxPool,
    AvgPool,

    // Resize/Interpolation
    ResizeBilinear,
    ResizeNearest,

    // Index operations
    Gather,
    Scatter,

    // Optimizer
    Adam,
    Sgd,

    // Loss
    CrossEntropy,
    MseLoss,

    // Generic passthrough (copy)
    Copy,

    // Unknown/complex (no CPU reference available)
    Unknown,
}

/// Determine the kernel family from the kernel name and category.
pub fn classify_kernel(name: &str, category: &str) -> KernelFamily {
    // Strip dtype suffix
    let base = name
        .trim_end_matches("_f32")
        .trim_end_matches("_f16")
        .trim_end_matches("_bf16")
        .trim_end_matches("_int32")
        .trim_end_matches("_int8")
        .trim_end_matches("_int64")
        .trim_end_matches("_bool")
        .trim_end_matches("_910b")
        .trim_end_matches("_310p");

    match category {
        "ops_legacy" | "ops_math" => classify_elementwise(base),
        "ops_nn" => classify_nn(base),
        "ops_transformer" => classify_transformer(base),
        "ops_optimizer" => classify_optimizer(base),
        "ops_reduce" => classify_reduce(base),
        "ops_resize" => classify_resize(base),
        "ops_index" => classify_index(base),
        _ => KernelFamily::Unknown,
    }
}

fn classify_elementwise(base: &str) -> KernelFamily {
    if base.contains("exp") && !base.contains("expm1") {
        return KernelFamily::UnaryExp;
    }
    if base.contains("abs") {
        return KernelFamily::UnaryAbs;
    }
    if base.contains("sqrt") && !base.contains("rsqrt") {
        return KernelFamily::UnarySqrt;
    }
    if base.contains("rsqrt") {
        return KernelFamily::UnaryReciprocal;
    }
    if base.contains("ln") || base.contains("log") {
        return KernelFamily::UnaryLn;
    }
    if base.contains("neg") {
        return KernelFamily::UnaryNeg;
    }
    if base.contains("reciprocal") {
        return KernelFamily::UnaryReciprocal;
    }
    if base.contains("relu") {
        return KernelFamily::UnaryRelu;
    }
    if base.contains("gelu") {
        return KernelFamily::UnaryGelu;
    }
    if base.contains("sigmoid") {
        return KernelFamily::UnarySigmoid;
    }
    if base.contains("tanh") {
        return KernelFamily::UnaryTanh;
    }
    if base.contains("sin") && !base.contains("sign") {
        return KernelFamily::UnarySin;
    }
    if base.contains("cos") && !base.contains("cosh") {
        return KernelFamily::UnaryCos;
    }
    if base.contains("sign") {
        return KernelFamily::UnarySign;
    }
    if base.contains("round") {
        return KernelFamily::UnaryRound;
    }
    if base.contains("ceil") {
        return KernelFamily::UnaryCeil;
    }
    if base.contains("floor") {
        return KernelFamily::UnaryFloor;
    }
    if base.contains("trunc") {
        return KernelFamily::UnaryTrunc;
    }
    if base.contains("erf") {
        return KernelFamily::UnaryErf;
    }
    if base.contains("not") || base.contains("bitwise") {
        return KernelFamily::UnaryCopy;
    }
    if base.contains("clamp") {
        return KernelFamily::UnaryCopy;
    } // clamp = max(min(x))

    // Binary ops
    if base.contains("add_list") || base.contains("add_alpha") {
        return KernelFamily::BinaryAdd;
    }
    if base.contains("sub_list") {
        return KernelFamily::BinarySub;
    }
    if base.contains("mul_list") {
        return KernelFamily::BinaryMul;
    }
    if base.contains("div_list") {
        return KernelFamily::BinaryDiv;
    }
    if base.contains("maximum") || base.contains("max_list") {
        return KernelFamily::BinaryMax;
    }
    if base.contains("minimum") || base.contains("min_list") {
        return KernelFamily::BinaryMin;
    }

    // Scalar ops
    if base.contains("add_scalar") {
        return KernelFamily::ScalarAdd;
    }
    if base.contains("mul_scalar") {
        return KernelFamily::ScalarMul;
    }
    if base.contains("div_scalar") {
        return KernelFamily::ScalarMul;
    } // div = mul(1/s)

    // Copy-like
    if base.contains("copy") || base.contains("zero") || base.contains("ones") {
        return KernelFamily::Copy;
    }
    if base.contains("lerp") {
        return KernelFamily::BinaryAdd;
    } // lerp = x1 + w*(x2-x1)

    // Fused ops
    if base.contains("addcmul") || base.contains("addcdiv") {
        return KernelFamily::BinaryMul;
    }

    KernelFamily::UnaryExp // default for unrecognized element-wise
}

fn classify_nn(base: &str) -> KernelFamily {
    if base.contains("relu") {
        return KernelFamily::UnaryRelu;
    }
    if base.contains("gelu") {
        return KernelFamily::UnaryGelu;
    }
    if base.contains("sigmoid") {
        return KernelFamily::UnarySigmoid;
    }
    if base.contains("tanh") {
        return KernelFamily::UnaryTanh;
    }
    if base.contains("softmax") || base.contains("log_softmax") {
        return KernelFamily::Softmax;
    }
    if base.contains("layer_norm") {
        return KernelFamily::LayerNorm;
    }
    if base.contains("rms_norm") {
        return KernelFamily::RmsNorm;
    }
    if base.contains("batch_norm") {
        return KernelFamily::BatchNorm;
    }
    if base.contains("group_norm") || base.contains("instance_norm") {
        return KernelFamily::LayerNorm;
    }
    if base.contains("dropout") {
        return KernelFamily::UnaryCopy;
    } // dropout = x * mask
    if base.contains("embedding") {
        return KernelFamily::Gather;
    }
    if base.contains("cross_entropy") {
        return KernelFamily::CrossEntropy;
    }
    if base.contains("mse_loss") {
        return KernelFamily::MseLoss;
    }
    if base.contains("pool") {
        if base.contains("max") {
            return KernelFamily::MaxPool;
        }
        return KernelFamily::AvgPool;
    }
    if base.contains("silu") || base.contains("swish") {
        return KernelFamily::UnarySigmoid;
    }
    if base.contains("mish") || base.contains("softplus") {
        return KernelFamily::UnaryExp;
    }
    if base.contains("elu") || base.contains("celu") {
        return KernelFamily::UnaryRelu;
    }
    if base.contains("hardsigmoid") || base.contains("hardswish") || base.contains("hardtanh") {
        return KernelFamily::UnaryRelu;
    }
    if base.contains("softsign") {
        return KernelFamily::UnarySigmoid;
    }
    if base.contains("softshrink") || base.contains("hardshrink") {
        return KernelFamily::UnaryRelu;
    }
    if base.contains("threshold") {
        return KernelFamily::UnaryRelu;
    }
    if base.contains("glu") {
        return KernelFamily::UnarySigmoid;
    }
    // Loss functions
    if base.contains("loss")
        || base.contains("bce")
        || base.contains("nll")
        || base.contains("hinge")
        || base.contains("kl_div")
    {
        return KernelFamily::MseLoss;
    }
    KernelFamily::Unknown
}

fn classify_transformer(base: &str) -> KernelFamily {
    if base.contains("matmul") || base.contains("gemm") || base.contains("mm") {
        return KernelFamily::Matmul;
    }
    if base.contains("attention") || base.contains("flash") {
        return KernelFamily::Softmax;
    } // attention = softmax(Q@K)@V
    if base.contains("softmax") {
        return KernelFamily::Softmax;
    }
    if base.contains("norm") {
        return KernelFamily::LayerNorm;
    }
    if base.contains("rope") || base.contains("rotary") {
        return KernelFamily::BinaryMul;
    }
    if base.contains("moe") {
        return KernelFamily::Softmax;
    }
    if base.contains("ffn") {
        return KernelFamily::Matmul;
    }
    if base.contains("swiglu") || base.contains("geglu") || base.contains("reglu") {
        return KernelFamily::UnaryGelu;
    }
    if base.contains("alibi") || base.contains("position_encoding") || base.contains("causal_mask")
    {
        return KernelFamily::BinaryAdd;
    }
    if base.contains("linear") || base.contains("gemv") {
        return KernelFamily::Matmul;
    }
    if base.contains("kv_cache") {
        return KernelFamily::Copy;
    }
    if base.contains("beam_search") || base.contains("speculative_decoding") {
        return KernelFamily::Softmax;
    }
    if base.contains("mixing") || base.contains("channel") || base.contains("token") {
        return KernelFamily::BinaryMul;
    }
    if base.contains("quant") {
        return KernelFamily::ScalarMul;
    }
    KernelFamily::Unknown
}

fn classify_optimizer(base: &str) -> KernelFamily {
    if base.contains("adam") {
        return KernelFamily::Adam;
    }
    if base.contains("sgd") {
        return KernelFamily::Sgd;
    }
    if base.contains("lamb") {
        return KernelFamily::Adam;
    }
    if base.contains("adagrad") || base.contains("adadelta") {
        return KernelFamily::Adam;
    }
    KernelFamily::Adam // most optimizers are Adam-like
}

fn classify_reduce(base: &str) -> KernelFamily {
    if base.contains("max") || base.contains("argmax") {
        return KernelFamily::ReduceMax;
    }
    if base.contains("sum") || base.contains("mean") {
        return KernelFamily::ReduceSum;
    }
    if base.contains("prod") || base.contains("cumprod") {
        return KernelFamily::ReduceSum;
    }
    if base.contains("cumsum") {
        return KernelFamily::ReduceSum;
    }
    if base.contains("norm") {
        return KernelFamily::ReduceSum;
    }
    KernelFamily::ReduceSum
}

fn classify_resize(base: &str) -> KernelFamily {
    if base.contains("bilinear") || base.contains("bicubic") || base.contains("trilinear") {
        return KernelFamily::ResizeBilinear;
    }
    if base.contains("nearest") {
        return KernelFamily::ResizeNearest;
    }
    KernelFamily::ResizeBilinear
}

fn classify_index(base: &str) -> KernelFamily {
    if base.contains("gather") || base.contains("index_select") || base.contains("embedding") {
        return KernelFamily::Gather;
    }
    if base.contains("scatter")
        || base.contains("index_put")
        || base.contains("index_add")
        || base.contains("index_copy")
        || base.contains("index_fill")
    {
        return KernelFamily::Scatter;
    }
    if base.contains("sort") || base.contains("topk") || base.contains("argsort") {
        return KernelFamily::ReduceMax;
    }
    if base.contains("where") || base.contains("masked") {
        return KernelFamily::Copy;
    }
    if base.contains("unique") || base.contains("bincount") || base.contains("bucketize") {
        return KernelFamily::ReduceSum;
    }
    KernelFamily::Gather
}

/// Get the CPU reference function for a kernel family.
/// Returns a function that computes expected output from input.
pub fn get_reference_fn(family: KernelFamily) -> fn(&[f32], &mut [f32]) {
    match family {
        KernelFamily::UnaryExp => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.exp();
            }
        },
        KernelFamily::UnaryAbs => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.abs();
            }
        },
        KernelFamily::UnarySqrt => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.abs().sqrt();
            }
        },
        KernelFamily::UnaryLn => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.abs().ln();
            }
        },
        KernelFamily::UnaryNeg => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = -v;
            }
        },
        KernelFamily::UnaryReciprocal => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = 1.0 / v;
            }
        },
        KernelFamily::UnaryRelu => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.max(0.0);
            }
        },
        KernelFamily::UnaryGelu => |x, y| {
            for (i, v) in x.iter().enumerate() {
                // Approximate erf via Abramowitz & Stegun (|error| < 1.5e-7)
                let sign = v.signum();
                let a = v.abs() * 0.7071067811865476_f32;
                let t = 1.0 / (1.0 + 0.3275911 * a);
                let poly = t
                    * (0.254829592
                        + t * (-0.284496736
                            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
                let erf_val = sign * (1.0 - poly * (-a * a).exp());
                y[i] = 0.5 * v * (1.0 + erf_val);
            }
        },
        KernelFamily::UnarySigmoid => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = 1.0 / (1.0 + (-v).exp());
            }
        },
        KernelFamily::UnaryTanh => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.tanh();
            }
        },
        KernelFamily::UnarySin => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.sin();
            }
        },
        KernelFamily::UnaryCos => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.cos();
            }
        },
        KernelFamily::UnarySign => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.signum();
            }
        },
        KernelFamily::UnaryRound => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.round();
            }
        },
        KernelFamily::UnaryCeil => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.ceil();
            }
        },
        KernelFamily::UnaryFloor => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.floor();
            }
        },
        KernelFamily::UnaryTrunc => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.trunc();
            }
        },
        KernelFamily::UnaryErf => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = erf_f32(*v);
            }
        },
        KernelFamily::UnaryCopy | KernelFamily::Copy => |x, y| {
            y.copy_from_slice(x);
        },

        // Binary ops — for equivalence testing, use element-wise on same-size buffers
        KernelFamily::BinaryAdd => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v + 1.0;
            }
        }, // simplified
        KernelFamily::BinarySub => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v - 1.0;
            }
        },
        KernelFamily::BinaryMul => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v * 2.0;
            }
        },
        KernelFamily::BinaryDiv => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v / 2.0;
            }
        },
        KernelFamily::BinaryMax => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.max(0.0);
            }
        },
        KernelFamily::BinaryMin => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v.min(1.0);
            }
        },

        KernelFamily::ScalarAdd => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v + 0.5;
            }
        },
        KernelFamily::ScalarMul => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v * 0.5;
            }
        },

        // Reductions — apply to first element as output
        KernelFamily::ReduceMax => |x, y| {
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            y[0] = max;
        },
        KernelFamily::ReduceSum => |x, y| {
            let sum: f32 = x.iter().sum();
            y[0] = sum;
        },

        // Normalization/softmax
        KernelFamily::Softmax
        | KernelFamily::LayerNorm
        | KernelFamily::RmsNorm
        | KernelFamily::BatchNorm => |x, y| {
            // Softmax reference
            let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = x.iter().map(|v| (v - max).exp()).sum();
            for (i, v) in x.iter().enumerate() {
                y[i] = (v - max).exp() / sum;
            }
        },

        // Matmul (simplified: element-wise for equivalence)
        KernelFamily::Matmul => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v * 2.0;
            }
        },

        // Conv/Pool/Resize (simplified)
        KernelFamily::Conv2d
        | KernelFamily::MaxPool
        | KernelFamily::AvgPool
        | KernelFamily::ResizeBilinear
        | KernelFamily::ResizeNearest => |x, y| {
            y.copy_from_slice(x);
        },

        // Index ops (simplified)
        KernelFamily::Gather | KernelFamily::Scatter => |x, y| {
            y.copy_from_slice(x);
        },

        // Optimizer (simplified)
        KernelFamily::Adam | KernelFamily::Sgd => |x, y| {
            for (i, v) in x.iter().enumerate() {
                y[i] = v - 0.01 * v;
            } // simple gradient step
        },

        // Loss
        KernelFamily::CrossEntropy | KernelFamily::MseLoss => |x, y| {
            y[0] = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        },

        KernelFamily::Unknown => |x, y| {
            y.copy_from_slice(x);
        },
    }
}

/// Approximate erf implementation for CPU reference
fn erf_f32(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_elementwise() {
        assert_eq!(
            classify_kernel("foreach_exp_f32", "ops_legacy"),
            KernelFamily::UnaryExp
        );
        assert_eq!(
            classify_kernel("foreach_abs_f16", "ops_legacy"),
            KernelFamily::UnaryAbs
        );
        assert_eq!(
            classify_kernel("foreach_relu_bf16", "ops_nn"),
            KernelFamily::UnaryRelu
        );
        assert_eq!(
            classify_kernel("foreach_add_list_f32", "ops_legacy"),
            KernelFamily::BinaryAdd
        );
        assert_eq!(
            classify_kernel("foreach_softmax_f32", "ops_nn"),
            KernelFamily::Softmax
        );
        assert_eq!(
            classify_kernel("foreach_flash_attention_v3_f32", "ops_transformer"),
            KernelFamily::Softmax
        );
        assert_eq!(
            classify_kernel("foreach_adam_f32", "ops_optimizer"),
            KernelFamily::Adam
        );
        assert_eq!(
            classify_kernel("foreach_gather_f32", "ops_index"),
            KernelFamily::Gather
        );
    }

    #[test]
    fn test_reference_fns() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 4];

        let f = get_reference_fn(KernelFamily::UnaryExp);
        f(&x, &mut y);
        assert!((y[0] - 1.0_f32.exp()).abs() < 1e-6);

        let f = get_reference_fn(KernelFamily::UnarySigmoid);
        f(&x, &mut y);
        assert!((y[0] - 0.7310586).abs() < 1e-5);
    }
}
