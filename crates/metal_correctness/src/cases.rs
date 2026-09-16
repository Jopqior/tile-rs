//! Delivered correctness cases. Issue #32 ships one small f32 add case.
//! Later compute families append to [`delivered_cases`] without changing the
//! entry's summary / status protocol.

/// Elementwise f32 add: out[i] = a[i] + b[i] for i in 0..N.
///
/// MLIR: `__tile_load_f32` of two contiguous f32 buffers, `__tile_add_f32`,
/// `__tile_store_f32` into a third buffer. Runtime length is `num_elements`
/// at buffer(3); the IR constants record the same length.
///
/// Layout: 1-D contiguous. Dtype: f32.
/// Bindings: p0=a, p1=b, p2=out, buffer(3)=num_elements.
/// Dispatch: gid = row * tcount + tid, write if gid < num_elements.
/// Preserved: inputs a and b; output padding past N.
pub const ADD_F32_SMALL_ID: &str = "add_f32_small";

pub const ADD_F32_SMALL_KERNEL: &str = "add_f32_small";

pub const ADD_F32_SMALL_N: usize = 8;

/// Non-degenerate hand-computed inputs. Every pair is a distinct finite
/// f32 value, including zeros, mixed signs, and an exact cancellation.
pub const ADD_F32_SMALL_A: [f32; 8] = [1.0, 2.5, -3.0, 0.0, 0.5, -1.25, 4.0, -0.5];
pub const ADD_F32_SMALL_B: [f32; 8] = [0.5, -1.0, 3.0, 2.0, 1.5, 1.25, -2.0, 0.25];

/// Independent hand anchors, not derived from the emitter or from PyTorch:
/// 1+0.5=1.5, 2.5-1=1.5, -3+3=0, 0+2=2, 0.5+1.5=2, -1.25+1.25=0, 4-2=2, -0.5+0.25=-0.25.
pub const ADD_F32_SMALL_HAND: [f32; 8] = [1.5, 1.5, 0.0, 2.0, 2.0, 0.0, 2.0, -0.25];

/// Sentinel written into the output (including padding) before dispatch so a
/// skipped write is visible.
pub const OUTPUT_SENTINEL: f32 = 123456.75;

/// Extra output elements the kernel must not touch.
pub const PRESERVE_PAD: usize = 2;

pub const ADD_F32_SMALL_MLIR: &str = r#"
module {
  llvm.func @add_f32_small(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(8 : i32) : i32
    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @__tile_add_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %s, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

#[derive(Clone, Debug)]
pub struct Case {
    pub id: &'static str,
    pub meaning: &'static str,
    pub mlir: &'static str,
    pub kernel_name: &'static str,
    pub dtype: &'static str,
    pub shape: &'static [usize],
    pub layout: &'static str,
    pub a: &'static [f32],
    pub b: &'static [f32],
    pub hand_expected: &'static [f32],
    pub preserve_pad: usize,
}

pub fn add_f32_small() -> Case {
    Case {
        id: ADD_F32_SMALL_ID,
        meaning: "elementwise f32 add: out[i] = a[i] + b[i]",
        mlir: ADD_F32_SMALL_MLIR,
        kernel_name: ADD_F32_SMALL_KERNEL,
        dtype: "f32",
        shape: &[ADD_F32_SMALL_N],
        layout: "contiguous-1d",
        a: &ADD_F32_SMALL_A,
        b: &ADD_F32_SMALL_B,
        hand_expected: &ADD_F32_SMALL_HAND,
        preserve_pad: PRESERVE_PAD,
    }
}

/// Expected execution list. Independent of whether a run actually executed
/// each case. Completeness checking compares this list to observed statuses.
pub fn delivered_cases() -> Vec<Case> {
    vec![add_f32_small()]
}

pub fn delivered_ids() -> Vec<&'static str> {
    delivered_cases().into_iter().map(|c| c.id).collect()
}

pub fn case_by_id(id: &str) -> Option<Case> {
    delivered_cases().into_iter().find(|c| c.id == id)
}

/// Required input buffers must be present, same length, and match the hand
/// anchors. Empty or truncated fixtures are a fail, not a skip.
pub fn required_inputs_present(case: &Case) -> Result<(), String> {
    if case.a.is_empty() || case.b.is_empty() || case.hand_expected.is_empty() {
        return Err("required input buffer is empty".into());
    }
    if case.a.len() != case.b.len() || case.a.len() != case.hand_expected.len() {
        return Err(format!(
            "input length mismatch a={} b={} hand={}",
            case.a.len(),
            case.b.len(),
            case.hand_expected.len()
        ));
    }
    if case.shape.len() != 1 || case.shape[0] != case.a.len() {
        return Err(format!(
            "declared shape {:?} does not match input length {}",
            case.shape,
            case.a.len()
        ));
    }
    Ok(())
}
