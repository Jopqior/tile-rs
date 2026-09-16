//! Delivered correctness cases for issue #33: the five elementwise-family
//! paths `add`, `sub`, `mul`, standalone `exp`, and the two-step combination
//! `(a+b)*c`.
//!
//! Every case is defined by a declarative [`CaseSpec`] and materialized to a
//! concrete [`Case`] against the executing device. Boundary sizes are
//! expressed relative to the device's SIMD width and maximum threadgroup size
//! so they follow the actual device instead of a hard-coded 32/256 contract.
//!
//! Fixed small cases carry literal hand anchors. Larger cases use a fixed-seed
//! structured generator; their anchor is an independent Rust evaluation of the
//! declared computation (not PyTorch, not the emitter), and PyTorch CPU is the
//! reference that the GPU result is compared against.

/// Sentinel written into the output (including padding) before dispatch so a
/// skipped write is visible.
pub const OUTPUT_SENTINEL: f32 = 123456.75;

/// Extra output elements the kernel must not touch.
pub const PRESERVE_PAD: usize = 2;

/// Fixed seed for the deterministic structured input generator.
pub const SEED: u64 = 0x5EED_1234_ABCD_0001;

/// Device capability the entry discovered from the Metal pipeline/device.
/// Boundary sizes are instantiated from this, not from a universal constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCaps {
    /// `thread_execution_width` of the compiled probe pipeline (SIMD width).
    pub simd_width: usize,
    /// Device maximum threads per threadgroup for a 1-D dispatch.
    pub threadgroup_max: usize,
}

/// Declared computation of a case. This names the meaning; it is not derived
/// from the emitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefOp {
    Add,
    Sub,
    Mul,
    Exp,
    AddMul,
}

impl RefOp {
    pub fn arity(self) -> usize {
        match self {
            RefOp::Add | RefOp::Sub | RefOp::Mul => 2,
            RefOp::Exp => 1,
            RefOp::AddMul => 3,
        }
    }

    /// Stable id fragment and PyTorch op tag.
    pub fn tag(self) -> &'static str {
        match self {
            RefOp::Add => "add",
            RefOp::Sub => "sub",
            RefOp::Mul => "mul",
            RefOp::Exp => "exp",
            RefOp::AddMul => "add_mul",
        }
    }

    pub fn meaning(self) -> &'static str {
        match self {
            RefOp::Add => "elementwise f32 add: out[i] = a[i] + b[i]",
            RefOp::Sub => "elementwise f32 subtract: out[i] = a[i] - b[i]",
            RefOp::Mul => "elementwise f32 multiply: out[i] = a[i] * b[i]",
            RefOp::Exp => "elementwise f32 exp: out[i] = exp(x[i])",
            RefOp::AddMul => "two-step f32 combination: out[i] = (a[i] + b[i]) * c[i]",
        }
    }

    pub fn binding_note(self) -> &'static str {
        match self {
            RefOp::Add | RefOp::Sub | RefOp::Mul => {
                "p0=a p1=b p2=out buffer(3)=num_elements"
            }
            RefOp::Exp => "p0=x p1=out buffer(2)=num_elements",
            RefOp::AddMul => "p0=a p1=b p2=c p3=out buffer(4)=num_elements",
        }
    }

    /// Independent evaluation of the declared computation. Used as the anchor
    /// the PyTorch reference is checked against; never used as the GPU
    /// reference itself.
    pub fn apply(self, inputs: &[Vec<f32>]) -> Vec<f32> {
        let n = inputs[0].len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let v = match self {
                RefOp::Add => inputs[0][i] + inputs[1][i],
                RefOp::Sub => inputs[0][i] - inputs[1][i],
                RefOp::Mul => inputs[0][i] * inputs[1][i],
                RefOp::Exp => inputs[0][i].exp(),
                RefOp::AddMul => (inputs[0][i] + inputs[1][i]) * inputs[2][i],
            };
            out.push(v);
        }
        out
    }
}

/// How a case's length is instantiated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizePlan {
    /// Device-independent length.
    Fixed(usize),
    /// `simd_width + delta`.
    SimdDelta(i64),
    /// `threadgroup_max + delta`.
    TgDelta(i64),
    /// `groups * threadgroup_max + tail`.
    GroupsTail { groups: usize, tail: usize },
}

impl SizePlan {
    pub fn needs_device(self) -> bool {
        !matches!(self, SizePlan::Fixed(_))
    }

    pub fn resolve(self, caps: &DeviceCaps) -> usize {
        let n = match self {
            SizePlan::Fixed(n) => n as i64,
            SizePlan::SimdDelta(d) => caps.simd_width as i64 + d,
            SizePlan::TgDelta(d) => caps.threadgroup_max as i64 + d,
            SizePlan::GroupsTail { groups, tail } => {
                (groups * caps.threadgroup_max + tail) as i64
            }
        };
        n.max(1) as usize
    }

    pub fn label(self) -> String {
        match self {
            SizePlan::Fixed(n) => format!("fixed-{n}"),
            SizePlan::SimdDelta(d) => format!("simd{d:+}"),
            SizePlan::TgDelta(d) => format!("threadgroup{d:+}"),
            SizePlan::GroupsTail { groups, tail } => {
                format!("{groups}xthreadgroup+{tail}")
            }
        }
    }
}

/// Declarative case definition, independent of the device.
#[derive(Clone, Debug)]
pub struct CaseSpec {
    pub id: String,
    pub op: RefOp,
    pub size: SizePlan,
    /// True when literal hand anchors exist for this case.
    pub hand: bool,
}

fn spec(id: &str, op: RefOp, size: SizePlan, hand: bool) -> CaseSpec {
    CaseSpec {
        id: id.to_string(),
        op,
        size,
        hand,
    }
}

/// Concrete case instantiated against a device.
#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub op: RefOp,
    pub mlir: String,
    pub kernel_name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub layout: &'static str,
    pub inputs: Vec<Vec<f32>>,
    /// Anchor the PyTorch reference is checked against. For hand cases this is
    /// the literal; otherwise an independent Rust evaluation of `op`.
    pub expected_anchor: Vec<f32>,
    pub hand_anchor: bool,
    pub preserve_pad: usize,
    pub size_note: String,
}

impl Case {
    pub fn n(&self) -> usize {
        self.shape[0]
    }
}

/// Delivered case list for this entry. Stable ids; sizes resolve per device.
pub fn delivered_specs() -> Vec<CaseSpec> {
    let mut v = vec![
        // Fixed small cases with literal hand anchors, one per path.
        spec("add_f32_small", RefOp::Add, SizePlan::Fixed(8), true),
        spec("sub_f32_small", RefOp::Sub, SizePlan::Fixed(8), true),
        spec("mul_f32_small", RefOp::Mul, SizePlan::Fixed(8), true),
        spec("exp_f32_small", RefOp::Exp, SizePlan::Fixed(8), true),
        spec("add_mul_f32_small", RefOp::AddMul, SizePlan::Fixed(8), true),
    ];

    // add/sub/mul: single element, SIMD-width boundaries, threadgroup
    // boundaries, and multiple groups with a tail.
    for op in [RefOp::Add, RefOp::Sub, RefOp::Mul] {
        let p = op.tag();
        v.push(spec(
            &format!("{p}_f32_single"),
            op,
            SizePlan::Fixed(1),
            false,
        ));
        v.push(spec(
            &format!("{p}_f32_vec_minus_1"),
            op,
            SizePlan::SimdDelta(-1),
            false,
        ));
        v.push(spec(
            &format!("{p}_f32_vec"),
            op,
            SizePlan::SimdDelta(0),
            false,
        ));
        v.push(spec(
            &format!("{p}_f32_vec_plus_1"),
            op,
            SizePlan::SimdDelta(1),
            false,
        ));
        v.push(spec(
            &format!("{p}_f32_threadgroup_minus_1"),
            op,
            SizePlan::TgDelta(-1),
            false,
        ));
        v.push(spec(
            &format!("{p}_f32_threadgroup_plus_1"),
            op,
            SizePlan::TgDelta(1),
            false,
        ));
        v.push(spec(
            &format!("{p}_f32_multi_group_tail"),
            op,
            SizePlan::GroupsTail { groups: 2, tail: 1 },
            false,
        ));
    }

    // exp: single element and multiple groups with a tail.
    v.push(spec("exp_f32_single", RefOp::Exp, SizePlan::Fixed(1), false));
    v.push(spec(
        "exp_f32_multi_group_tail",
        RefOp::Exp,
        SizePlan::GroupsTail { groups: 2, tail: 1 },
        false,
    ));

    // Combination: cross-group and tail, verifying the actual two-step order.
    v.push(spec(
        "add_mul_f32_multi_group_tail",
        RefOp::AddMul,
        SizePlan::GroupsTail { groups: 2, tail: 1 },
        false,
    ));

    v
}

pub fn delivered_ids() -> Vec<String> {
    delivered_specs().into_iter().map(|s| s.id).collect()
}

pub fn spec_by_id(id: &str) -> Option<CaseSpec> {
    delivered_specs().into_iter().find(|s| s.id == id)
}

/// Materialize a spec against the discovered device capability. Device-relative
/// sizes require `caps`; fixed cases materialize without a device so local
/// self-check and `--case` debugging still work off-device.
pub fn materialize(spec: &CaseSpec, caps: Option<&DeviceCaps>) -> Result<Case, String> {
    let n = if spec.size.needs_device() {
        let caps = caps.ok_or_else(|| {
            format!(
                "case {} size {} needs a Metal device to instantiate",
                spec.id,
                spec.size.label()
            )
        })?;
        spec.size.resolve(caps)
    } else {
        spec.size.resolve(&DeviceCaps {
            simd_width: 1,
            threadgroup_max: 1,
        })
    };

    let kernel_name = format!("kc_{}", spec.id);
    let mlir = mlir_for(spec.op, &kernel_name, n);

    let (inputs, expected_anchor, hand_anchor) = if spec.hand {
        let inputs = hand_inputs(spec.op);
        let expected = hand_expected(spec.op);
        (inputs, expected, true)
    } else {
        let inputs = gen_inputs(spec.op, n);
        let expected = spec.op.apply(&inputs);
        (inputs, expected, false)
    };

    Ok(Case {
        id: spec.id.clone(),
        op: spec.op,
        mlir,
        kernel_name,
        dtype: "f32",
        shape: vec![n],
        layout: "contiguous-1d",
        inputs,
        expected_anchor,
        hand_anchor,
        preserve_pad: PRESERVE_PAD,
        size_note: format!("{} (n={n})", spec.size.label()),
    })
}

/// MLIR for a case, with the length constant matching the runtime
/// `num_elements`. Binary/unary ops use the emitter's canned single-op forms;
/// the combination uses the elementwise chain form so the two-step order is
/// carried by the module.
pub fn mlir_for(op: RefOp, kernel_name: &str, n: usize) -> String {
    let body = match op {
        RefOp::Add | RefOp::Sub | RefOp::Mul => format!(
            r#"    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %y = llvm.call @__tile_{op}_f32(%a, %a, %b, %r, %c) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()"#,
            op = op.tag()
        ),
        RefOp::Exp => r#"    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %y = llvm.call @__tile_exp_f32(%a, %a, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()"#
            .to_string(),
        RefOp::AddMul => r#"    %a = llvm.call @__tile_load_f32(%arg0, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %d = llvm.call @__tile_load_f32(%arg2, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %s = llvm.call @__tile_add_f32(%a, %b, %r, %c) : (i32, i32, i32, i32) -> i32
    %y = llvm.call @__tile_mul_f32(%s, %d, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %y, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()"#
            .to_string(),
    };

    let arg_count = match op {
        RefOp::Add | RefOp::Sub | RefOp::Mul => 3,
        RefOp::Exp => 2,
        RefOp::AddMul => 4,
    };
    let args: Vec<String> = (0..arg_count)
        .map(|i| format!("%arg{i}: !llvm.ptr<1>"))
        .collect();

    format!(
        r#"
module {{
  llvm.func @{kernel_name}({args}) attributes {{hacc.entry}} {{
    ^bb0:
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant({n} : i32) : i32
{body}
    llvm.return
  }}
}}
"#,
        args = args.join(", "),
    )
}

// ---------------------------------------------------------------------------
// Deterministic structured inputs
// ---------------------------------------------------------------------------

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = ((*state >> 33) & 0xFFFF) as f32;
    // Multiples of 1/8192 in [-4, 4): finite, normal, no subnormals.
    bits / 8192.0 - 4.0
}

/// Fill with structure (zeros, mixed signs, exact values) plus seeded noise,
/// and pin the head and tail so a tail-only miss is visible.
fn structured(v: &mut [f32], state: &mut u64) {
    for i in 0..v.len() {
        v[i] = match i % 7 {
            0 => 0.0,
            1 => 0.5,
            2 => -0.25,
            3 => 3.0,
            4 => -2.0,
            _ => lcg(state),
        };
    }
    let l = v.len();
    if l >= 1 {
        v[0] = 1.5;
    }
    if l >= 2 {
        v[l - 2] = 0.75;
    }
    if l >= 1 {
        v[l - 1] = -1.5;
    }
}

fn gen_inputs(op: RefOp, n: usize) -> Vec<Vec<f32>> {
    let mut inputs: Vec<Vec<f32>> = Vec::new();
    for k in 0..op.arity() {
        let mut state = SEED ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(k as u64 + 1);
        let mut v = vec![0.0f32; n];
        structured(&mut v, &mut state);
        match (op, k) {
            (RefOp::Add, 1) => {
                // Exact cancellation against operand 0 at a fixed stride.
                for i in (0..n).step_by(5) {
                    v[i] = -inputs[0][i];
                }
            }
            (RefOp::Sub, 1) => {
                // Equal at a stride gives an exact zero; negated gives a-b = 2a.
                for i in (0..n).step_by(5) {
                    v[i] = inputs[0][i];
                }
                for i in (1..n).step_by(5) {
                    v[i] = -inputs[0][i];
                }
            }
            (RefOp::Mul, 1) => {
                for i in (0..n).step_by(5) {
                    v[i] = 1.0;
                }
                for i in (1..n).step_by(5) {
                    v[i] = 0.0;
                }
            }
            (RefOp::Exp, 0) => {
                for x in v.iter_mut() {
                    *x = (*x * 2.0).clamp(-8.0, 8.0);
                }
            }
            (RefOp::AddMul, 2) => {
                for i in (0..n).step_by(5) {
                    v[i] = 1.0;
                }
                for i in (1..n).step_by(5) {
                    v[i] = -1.0;
                }
            }
            _ => {}
        }
        inputs.push(v);
    }
    inputs
}

// ---------------------------------------------------------------------------
// Literal hand anchors
// ---------------------------------------------------------------------------

/// Distinct finite f32 values with zeros, mixed signs, exact cancellation,
/// and repeated values.
pub const HAND_A: [f32; 8] = [1.0, 2.5, -3.0, 0.0, 0.5, -1.25, 4.0, -0.5];
pub const HAND_B: [f32; 8] = [0.5, -1.0, 3.0, 2.0, 1.5, 1.25, -2.0, 0.25];

/// Hand-computed from the add definition:
/// 1+0.5=1.5, 2.5-1=1.5, -3+3=0, 0+2=2, 0.5+1.5=2, -1.25+1.25=0, 4-2=2, -0.5+0.25=-0.25.
pub const HAND_ADD: [f32; 8] = [1.5, 1.5, 0.0, 2.0, 2.0, 0.0, 2.0, -0.25];
/// Hand-computed from the subtract definition.
pub const HAND_SUB: [f32; 8] = [0.5, 3.5, -6.0, -2.0, -1.0, -2.5, 6.0, -0.75];
/// Hand-computed from the multiply definition.
pub const HAND_MUL: [f32; 8] = [0.5, -2.5, -9.0, 0.0, 0.75, -1.5625, -8.0, -0.125];

/// Hand-picked exp inputs and independently computed values (standard math,
/// not PyTorch, not the emitter).
pub const HAND_EXP_X: [f32; 8] = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0];
pub const HAND_EXP: [f32; 8] = [
    1.0,
    1.6487212,
    0.60653067,
    2.7182817,
    0.36787945,
    7.389056,
    0.13533528,
    20.085537,
];

/// Combination inputs chosen so `(a+b)*c` differs from `a+(b*c)` at 7 of 8
/// positions, so a wrong association cannot pass.
pub const HAND_COMBO_A: [f32; 8] = [1.0, 2.0, -3.0, 0.5, -0.25, 4.0, -1.5, 0.0];
pub const HAND_COMBO_B: [f32; 8] = [0.5, -1.0, 1.0, 1.5, 0.75, -2.0, 0.5, 2.0];
pub const HAND_COMBO_C: [f32; 8] = [2.0, -3.0, 0.5, -2.0, 4.0, 0.25, -1.0, 5.0];
pub const HAND_COMBO: [f32; 8] = [3.0, -3.0, -1.0, -4.0, 2.0, 0.5, 1.0, 10.0];
/// What a right-associated `a + (b*c)` would produce for the same inputs.
pub const HAND_COMBO_WRONG: [f32; 8] =
    [2.0, 5.0, -2.5, -2.5, 2.75, 3.5, -2.0, 10.0];

fn hand_inputs(op: RefOp) -> Vec<Vec<f32>> {
    match op {
        RefOp::Add | RefOp::Sub | RefOp::Mul => {
            vec![HAND_A.to_vec(), HAND_B.to_vec()]
        }
        RefOp::Exp => vec![HAND_EXP_X.to_vec()],
        RefOp::AddMul => vec![
            HAND_COMBO_A.to_vec(),
            HAND_COMBO_B.to_vec(),
            HAND_COMBO_C.to_vec(),
        ],
    }
}

fn hand_expected(op: RefOp) -> Vec<f32> {
    match op {
        RefOp::Add => HAND_ADD.to_vec(),
        RefOp::Sub => HAND_SUB.to_vec(),
        RefOp::Mul => HAND_MUL.to_vec(),
        RefOp::Exp => HAND_EXP.to_vec(),
        RefOp::AddMul => HAND_COMBO.to_vec(),
    }
}

/// Required input buffers must be present, same length, finite, and match the
/// declared shape. Empty or truncated fixtures are a fail, not a skip.
pub fn required_inputs_present(case: &Case) -> Result<(), String> {
    if case.inputs.len() != case.op.arity() {
        return Err(format!(
            "expected {} input buffers, have {}",
            case.op.arity(),
            case.inputs.len()
        ));
    }
    let n = case.n();
    if n == 0 {
        return Err("case length is zero".into());
    }
    for (i, x) in case.inputs.iter().enumerate() {
        if x.is_empty() {
            return Err(format!("required input buffer {i} is empty"));
        }
        if x.len() != n {
            return Err(format!(
                "input {i} length {} does not match declared length {n}",
                x.len()
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(format!("input {i} contains a non-finite value"));
        }
    }
    if case.expected_anchor.len() != n {
        return Err(format!(
            "anchor length {} does not match declared length {n}",
            case.expected_anchor.len()
        ));
    }
    if !case.expected_anchor.iter().all(|v| v.is_finite()) {
        return Err("anchor contains a non-finite value".into());
    }
    if case.shape.len() != 1 || case.shape[0] != n {
        return Err(format!("declared shape {:?} is not 1-D length {n}", case.shape));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivered_specs_have_unique_ids_and_cover_all_paths() {
        let ids = delivered_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "duplicate case id: {ids:?}");
        for op in [
            RefOp::Add,
            RefOp::Sub,
            RefOp::Mul,
            RefOp::Exp,
            RefOp::AddMul,
        ] {
            assert!(
                ids.iter().any(|id| id.starts_with(op.tag())),
                "no case for {op:?}: {ids:?}"
            );
        }
    }

    #[test]
    fn literal_hand_anchors_match_the_declared_computation() {
        for op in [
            RefOp::Add,
            RefOp::Sub,
            RefOp::Mul,
            RefOp::Exp,
            RefOp::AddMul,
        ] {
            let inputs = hand_inputs(op);
            let derived = op.apply(&inputs);
            let literal = hand_expected(op);
            for (i, (a, b)) in derived.iter().zip(literal.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff <= 1e-5 + 1.3e-6 * b.abs(),
                    "{op:?}[{i}] derived={a} literal={b}"
                );
            }
        }
    }

    #[test]
    fn combination_inputs_distinguish_association_order() {
        let expected = RefOp::AddMul.apply(&hand_inputs(RefOp::AddMul));
        let wrong: Vec<f32> = HAND_COMBO_WRONG.to_vec();
        let differing = expected
            .iter()
            .zip(wrong.iter())
            .filter(|(a, b)| (**a - **b).abs() > 1e-3)
            .count();
        assert!(
            differing >= 4,
            "combo inputs do not expose association: {expected:?} vs {wrong:?}"
        );
        assert_eq!(expected, HAND_COMBO.to_vec());
    }

    #[test]
    fn generated_inputs_are_finite_and_structured_for_every_op() {
        for op in [
            RefOp::Add,
            RefOp::Sub,
            RefOp::Mul,
            RefOp::Exp,
            RefOp::AddMul,
        ] {
            for n in [1usize, 7, 32, 33, 100] {
                let inputs = gen_inputs(op, n);
                assert_eq!(inputs.len(), op.arity());
                for (k, x) in inputs.iter().enumerate() {
                    assert_eq!(x.len(), n);
                    assert!(
                        x.iter().all(|v| v.is_finite()),
                        "{op:?} input {k} not finite"
                    );
                }
                let expected = op.apply(&inputs);
                assert!(expected.iter().all(|v| v.is_finite()), "{op:?} anchor not finite");
                if n >= 8 {
                    assert!(
                        inputs.iter().any(|x| x.iter().any(|v| *v == 0.0)),
                        "{op:?} n={n} has no zero in any input"
                    );
                }
            }
        }
    }

    #[test]
    fn device_relative_sizes_resolve_from_caps_not_constants() {
        let caps = DeviceCaps {
            simd_width: 31,
            threadgroup_max: 257,
        };
        assert_eq!(SizePlan::SimdDelta(-1).resolve(&caps), 30);
        assert_eq!(SizePlan::SimdDelta(0).resolve(&caps), 31);
        assert_eq!(SizePlan::TgDelta(1).resolve(&caps), 258);
        assert_eq!(
            SizePlan::GroupsTail { groups: 2, tail: 1 }.resolve(&caps),
            515
        );
    }

    #[test]
    fn materialize_emits_the_length_constant_and_matching_inputs() {
        let caps = DeviceCaps {
            simd_width: 32,
            threadgroup_max: 64,
        };
        let s = spec("add_f32_threadgroup_plus_1", RefOp::Add, SizePlan::TgDelta(1), false);
        let case = materialize(&s, Some(&caps)).unwrap();
        assert_eq!(case.n(), 65);
        assert!(case.mlir.contains("llvm.mlir.constant(65 : i32)"));
        assert_eq!(case.inputs[0].len(), 65);
        assert_eq!(case.expected_anchor.len(), 65);
        assert!(!case.hand_anchor);
    }

    #[test]
    fn device_relative_size_without_device_is_an_error_not_a_guess() {
        let s = spec("add_f32_vec", RefOp::Add, SizePlan::SimdDelta(0), false);
        assert!(materialize(&s, None).is_err());
    }

    #[test]
    fn hard_coded_hand_case_materializes_off_device() {
        let s = spec("add_f32_small", RefOp::Add, SizePlan::Fixed(8), true);
        let case = materialize(&s, None).unwrap();
        assert_eq!(case.n(), 8);
        assert!(case.hand_anchor);
        assert_eq!(case.expected_anchor, HAND_ADD.to_vec());
    }

    #[test]
    fn missing_or_truncated_inputs_are_rejected() {
        let s = spec("add_f32_small", RefOp::Add, SizePlan::Fixed(8), true);
        let mut case = materialize(&s, None).unwrap();
        assert!(required_inputs_present(&case).is_ok());
        case.inputs[0].clear();
        assert!(required_inputs_present(&case).is_err());
    }
}
