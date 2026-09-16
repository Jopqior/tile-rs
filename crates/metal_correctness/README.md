# Metal correctness entry

Independent verification of the current-source Metal emitter: MLIR in, MSL from
`convert_mlir_to_msl`, runtime Metal compile, GPU execute and readback, compare
to a PyTorch CPU f32 reference.

This entry delivers the issue #33 elementwise family:

- `add`, `sub`, `mul` — each covers single element, SIMD-width boundaries,
  threadgroup-size boundaries, and multiple groups with a tail.
- `exp` — single element and multiple groups with a tail.
- `add_mul` — the two-step combination `(a+b)*c`, cross-group and tail.

Plus one fixed small hand-anchored case per path. That is not the whole #31
first-batch list; reductions, softmax, matmul, gather, and transpose come later
and reuse this same binary.

## What it checks

Every case declares its computation meaning, MLIR, shape, dtype, layout,
bindings, dispatch, and preserved regions, and is compared per path against the
same-input PyTorch CPU reference. Bindings follow the emitter's layout: for a
binary op `p0=a`, `p1=b`, `p2=out`, `buffer(3)=num_elements`; `exp` uses
`p0=x`, `p1=out`, `buffer(2)`; the combination uses `p0=a`, `p1=b`, `p2=c`,
`p3=out`, `buffer(4)`. All inputs and two output pad elements must stay
unchanged. Fast-math is off. Compare uses rtol=1.3e-6, atol=1e-5 after readback
to CPU, with shape/dtype and finite checks, over all elements (no sampling).

### Sizes follow the device

Boundary sizes are instantiated from the discovered device, not hard-coded:

- SIMD boundaries come from the probe pipeline's `thread_execution_width`.
- Threadgroup boundaries come from `Device::max_threads_per_threadgroup`.
- Dispatch uses the discovered threadgroup size. It is not a fixed 32/256 grid.

Fixed small cases materialize without a device so self-check and `--case`
debugging still work off-device. Device-relative cases report `unverified`
when no device is available instead of guessing a size.

### Inputs and anchors

Fixed small cases carry literal hand anchors. Larger cases use a fixed-seed
(Lcg64) structured generator — zeros, mixed signs, exact cancellation,
identity, magnitude differences, and pinned head/tail elements — and their
anchor is an independent Rust evaluation of the declared computation. The GPU
result is always compared against PyTorch, never against the emitter or the
anchor.

The combination inputs are chosen so `(a+b)*c` differs from `a+(b*c)` at most
positions, so a wrong association cannot pass.

## Re-run at a source version

Needs a Mac with a Metal device and `python3` with CPU PyTorch.

```bash
git checkout <commit>
cd crates/metal_correctness
python3 -c "import torch; print(torch.__version__)"
cargo run --release
```

Full delivered list (what CI runs):

```bash
cargo run --release
```

One case, for local debug. This is not CI-equivalent and must not be used as
the workflow command:

```bash
cargo run --release -- --case add_f32_small
```

Self-check only (comparator injections, no GPU):

```bash
cargo run --release -- --self-check
```

Case ids:

```bash
cargo run --release -- --list
```

Look at the log for `SOURCE_COMMIT`, `DEVICE_CAPS`, `CASE id=`, `CASE_INPUT_*`,
`DISPATCH`, and `RUN_RESULT`. Failures print the generated MSL (or why it was
not generated), the failing stage, and the mismatch index / actual / expected.
Set `METAL_CORRECTNESS_DUMP_MSL=1` to print generated MSL on every case.

Exit 0 is success of the invoked mode. Exit 1 is fail. Exit 2 is unverified
(missing device or PyTorch). CI uses the default full list; only exit 0 of that
mode is a passing check.

## CI

`.github/workflows/metal-correctness.yml` runs the default command on the free
`macos-15` arm64 runner for every PR, every `main` update, and `workflow_dispatch`.
Manual dispatch is available after this workflow exists on the default branch.
The job title states this is issue #33 coverage, not #31 complete.
