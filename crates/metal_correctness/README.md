# Metal correctness entry

Independent verification of the current-source Metal emitter: MLIR in, MSL from
`convert_mlir_to_msl`, runtime Metal compile, GPU execute and readback, compare
to a PyTorch CPU f32 reference.

This entry currently delivers one case, `add_f32_small`. That is not the #31
first-batch list. Later compute families reuse this same binary.

## What it checks

- `add_f32_small`: elementwise f32 add `out[i] = a[i] + b[i]` on 8 contiguous
  values. Bindings `p0=a`, `p1=b`, `p2=out`, `buffer(3)=num_elements`. Inputs
  and two output pad elements must stay unchanged. Fast-math is off. Compare
  uses rtol=1.3e-6, atol=1e-5 after readback to CPU.

Hand anchors (independent of PyTorch and of the emitter):

```
a = [1.0, 2.5, -3.0, 0.0, 0.5, -1.25, 4.0, -0.5]
b = [0.5, -1.0, 3.0, 2.0, 1.5, 1.25, -2.0, 0.25]
out = [1.5, 1.5, 0.0, 2.0, 2.0, 0.0, 2.0, -0.25]
```

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

Look at the log for `SOURCE_COMMIT`, `CASE id=`, `CASE_INPUT_*`, dispatch, and
`RUN_RESULT`. Failures print the generated MSL (or why it was not generated),
the failing stage, and the mismatch index / actual / expected. Set
`METAL_CORRECTNESS_DUMP_MSL=1` to print generated MSL on every case.

Exit 0 is success of the invoked mode. Exit 1 is fail. Exit 2 is unverified
(missing device or PyTorch). CI uses the default full list; only exit 0 of that
mode is a passing check.

## CI

`.github/workflows/metal-correctness.yml` runs the default command on the free
`macos-15` arm64 runner for every PR, every `main` update, and `workflow_dispatch`.
Manual dispatch is available after this workflow exists on the default branch.
The job title states this is initial add coverage, not #31 complete.
