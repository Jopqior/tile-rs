# Throwaway: real PTO device code on camodel

Issue: [验证 camodel 的 f32 add 无卡正确性 CI](https://github.com/Jopqior/tile-rs/issues/59).

Question: can a no-NPU Ubuntu 22.04 host compile and numerically execute both
handwritten PTO C++ and `.pto → PTOAS → C++` using CANN's A2 camodel?
This is not a tile-rs backend, CPU-SIM test, benchmark, or compile-only check.
Keep it on `prototype/59-camodel`, out of main.

## Run

Use a disposable Ubuntu 22.04 x86_64 container for local work; never install this
Toolkit globally on a development machine. Installation/use must be authorized.
Install the Ubuntu dependencies listed in the workflow, then inside that container:

```sh
export BUILD_DIR=/scratch/add INSTALL_ROOT=/scratch/ascend
bash prototypes/pto-camodel/install.sh
PROGRAM_MODE=handwritten bash prototypes/pto-camodel/run.sh
PROGRAM_MODE=generated bash prototypes/pto-camodel/run.sh
PROGRAM_MODE=generated INJECT_ERROR=1 bash prototypes/pto-camodel/run.sh # must exit 1
```

Allow at least 20 GiB free before installing. `TOOLKIT_PACKAGE` optionally selects
an already downloaded package; it is still SHA256-checked. Neither SDK nor model
libraries are redistributed or cached by the workflow. Installation is full
Toolkit, without driver/firmware, to reduce missing-component variables.

Pins: CANN Toolkit 9.1.0 (north-4 OBS endpoint, whole-package SHA256 in
`install.sh`); PTO ISA `82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19`; PTOAS 0.65
CPython 3.10 manylinux x86_64 wheel (mandatory SHA256 in `run.sh`). Ubuntu system
dependencies use the 22.04 distribution versions, recorded with `dpkg-query`;
they are not claimed bit-for-bit reproducible across runner image updates.

## Contract

Both modes share one ACL host, one kernel launch wrapper, and one independent
Python standard-library oracle. The generated C++ is included unchanged, not
patched. Bisheng compiles `dav-c220-vec`; installation-manifest symlinks must
resolve `Ascend910B1` to `dav_2201`. No `__CPU_SIM` is defined.

The kernel loads two f32 [1,256] tiles, adds, and stores, with device pipeline
synchronization. Inputs and sums are exactly representable dyadic values;
comparison is exact for all 256 elements and rejects non-finite/missing output.
Device output starts with NaNs. Every ACL call is checked, including launch
last-error and stream synchronization. Missing software, compilation failures,
runtime errors and mismatches are nonzero failures, never green skips.

The host also checks process-global resolution of five critical `rt*` symbols
against `libruntime_camodel.so`. **CANN 9.1 ACL transitively loads hardware
runtime/hal shared libraries too**; this is not proof that these libraries are
absent. Private local `LD_DEBUG=bindings` inspection confirmed ACL memcpy and
kernel launch resolve to camodel, as do device selection, allocation and sync.
There is no driver installation or exposed NPU device.

`INJECT_ERROR=1` modifies the returned numeric output[0] by +1 before comparison.
It is a negative test of the oracle/failure propagation, not a different device
kernel. The workflow initially has `INJECT_ERROR: '0'`; change it to `'1'` and
push for a deliberately red matrix, then restore to `'0'`. It only responds to
its prototype branch and its own paths, with fail-fast disabled.

## Evidence and boundary

Completed in an isolated Ubuntu 22.04 Docker container (not GitHub Actions):

- Full Toolkit SHA check and install, both root and unprivileged-user installs.
- Old PTO ISA pin compiled successfully with 9.1; SDK PTO headers not substituted.
- Both complete `run.sh` entry points: 256/256 exact PASS.
- Both entry points with corrupted output: exit 1, index 0 actual 1 vs reference 0.
- Both restored to normal: PASS; unprivileged generated run also PASS.
- Actual Bisheng banner: clang 15.0.5, clang/flang build `5c68a1cb1231`,
  dated `2026-07-30T20:53:21+08:00`.

GitHub-hosted Ubuntu 22.04 subsequently completed both modes:
[normal PASS](https://github.com/Jopqior/tile-rs/actions/runs/35761288167),
[corrupted-output failure](https://github.com/Jopqior/tile-rs/actions/runs/35761853455),
and [restored PASS](https://github.com/Jopqior/tile-rs/actions/runs/35762428063).
Both negative jobs failed specifically at index 0 comparison with exit 1.
See the [experiment report and preserved evidence](../../docs/research/camodel-prototype.md).
This only covers the fixed add/input contract, not tile-rs integration, other
instructions, general floating-point behavior, hardware correctness or performance.

## Publication boundary

Artifact upload is an explicit allowlist: our IR, untouched generated C++, our
wrapper/host/oracle, input/output text, comparison result and necessary inventory.
No SDK files, executable, model trace, performance data, disassembly, compiler
internal logs, installation logs or dynamic-linker logs are uploaded. Tool/runtime
logs stay private in the scratch directory. Failure reports deliberately expose
only the stage/exit status, not arbitrary vendor diagnostics. Do not upload the
whole scratch directory when debugging a failure.
