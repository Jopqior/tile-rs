# Metal runner probe (issue #9)

One-shot experiment for
[实测免费托管 runner 的最小 Metal 编译与执行链路](https://github.com/Jopqior/tile-rs/issues/9).

Scope: free GitHub-hosted `macos-15` runner, single run, no OS matrix. One custom
f32 vector-add MSL compute kernel (`probe/metal-runtime/src/vec_add.metal`) is
exercised through both consumption paths of the emitted MSL (see
`crates/rustc_codegen_tile/src/mlir_to_msl.rs`):

1. **Runtime path**: `probe/metal-runtime` — standalone Rust crate using
   `metal = 0.33`. Enumerates Metal devices, compiles the kernel from source via
   `Device::new_library_with_source`, dispatches it, reads back the result and
   compares against an independently computed CPU f32 reference. Prints
   `PROBE_RESULT=pass|device_missing|library_compile_failed|...`.
2. **Offline path**: `xcrun metal -c` + `xcrun metallib` on the same source
   (workflow step), artifacts uploaded with the run.

Triggered manually via `workflow_dispatch` on this branch. This branch is not
intended to be merged; it exists to give the probe an immutable commit and an
Actions run link for the ticket record.
