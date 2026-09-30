#!/usr/bin/env bash
# Throwaway audit harness. No converter code is changed.
set -euo pipefail
harness_dir="$(cd "$(dirname "$0")" && pwd)"
subject="$1"
evidence="$2"
mkdir -p "$evidence"
exec > >(tee "$evidence/harness.log") 2>&1
cd "$subject"
export TILE_HOME="$RUNNER_TEMP/wayfinder64-tile-home"
export TILE_STD_PATH="$subject/crates/tile_std"
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_TARGET_DIR TILE_SIMULATE
{
  git rev-parse HEAD
  git status --porcelain
  uname -a
  sw_vers
  rustup toolchain list
  rustc +stable -Vv
  rustc +nightly-2025-08-04 -Vv
  printf 'ImageOS=%s ImageVersion=%s\n' "${ImageOS-}" "${ImageVersion-}"
} > "$evidence/environment.txt"
cp crates/tile_cli/assets/provision.toml "$evidence/provision.toml"
cargo +stable build --locked --manifest-path crates/tile_cli/Cargo.toml --target-dir "$RUNNER_TEMP/audit-target"
tile="$RUNNER_TEMP/audit-target/debug/tile"
run() {
  name="$1"; shift
  printf '%q ' "$@" > "$evidence/$name.command"; printf '\n' >> "$evidence/$name.command"
  set +e
  "$@" > "$evidence/$name.stdout" 2> "$evidence/$name.stderr"
  rc=$?
  set -e
  echo "$rc" > "$evidence/$name.exit"
  printf '%s exit=%s\n' "$name" "$rc"
}
input="$subject/crates/tile_cli/testdata/forms/softmax.rs"
run auto-install "$tile" "$input" -t pto --cross ascend -O0 -o "$evidence/first.pto.mlir"
# The current CLI deliberately exits 4 after successful provisioning; rerun it.
run rust-pto "$tile" "$input" -t pto --cross ascend -O0 --keep-dir "$evidence/keep-o0" -o "$evidence/rust.pto.mlir"
run rust-pto-o2 "$tile" "$input" -t pto --cross ascend --keep-dir "$evidence/keep-o2" -o "$evidence/rust-o2.pto.mlir"
run mlir-pto "$tile" "$subject/crates/tile_cli/testdata/forms/softmax.mlir" -t pto --cross ascend -O0 -o "$evidence/fixture.pto.mlir"
find "$TILE_HOME" -type f -print > "$evidence/installed-files.txt"
while IFS= read -r lib; do
  file "$lib"
  shasum -a 256 "$lib"
  otool -L "$lib"
done < <(find "$TILE_HOME" -name 'librustc_codegen_tile.dylib') > "$evidence/backend.txt"
# A separate instrumented run copies compiler outputs before lower_rs removes its
# temporary directory. It does not modify arguments, environment or compiler output.
export AUDIT_REAL_CARGO="$(command -v cargo)"
export AUDIT_EVIDENCE="$evidence"
mkdir -p "$RUNNER_TEMP/audit-shim"
cat > "$RUNNER_TEMP/audit-shim/cargo" <<'SHIM'
#!/bin/bash
"$AUDIT_REAL_CARGO" "$@"
rc=$?
case "$PWD" in
  */tile-lower-*)
    dst="$AUDIT_EVIDENCE/backend-capture/$(basename "$PWD")"
    mkdir -p "$dst"
    printf '%q ' "$@" > "$dst/cargo.args"
    printf '\nexit=%s\n' "$rc" >> "$dst/cargo.args"
    cp Cargo.toml "$dst/"
    if [ -f Cargo.lock ]; then cp Cargo.lock "$dst/"; fi
    cp -R src .cargo "$dst/"
    find target -type f \( -name '*.mlir' -o -name '*.tile.*' \) -print > "$dst/artifact-paths.txt"
    while IFS= read -r f; do
      mkdir -p "$dst/$(dirname "$f")"
      cp "$f" "$dst/$f"
    done < "$dst/artifact-paths.txt"
    ;;
esac
exit "$rc"
SHIM
chmod +x "$RUNNER_TEMP/audit-shim/cargo"
run rust-pto-captured env PATH="$RUNNER_TEMP/audit-shim:$PATH" "$tile" "$input" -t pto --cross ascend -O0 -o "$evidence/captured.pto.mlir"
# Controlled comparison: same CLI, backend, nightly and input; only the path
# dependency graph tile_std -> adjacent tile_std_macros changes to the release tag.
release_libs="$3"
{
  git -C "$release_libs" rev-parse HEAD
  git -C "$release_libs" status --porcelain
  printf 'baseline TILE_STD_PATH=%s\nrelease TILE_STD_PATH=%s\n' "$TILE_STD_PATH" "$release_libs/crates/tile_std"
  diff -u "$release_libs/crates/tile_std/src/lib.rs" "$TILE_STD_PATH/src/lib.rs" || true
} > "$evidence/library-comparison.txt"
export TILE_STD_PATH="$release_libs/crates/tile_std"
run release-libs-o0 "$tile" "$input" -t pto --cross ascend -O0 --keep-dir "$evidence/keep-release-o0" -o "$evidence/release-libs.pto.mlir"
run release-libs-o2 "$tile" "$input" -t pto --cross ascend --keep-dir "$evidence/keep-release-o2" -o "$evidence/release-libs-o2.pto.mlir"
run release-libs-captured env PATH="$RUNNER_TEMP/audit-shim:$PATH" "$tile" "$input" -t pto --cross ascend -O0 -o "$evidence/release-libs-captured.pto.mlir"
bash "$harness_dir/emit-only.sh" "$evidence" "$tile"
git -C "$release_libs" status --porcelain > "$evidence/release-libs-final-status.txt"
# Record source-tree cleanliness separately from remote harness changes.
git status --porcelain > "$evidence/subject-final-status.txt"
find "$evidence" -name '*.mlir' -exec shasum -a 256 {} \; > "$evidence/mlir-sha256.txt"
