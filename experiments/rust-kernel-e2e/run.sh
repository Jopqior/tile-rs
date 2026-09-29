#!/usr/bin/env bash
# Temporary, public-source-only experiment. Evidence contains no credentials or bundled binaries.
set -euo pipefail
BASE_SHA=2924af350f8843738f276b9a8695b0416a2d3d49
TAG='v0.0.2+nightly-2025-08-04'
EXPECTED_SHA=70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891
EXPERIMENT=experiment/rust-kernel-e2e-2924af3-v002-20260929
OUT="$RUNNER_TEMP/rust-kernel-e2e"
EV="$OUT/evidence"
WORK="$OUT/work"
BASE="$WORK/baseline"
mkdir -p "$EV/logs" "$EV/inputs" "$EV/mlir" "$EV/msl" "$WORK"
trap 'rc=$?; printf "%s\n" "$rc" > "$EV/exit-code.txt"; echo "experiment exit=$rc (evidence: $EV)"' EXIT

record() { printf '%s\t%s\n' "$1" "$2" | tee -a "$EV/stages.tsv"; }
# Keep stdout and stderr separate, even if a stage fails; exact argv in evidence.
run() {
  local name="$1" rc
  shift
  printf '%q ' "$@" >> "$EV/commands.txt"
  printf '\n' >> "$EV/commands.txt"
  set +e
  "$@" > "$EV/logs/$name.stdout" 2> "$EV/logs/$name.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$EV/logs/$name.exit"
  echo "$name exit=$rc"
  tail -n 60 "$EV/logs/$name.stderr" || true
}
required() {
  local name="$1"
  shift
  run "$name" "$@"
  if [ "$(< "$EV/logs/$name.exit")" != 0 ]; then
    record "$name" setup_failed
    echo "Setup/experiment wiring failed: $name; see uploaded stdout/stderr" >&2
    exit 2
  fi
}

printf 'baseline=%s\nrelease=%s\nexpected_sha256=%s\nbranch=%s\nrun=%s attempt=%s\n' \
  "$BASE_SHA" "$TAG" "$EXPECTED_SHA" "$EXPERIMENT" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT" > "$EV/identity.txt"
if [ "$GITHUB_REF_NAME" != "$EXPERIMENT" ]; then record branch wrong_branch; exit 2; fi
if [ "$(uname -s -m)" != 'Darwin arm64' ]; then record platform wrong_arch; exit 2; fi
{
  date -u
  sw_vers
  uname -a
  echo "ImageOS=${ImageOS:-} RUNNER_IMAGE_VERSION=${RUNNER_IMAGE_VERSION:-} RUNNER_ARCH=${RUNNER_ARCH:-}"
  git rev-parse HEAD
  xcodebuild -version
  rustup --version
} > "$EV/environment.txt" 2>&1
required baseline git worktree add --detach "$BASE" "$BASE_SHA"
if [ "$(git -C "$BASE" rev-parse HEAD)" != "$BASE_SHA" ]; then record baseline wrong_commit; exit 2; fi
record baseline "$BASE_SHA"

# Pinned toolchain plus rustc internals needed by the published backend. Ephemeral hosted runner only.
required toolchain rustup toolchain install nightly-2025-08-04 --profile minimal --component rustc-dev --component llvm-tools --component rust-src
required rustc rustc +nightly-2025-08-04 -vV
required cargo_version cargo +nightly-2025-08-04 -V
required components rustup component list --installed --toolchain nightly-2025-08-04

API='https://api.github.com/repos/yijunyu/tile-rs/releases/tags/v0.0.2%2Bnightly-2025-08-04'
required release_api curl --fail --silent --show-error --location --max-time 90 "$API" -o "$EV/release-api.json"
# Do not infer source provenance from the public tag or the packaged macro dylib.
python3 - "$EV/release-api.json" "$EV/asset-url.txt" "$TAG" "$EXPECTED_SHA" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
assert j['tag_name'] == sys.argv[3], 'release tag changed'
assets = [a for a in j['assets'] if a['name'] == 'tile-rs-codegen-aarch64-apple-darwin.tar.gz']
assert len(assets) == 1, 'expected exactly one macOS arm64 asset'
a = assets[0]
assert a['digest'] == 'sha256:' + sys.argv[4], 'API digest differs from pinned checksum'
assert a['browser_download_url'].startswith('https://github.com/yijunyu/tile-rs/releases/download/'), 'unexpected URL'
open(sys.argv[2], 'w').write(a['browser_download_url'] + '\n')
print('API size=', a['size'], 'digest=', a['digest'])
PY
URL="$(< "$EV/asset-url.txt")"
TARBALL="$WORK/release.tar.gz"
required download curl --fail --silent --show-error --location --max-time 600 "$URL" -o "$TARBALL"
required sha256 shasum -a 256 "$TARBALL"
ACTUAL="$(awk '{print $1}' "$EV/logs/sha256.stdout")"
if [ "$ACTUAL" != "$EXPECTED_SHA" ]; then record download checksum_mismatch; exit 2; fi
python3 - "$EV/release-api.json" "$TARBALL" <<'PY'
import json, os, sys
j = json.load(open(sys.argv[1]))
a = next(a for a in j['assets'] if a['name'] == 'tile-rs-codegen-aarch64-apple-darwin.tar.gz')
assert os.path.getsize(sys.argv[2]) == a['size'], 'download size differs from release API'
PY
record download verified_sha256

# Validate every member BEFORE extraction; no absolute/traversal paths, links, devices or scripts.
python3 - "$TARBALL" "$EV/archive-members.txt" <<'PY'
import pathlib, sys, tarfile
root = 'tile-rs-codegen-aarch64-apple-darwin'
with tarfile.open(sys.argv[1], 'r:gz') as archive:
    names = []
    for member in archive:
        p = pathlib.PurePosixPath(member.name)
        assert not p.is_absolute() and '..' not in p.parts and p.parts[0] == root, member.name
        assert member.isdir() or member.isfile(), 'link/special member: ' + member.name
        names.append(member.name)
    for required in (root + '/USAGE.md', root + '/.cargo/config.toml', root + '/lib/librustc_codegen_tile.dylib'):
        assert required in names, 'missing bundle member: ' + required
    pathlib.Path(sys.argv[2]).write_text('\n'.join(names) + '\n')
PY
mkdir -p "$WORK/unpacked"
required extract tar -xzf "$TARBALL" -C "$WORK/unpacked"
BUNDLE="$WORK/unpacked/tile-rs-codegen-aarch64-apple-darwin"
cp "$BUNDLE/USAGE.md" "$EV/inputs/release-USAGE.md"
cp "$BUNDLE/.cargo/config.toml" "$EV/inputs/release-config.toml"
# Audit the documented flags / target against the selected usage before running the dylib.
if ! grep -q 'nightly-2025-08-04' "$BUNDLE/USAGE.md" ||
   ! grep -q 'register_tool(tile)' "$BUNDLE/.cargo/config.toml" ||
   ! grep -q 'codegen-backend=' "$BUNDLE/.cargo/config.toml"; then
  record release_usage unexpected_config; exit 2
fi
record release_usage reviewed

mkdir -p "$WORK/kernel/src" "$WORK/kernel/.cargo" "$WORK/emitter/src"
cp experiments/rust-kernel-e2e/kernel/src/lib.rs "$WORK/kernel/src/lib.rs"
cp experiments/rust-kernel-e2e/emitter/src/main.rs "$WORK/emitter/src/main.rs"
cat > "$WORK/kernel/Cargo.toml" <<EOF
[package]
name = "kernel_add_e2e"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
tile_std = { path = "$BASE/crates/tile_std" }
[lib]
crate-type = ["cdylib", "lib"]
EOF
# Explicit --target makes target.rustflags apply to the kernel/tile_std, not host proc macros.
# This still builds tile_std_macros FROM THE FIXED SOURCE, never from the bundle dylib.
cat > "$WORK/kernel/.cargo/config.toml" <<EOF
[target.aarch64-apple-darwin]
rustflags = [
  "-Zcodegen-backend=$BUNDLE/lib/librustc_codegen_tile.dylib",
  "-Zcrate-attr=feature(register_tool)",
  "-Zcrate-attr=register_tool(tile)",
  "-Cpanic=abort",
  "-Clto=off",
]
EOF
cat > "$WORK/emitter/Cargo.toml" <<EOF
[package]
name = "fixed_source_emitter_probe"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
tile_codegen = { path = "$BASE/crates/tile_codegen", features = ["emitters"] }
EOF
cp "$WORK/kernel/src/lib.rs" "$EV/inputs/kernel-lib.rs"
cp "$WORK/emitter/src/main.rs" "$EV/inputs/emitter-main.rs"
cp "$WORK/kernel/Cargo.toml" "$EV/inputs/kernel-Cargo.toml"
cp "$WORK/kernel/.cargo/config.toml" "$EV/inputs/kernel-config.toml"
cp "$WORK/emitter/Cargo.toml" "$EV/inputs/emitter-Cargo.toml"
(cd "$EV/inputs" && shasum -a 256 kernel-Cargo.toml kernel-config.toml kernel-lib.rs emitter-Cargo.toml emitter-main.rs > ../input-sha256.txt)

run frontend env CARGO_TARGET_DIR="$WORK/kernel-target" TILERS_CODEGEN_PATH=metal \
  TILERS_CODEGEN_SO="$BUNDLE/lib/librustc_codegen_tile.dylib" \
  cargo +nightly-2025-08-04 build --manifest-path "$WORK/kernel/Cargo.toml" \
  --target aarch64-apple-darwin -v
FRONT_RC="$(< "$EV/logs/frontend.exit")"
if [ "$FRONT_RC" = 0 ]; then record frontend compiled; else record frontend compilation_failed; fi
find "$WORK/kernel-target" -type f -name '*.mlir' -print | sort > "$EV/mlir-paths.txt" || true
# Dependency MLIR is not the test: select only exported MLIR containing the kernel symbol.
selected=''
while IFS= read -r file; do
  cp "$file" "$EV/mlir/$(basename "$file")"
  if grep -q 'add_f32_small' "$file"; then
    if [ -n "$selected" ]; then record mlir ambiguous_kernel_files; exit 2; fi
    selected="$file"
  fi
done < "$EV/mlir-paths.txt"
if [ -z "$selected" ]; then
  record mlir absent_kernel_mlir
  exit 1
fi
cp "$selected" "$EV/mlir/actual-kernel.mlir"
shasum -a 256 "$EV/mlir/actual-kernel.mlir" > "$EV/mlir/actual-kernel.sha256"
record mlir exported_actual_kernel

# No packaged *.tile.* file is consumed. Host Cargo has no codegen backend rustflags.
run emitter_build env CARGO_TARGET_DIR="$WORK/emitter-target" \
  cargo +nightly-2025-08-04 build --manifest-path "$WORK/emitter/Cargo.toml" -v
if [ "$(< "$EV/logs/emitter_build.exit")" != 0 ]; then record emitter setup_failed; exit 2; fi
run emitter "$WORK/emitter-target/debug/fixed_source_emitter_probe" \
  "$EV/mlir/actual-kernel.mlir" "$EV/msl/fixed-source.metal"
if [ "$(< "$EV/logs/emitter.exit")" != 0 ]; then record emitter rejected_actual_mlir; exit 1; fi
shasum -a 256 "$EV/msl/fixed-source.metal" > "$EV/msl/fixed-source.sha256"
record emitter accepted_generated_msl
if [ "$FRONT_RC" != 0 ]; then exit 1; fi
