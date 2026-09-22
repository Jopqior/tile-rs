#!/usr/bin/env bash
# PROTOTYPE: no CANN, NPU runtime, Rust backend, pip or GoogleTest required.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
sha=82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19
cxx=${CXX:-g++-13}
"$cxx" --version
python3 --version
git --version
git init -q "$work/pto-isa"
git -C "$work/pto-isa" remote add origin https://github.com/hw-native-sys/pto-isa.git
git -C "$work/pto-isa" fetch --depth=1 origin "$sha"
git -C "$work/pto-isa" checkout --detach FETCH_HEAD
test "$(git -C "$work/pto-isa" rev-parse HEAD)" = "$sha"
"$cxx" -std=c++23 -O0 -D__CPU_SIM -I"$work/pto-isa/include" "$here/add.cpp" -o "$work/add"
python3 "$here/check.py" "$work/add" "$work" "${INJECT_ERROR:-0}"
