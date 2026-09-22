#!/usr/bin/env bash
# PROTOTYPE: no CANN, NPU runtime, Rust backend or GoogleTest required.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work=${BUILD_DIR:-$(mktemp -d)}
mkdir -p "$work"
if [[ -z ${BUILD_DIR:-} ]]; then trap 'rm -rf "$work"' EXIT; fi
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
source_file="$here/add.cpp"
if [[ ${PROGRAM_MODE:-handwritten} == generated ]]; then
    # Release wheel is CPython 3.12 / Linux x86_64. No floating pip dependencies.
    python3 -c 'import sys, platform; assert sys.version_info[:2] == (3, 12) and platform.machine() == "x86_64"'
    wheel=ptoas-0.65-cp312-cp312-manylinux_2_34_x86_64.whl
    curl -fL --retry 2 "https://github.com/hw-native-sys/PTOAS/releases/download/v0.65/$wheel" -o "$work/$wheel"
    echo "8fc9f14a4402db178341938f41faf9a9d66474a57b7524b3fc331ffc6db27795  $work/$wheel" | sha256sum -c -
    python3 -m venv "$work/venv"
    "$work/venv/bin/python" -m pip install --no-index --no-deps "$work/$wheel"
    "$work/venv/bin/ptoas" --version
    cp "$here/add.pto" "$work/input.pto"
    "$work/venv/bin/ptoas" "$work/input.pto" --pto-arch=a3 --enable-insert-sync -o "$work/generated.cpp"
    sha256sum "$work/input.pto" "$work/generated.cpp"
    source_file="$here/generated-host.cpp"
fi
printf 'PROGRAM_MODE=%s\n' "${PROGRAM_MODE:-handwritten}"
"$cxx" -std=c++23 -O0 -D__CPU_SIM -I"$work/pto-isa/include" -I"$work" "$source_file" -o "$work/add"
python3 "$here/check.py" "$work/add" "$work" "${INJECT_ERROR:-0}"
