#!/usr/bin/env bash
# THROWAWAY: Rust-derived 1x256 add, no handwritten-IR fallback.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${BUILD_DIR:?Set an isolated absolute build directory}"
: "${CANN_ENV:?Set the installed CANN environment script path}"
: "${EXECUTION_MODE:?Set sim or npu}"
: "${ASCEND_DEVICE_ID:?Set the allocated logical ACL device ID}"
[[ $BUILD_DIR == /* ]] || exit 2
[[ $EXECUTION_MODE == sim || $EXECUTION_MODE == npu ]] || exit 2
[[ ${INJECT_ERROR:-0} == 0 || ${INJECT_ERROR:-0} == 1 ]] || exit 2
mkdir -p "$BUILD_DIR"
work=$(cd "$BUILD_DIR" && pwd)
set +u
source "$CANN_ENV"
set -u
: "${ASCEND_HOME_PATH:?CANN environment did not set ASCEND_HOME_PATH}"
link_runtime=()
if [[ $EXECUTION_MODE == sim ]]; then
    [[ $(uname -m) == x86_64 && $ASCEND_DEVICE_ID == 0 ]] || exit 2
    sim="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib"
    test -f "$sim/libruntime_camodel.so"
    export LD_LIBRARY_PATH="$sim:${LD_LIBRARY_PATH:-}"
    link_runtime=(-L"$sim" -Wl,--no-as-needed -lruntime_camodel)
    python3 - <<'PY'
import glob, sys
assert sys.version_info[:2] == (3, 10)
assert not any(glob.glob(p) for p in ('/dev/davinci*', '/dev/devmm_svm', '/dev/hisi_hdc'))
PY
else
    # This prototype's vector target is limited to the 910B family.
    : "${ASCEND_SOC:?Set the actual allocated SoC, e.g. Ascend910B1}"
    [[ $ASCEND_SOC == Ascend910B* ]] || { echo 'FAIL: unsupported prototype SoC'; exit 2; }
    [[ ${LD_LIBRARY_PATH:-} != *simulator* && ${LD_PRELOAD:-} != *camodel* ]] || exit 2
    npu-smi info > "$work/device.txt"
    link_runtime=(-L"$ASCEND_HOME_PATH/lib64" -Wl,--no-as-needed -lruntime)
fi
sha=82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19
if [[ ! -d $work/pto-isa/.git ]]; then
    git init -q "$work/pto-isa"
    git -C "$work/pto-isa" remote add origin https://github.com/hw-native-sys/pto-isa.git
    git -C "$work/pto-isa" fetch --depth=1 origin "$sha"
    git -C "$work/pto-isa" checkout --detach FETCH_HEAD
fi
test "$(git -C "$work/pto-isa" rev-parse HEAD)" = "$sha"
test -z "$(git -C "$work/pto-isa" status --porcelain)"
{
    printf 'EXECUTION_MODE=%s\nDEVICE_ID=%s\nPTO_ISA=%s\n' "$EXECUTION_MODE" "$ASCEND_DEVICE_ID" "$sha"
    printf 'SOC=%s\nCCE_ARCH=dav-c220-vec\n' "${ASCEND_SOC:-Ascend910B1}"
    uname -a
    bisheng --version
    sha256sum "$(command -v bisheng)"
} > "$work/inventory.txt"
if [[ -n ${GENERATED_CPP:-} ]]; then
    # NPU job can consume the exact C++ emitted by the preceding simulator job.
    test -s "$GENERATED_CPP"
    cp "$GENERATED_CPP" "$work/generated.cpp"
else
    : "${PTO_INPUT:?Set the actual Rust frontend PTO artifact path}"
    test -s "$PTO_INPUT"
    [[ $(uname -m) == x86_64 ]] || exit 2
    cp "$PTO_INPUT" "$work/input.pto"
    wheel=ptoas-0.65-cp310-cp310-manylinux_2_34_x86_64.whl
    curl -fL --retry 3 "https://github.com/hw-native-sys/PTOAS/releases/download/v0.65/$wheel" -o "$work/$wheel"
    echo "28ba50ddc684b3b011262cd26677a2509428beaf5e267cb6bee5b4c1f776712b  $work/$wheel" | sha256sum -c -
    python3 -m venv "$work/venv"
    "$work/venv/bin/python" -m pip install --no-index --no-deps "$work/$wheel"
    "$work/venv/bin/ptoas" --version >> "$work/inventory.txt" 2>&1
    "$work/venv/bin/ptoas" "$work/input.pto" --pto-arch=a3 --enable-insert-sync -o "$work/generated.cpp" > "$work/ptoas.log" 2>&1
fi
sha256sum "$work/generated.cpp" >> "$work/inventory.txt"
cd "$work"
bisheng -xcce -O2 -std=c++17 --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE \
    -fPIC -shared --cce-fatobj-link -I"$work/pto-isa/include" -I"$work" \
    -I"$ASCEND_HOME_PATH/include" "$here/kernel.cpp" -o libadd.so > compile.log 2>&1
g++ -std=c++17 "$here/host.cpp" -I"$ASCEND_HOME_PATH/include" -L. -ladd \
    "${link_runtime[@]}" -L"$ASCEND_HOME_PATH/lib64" -lascendcl -ldl \
    '-Wl,-rpath,$ORIGIN' -o add > link.log 2>&1
python3 "$here/check.py" "$work/add" "$work" "${INJECT_ERROR:-0}" 2>&1 | tee "$work/result.txt"
