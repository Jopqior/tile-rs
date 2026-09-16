#!/usr/bin/env bash
# =============================================================================
# Unified Kernel Benchmark: Rust vs C++ on Ascend NPU
# =============================================================================
#
# Builds and runs all kernel benchmark pairs, collects CSV results,
# and generates a combined report.
#
# Prerequisites:
#   source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
#   export ACLRS_RUN_MODE=npu
#   cargo build -p rustc_codegen_mlir --release
#   export LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH
#
# Usage:
#   bash benchmarks/kernel_bench/bench.sh [--skip-build] [--only <kernel>]

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/results"

SKIP_BUILD=0
ONLY_KERNEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1; shift ;;
        --only) ONLY_KERNEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

# Auto-source CANN environment if not already set
if [ -z "$ASCEND_HOME_PATH" ]; then
    SETENV="/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"
    if [ -f "$SETENV" ]; then
        echo "Sourcing CANN environment: $SETENV"
        source "$SETENV"
    else
        echo "WARNING: ASCEND_HOME_PATH not set and $SETENV not found"
    fi
fi

export ACLRS_RUN_MODE="${ACLRS_RUN_MODE:-npu}"

echo "========================================"
echo " Unified Kernel Benchmark Suite"
echo "========================================"
echo "Root: $ROOT_DIR"
echo "Output: $OUT_DIR"
echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-<unset>}"
echo "ACLRS_RUN_MODE: $ACLRS_RUN_MODE"
echo ""

# --- Helper functions ---
build_crate() {
    local name="$1"
    local manifest="$2"
    if [ $SKIP_BUILD -eq 0 ]; then
        echo "--- Building $name ---"
        cargo build --release --manifest-path "$manifest" 2>&1
        echo ""
    fi
}

run_bench() {
    local name="$1"
    local bin="$2"
    local csv="$3"
    echo "--- Running $name ---"
    if [ -x "$bin" ]; then
        "$bin" --csv "$csv" 2>&1
        echo ""
    else
        echo "SKIP: $bin not found"
        echo ""
    fi
}

should_run() {
    local kernel="$1"
    [ -z "$ONLY_KERNEL" ] || [ "$ONLY_KERNEL" = "$kernel" ]
}

# --- Build codegen backend (needed for Rust kernels) ---
if [ $SKIP_BUILD -eq 0 ]; then
    echo "--- Building rustc_codegen_mlir ---"
    cargo build -p rustc_codegen_mlir --release 2>&1
    echo ""
fi

# ==========================================================================
# Softmax
# ==========================================================================
if should_run "softmax"; then
    build_crate "bench_softmax_rs" "$ROOT_DIR/examples/bench_softmax_rs/Cargo.toml"
    build_crate "bench_softmax_cpp" "$ROOT_DIR/examples/bench_softmax_cpp/Cargo.toml"
    run_bench "softmax (Rust)" \
        "$ROOT_DIR/examples/bench_softmax_rs/target/release/bench_softmax_rs" \
        "$OUT_DIR/softmax_rust.csv"
    run_bench "softmax (C++)" \
        "$ROOT_DIR/examples/bench_softmax_cpp/target/release/bench_softmax_cpp" \
        "$OUT_DIR/softmax_cpp.csv"
fi

# ==========================================================================
# Vec Add (f16)
# ==========================================================================
if should_run "vec_add"; then
    build_crate "bench_vec_add_rs" "$ROOT_DIR/examples/bench_vec_add_rs/Cargo.toml"
    build_crate "bench_vec_add_cpp" "$ROOT_DIR/examples/bench_vec_add_cpp/Cargo.toml"
    run_bench "vec_add (Rust)" \
        "$ROOT_DIR/examples/bench_vec_add_rs/target/release/bench_vec_add_rs" \
        "$OUT_DIR/vec_add_rust.csv"
    run_bench "vec_add (C++)" \
        "$ROOT_DIR/examples/bench_vec_add_cpp/target/release/bench_vec_add_cpp" \
        "$OUT_DIR/vec_add_cpp.csv"
fi

# ==========================================================================
# Matmul (f16 -> f32)
# ==========================================================================
if should_run "matmul"; then
    build_crate "bench_matmul_rs" "$ROOT_DIR/examples/bench_matmul_rs/Cargo.toml"
    build_crate "bench_matmul_cpp" "$ROOT_DIR/examples/bench_matmul_cpp/Cargo.toml"
    run_bench "matmul (Rust)" \
        "$ROOT_DIR/examples/bench_matmul_rs/target/release/bench_matmul_rs" \
        "$OUT_DIR/matmul_rust.csv"
    run_bench "matmul (C++)" \
        "$ROOT_DIR/examples/bench_matmul_cpp/target/release/bench_matmul_cpp" \
        "$OUT_DIR/matmul_cpp.csv"
fi

# ==========================================================================
# LayerNorm
# ==========================================================================
if should_run "layernorm"; then
    build_crate "bench_layernorm_rs" "$ROOT_DIR/examples/bench_layernorm_rs/Cargo.toml"
    build_crate "bench_layernorm_cpp" "$ROOT_DIR/examples/bench_layernorm_cpp/Cargo.toml"
    run_bench "layernorm (Rust)" \
        "$ROOT_DIR/examples/bench_layernorm_rs/target/release/bench_layernorm_rs" \
        "$OUT_DIR/layernorm_rust.csv"
    run_bench "layernorm (C++)" \
        "$ROOT_DIR/examples/bench_layernorm_cpp/target/release/bench_layernorm_cpp" \
        "$OUT_DIR/layernorm_cpp.csv"
fi

# ==========================================================================
# Tile Matmul (f32, Rust tile API — scalar fallback via mlir_to_cpp)
# ==========================================================================
if should_run "tile_matmul"; then
    build_crate "tile_matmul" "$ROOT_DIR/examples/tile_matmul/Cargo.toml"
    run_bench "tile_matmul (Rust tile API, f32)" \
        "$ROOT_DIR/target/release/tile_matmul" \
        "$OUT_DIR/tile_matmul.csv"
fi

# ==========================================================================
# Tile Matmul PTO (f32, Rust tile API — cube unit via ptoas)
# ==========================================================================
if should_run "tile_matmul_pto"; then
    # Find ptoas binary
    for ptoas_candidate in /data/czq/ptoas-bin/ptoas /data/linyifan/ptoas-bin/ptoas; do
        if [ -x "$ptoas_candidate" ]; then
            export ACLRS_PTOAS_PATH="$ptoas_candidate"
            break
        fi
    done
    if [ -n "${ACLRS_PTOAS_PATH:-}" ]; then
        export TILERS_CODEGEN_PATH=pto
        build_crate "tile_matmul (PTO)" "$ROOT_DIR/examples/tile_matmul/Cargo.toml"
        run_bench "tile_matmul (PTO path, f32)" \
            "$ROOT_DIR/target/release/tile_matmul" \
            "$OUT_DIR/tile_matmul_pto.csv"
        unset TILERS_CODEGEN_PATH
    else
        echo "SKIP: tile_matmul_pto — ptoas binary not found"
    fi
fi

# ==========================================================================
# Combine CSVs and generate report
# ==========================================================================
echo "--- Combining results ---"
> "$OUT_DIR/combined.csv"
for f in "$OUT_DIR"/*.csv; do
    if [ "$f" != "$OUT_DIR/combined.csv" ] && [ -f "$f" ]; then
        cat "$f" >> "$OUT_DIR/combined.csv"
    fi
done

echo ""
echo "--- Report ---"
if command -v python3 &>/dev/null; then
    python3 "$SCRIPT_DIR/report.py" "$OUT_DIR/combined.csv"
else
    echo "(python3 not found — raw CSV follows)"
    cat "$OUT_DIR/combined.csv"
fi

echo ""
echo "CSV files saved to: $OUT_DIR/"
