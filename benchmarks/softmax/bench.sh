#!/usr/bin/env bash
# =============================================================================
# Softmax Benchmark: Rust vs C++ on Ascend NPU
# =============================================================================
#
# Prerequisites:
#   source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
#   export ACLRS_RUN_MODE=npu
#   cargo build -p rustc_codegen_mlir --release
#   export LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH
#
# Usage:
#   bash benchmarks/softmax/bench.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/results"

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

echo "=== Softmax Benchmark ==="
echo "Root: $ROOT_DIR"
echo "Output: $OUT_DIR"
echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH:-<unset>}"
echo "ACLRS_RUN_MODE: $ACLRS_RUN_MODE"
echo ""

# -- ACL sanity check (C program, bypasses all Rust) -----------------------
echo "--- ACL sanity check (C) ---"
ACL_TEST_SRC="$SCRIPT_DIR/acl_test.c"
ACL_TEST_BIN="$OUT_DIR/acl_test"
CANN_INC="$ASCEND_HOME_PATH/include"
CANN_LIB="$ASCEND_HOME_PATH/lib64"

if [ -f "$ACL_TEST_SRC" ]; then
    gcc -o "$ACL_TEST_BIN" "$ACL_TEST_SRC" \
        -I"$CANN_INC" -L"$CANN_LIB" -lascendcl \
        -Wl,-rpath="$CANN_LIB" 2>&1 || echo "(C compile failed)"
    if [ -x "$ACL_TEST_BIN" ]; then
        "$ACL_TEST_BIN" || echo "(acl_test exited with $?)"
        echo ""
        echo "--- C binary library deps ---"
        ldd "$ACL_TEST_BIN" 2>&1 | grep -i -E "ascend|acl|runtime" || true
    fi
fi
echo ""

# -- Build both benchmarks ---------------------------------------------------
echo "--- Building bench_softmax_rs ---"
cargo build --release --manifest-path "$ROOT_DIR/examples/bench_softmax_rs/Cargo.toml"

echo ""
echo "--- Building bench_softmax_cpp ---"
cargo build --release --manifest-path "$ROOT_DIR/examples/bench_softmax_cpp/Cargo.toml"

echo ""

# -- Run Rust benchmark ------------------------------------------------------
echo "--- Running Rust softmax benchmark ---"
RS_BIN="$ROOT_DIR/examples/bench_softmax_rs/target/release/bench_softmax_rs"

echo "--- Rust binary library deps ---"
ldd "$RS_BIN" 2>&1 | grep -i -E "ascend|acl|runtime" || true
echo ""
echo "--- Rust binary RPATH ---"
readelf -d "$RS_BIN" 2>/dev/null | grep -i -E "rpath|runpath" || true
echo ""

"$RS_BIN" --csv "$OUT_DIR/rust.csv"

echo ""

# -- Run C++ benchmark -------------------------------------------------------
echo "--- Running C++ softmax benchmark ---"
CPP_BIN="$ROOT_DIR/examples/bench_softmax_cpp/target/release/bench_softmax_cpp"
"$CPP_BIN" --csv "$OUT_DIR/cpp.csv"

echo ""

# -- Combine CSVs and generate report ----------------------------------------
cat "$OUT_DIR/rust.csv" "$OUT_DIR/cpp.csv" > "$OUT_DIR/combined.csv"

echo "--- Results ---"
if command -v python3 &>/dev/null; then
    python3 "$SCRIPT_DIR/report.py" "$OUT_DIR/combined.csv"
else
    echo "(python3 not found — raw CSV follows)"
    cat "$OUT_DIR/combined.csv"
fi

echo ""
echo "CSV files saved to: $OUT_DIR/"
