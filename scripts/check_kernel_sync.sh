#!/bin/bash
# Check that every kernel in compiletests has a corresponding CPU correctness test.
# Usage: bash scripts/check_kernel_sync.sh

set -euo pipefail

COMPILETEST_DIR="tests/compiletest/ui"
CORRECTNESS_DIR="tests/kernel_correctness"

# Files containing the 68 new scalar-loop kernels
NEW_KERNEL_FILES=(
    conv_standard_kernel.rs
    conv_depthwise_kernel.rs
    conv_transpose_kernel.rs
    index_ops_kernel.rs
    broadcast_ext_kernel.rs
    math_ext_kernel.rs
    loss_ext_kernel.rs
    optimizer_ext_kernel.rs
    pooling_windowed_kernel.rs
    matmul_transpose_kernel.rs
    resize_spatial_kernel.rs
)

echo "=== Kernel Sync Check ==="
echo ""

# 1. Count compiletest kernels in new files
compile_count=0
compile_names=()
for f in "${NEW_KERNEL_FILES[@]}"; do
    path="$COMPILETEST_DIR/$f"
    if [ -f "$path" ]; then
        while IFS= read -r name; do
            compile_names+=("$name")
            compile_count=$((compile_count + 1))
        done < <(grep -oP '(?<=pub unsafe fn )\w+' "$path")
    else
        echo "WARNING: Missing compiletest file: $path"
    fi
done
echo "Compiletest kernels (new files): $compile_count"

# 2. Count correctness test functions
test_count=0
for f in "$CORRECTNESS_DIR"/tests/test_*.rs; do
    if [ -f "$f" ]; then
        c=$(grep -c '#\[test\]' "$f" || true)
        test_count=$((test_count + c))
    fi
done
echo "Correctness tests: $test_count"

# 3. Count CPU reference functions
ref_count=0
for f in "$CORRECTNESS_DIR"/src/*.rs; do
    if [ -f "$f" ] && [ "$(basename "$f")" != "lib.rs" ]; then
        c=$(grep -c '^pub fn ' "$f" || true)
        ref_count=$((ref_count + c))
    fi
done
echo "CPU reference functions: $ref_count"

# 4. Check for golden value files
golden_count=0
if [ -d "$CORRECTNESS_DIR/golden" ]; then
    golden_count=$(find "$CORRECTNESS_DIR/golden" -name "*.json" | wc -l)
fi
echo "Golden value files: $golden_count"

# 5. Map kernel categories to test files
echo ""
echo "=== Coverage by category ==="
declare -A CATEGORY_MAP=(
    [conv_standard_kernel.rs]="conv.rs"
    [conv_depthwise_kernel.rs]="conv.rs"
    [conv_transpose_kernel.rs]="conv.rs"
    [index_ops_kernel.rs]="index.rs"
    [broadcast_ext_kernel.rs]="broadcast.rs"
    [math_ext_kernel.rs]="math.rs"
    [loss_ext_kernel.rs]="loss.rs"
    [optimizer_ext_kernel.rs]="optimizer.rs"
    [pooling_windowed_kernel.rs]="pooling.rs"
    [matmul_transpose_kernel.rs]="matmul.rs"
    [resize_spatial_kernel.rs]="resize.rs"
)

missing=0
for f in "${NEW_KERNEL_FILES[@]}"; do
    ref_file="${CATEGORY_MAP[$f]}"
    path="$COMPILETEST_DIR/$f"
    ref_path="$CORRECTNESS_DIR/src/$ref_file"

    if [ ! -f "$path" ]; then
        echo "  MISSING compiletest: $f"
        missing=$((missing + 1))
        continue
    fi

    kernel_count=$(grep -c 'pub unsafe fn ' "$path" || true)

    if [ ! -f "$ref_path" ]; then
        echo "  MISSING reference: $ref_file (covers $f, $kernel_count kernels)"
        missing=$((missing + 1))
    else
        ref_fn_count=$(grep -c '^pub fn ' "$ref_path" || true)
        echo "  OK: $f ($kernel_count kernels) -> $ref_file ($ref_fn_count functions)"
    fi
done

echo ""
if [ $missing -eq 0 ]; then
    echo "All kernel categories have CPU reference implementations."
else
    echo "WARNING: $missing categories missing coverage!"
fi

# 6. Run the tests
echo ""
echo "=== Running correctness tests ==="
cargo test -p kernel_correctness 2>&1
