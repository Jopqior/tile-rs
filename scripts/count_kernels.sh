#!/usr/bin/env bash
# Count all Rust NPU kernels (#[ascend_std::aiv_kernel]) in the repository.
# Groups by category: deployable crates, compiletests, equivalence tests, docs.
#
# Usage: bash scripts/count_kernels.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Grep pattern: match the attribute, exclude comment lines (/// or //)
ATTR='#\[ascend_std::aiv_kernel\]'

# Find kernel attributes, skipping lines that are Rust doc-comments or comments
grep_kernels() {
    grep -rn "$ATTR" "$@" 2>/dev/null | grep -v '^\([^:]*:[0-9]*:\)\s*//' || true
}

count_kernels() {
    grep_kernels "$@" | wc -l
}

# Print each kernel: extract fn name from the line after the attribute
list_kernels() {
    grep_kernels "$@" | while IFS= read -r match; do
        file="${match%%:*}"
        rest="${match#*:}"
        lineno="${rest%%:*}"
        fn_line=$(sed -n "$((lineno + 1))p" "$file" 2>/dev/null)
        fn_name=$(echo "$fn_line" | grep -oP 'fn \K\w+' || echo "?")
        relpath="${file#$ROOT_DIR/}"
        printf "    %-30s  %s\n" "$fn_name" "$relpath"
    done
}

echo "=== Rust NPU Kernel Count ==="
echo ""

# --- Deployable kernel crates ---
deploy_count=0
deploy_list=""
for d in "$ROOT_DIR/examples" "$ROOT_DIR/debug/ref_store42_rs"; do
    [ -d "$d" ] || continue
    n=$(grep_kernels --include='*.rs' "$d" | grep -v 'compiletest\|kernel_equiv' | wc -l)
    deploy_count=$((deploy_count + n))
done
echo "Deployable kernels: $deploy_count"
for d in "$ROOT_DIR/examples" "$ROOT_DIR/debug/ref_store42_rs"; do
    [ -d "$d" ] || continue
    grep_kernels --include='*.rs' "$d" | grep -v 'compiletest\|kernel_equiv' | while IFS= read -r match; do
        file="${match%%:*}"
        rest="${match#*:}"
        lineno="${rest%%:*}"
        fn_line=$(sed -n "$((lineno + 1))p" "$file" 2>/dev/null)
        fn_name=$(echo "$fn_line" | grep -oP 'fn \K\w+' || echo "?")
        relpath="${file#$ROOT_DIR/}"
        printf "    %-30s  %s\n" "$fn_name" "$relpath"
    done
done
echo ""

# --- Compiletest kernels ---
ct_dir="$ROOT_DIR/tests/compiletest/ui"
if [ -d "$ct_dir" ]; then
    ct_count=$(count_kernels --include='*.rs' "$ct_dir")
    echo "Compiletest kernels: $ct_count"
    list_kernels --include='*.rs' "$ct_dir"
else
    ct_count=0
    echo "Compiletest kernels: 0"
fi
echo ""

# --- Equivalence test kernels ---
eq_dir="$ROOT_DIR/debug/kernel_equiv"
if [ -d "$eq_dir" ]; then
    eq_count=$(count_kernels --include='*.rs' "$eq_dir")
    echo "Equivalence test kernels: $eq_count"
    list_kernels --include='*.rs' "$eq_dir"
else
    eq_count=0
    echo "Equivalence test kernels: 0"
fi
echo ""

# --- Totals ---
total=$(count_kernels --include='*.rs' "$ROOT_DIR" | head -1)
doc_count=$(grep -rn "$ATTR" "$ROOT_DIR" --include='*.md' 2>/dev/null | wc -l)

echo "==============================="
printf "  Deployable:    %3d\n" "$deploy_count"
printf "  Compiletests:  %3d\n" "$ct_count"
printf "  Equiv tests:   %3d\n" "$eq_count"
printf "  (Docs/blog:    %3d)\n" "$doc_count"
echo "-------------------------------"
printf "  Total (code):  %3d\n" "$total"
echo "==============================="
