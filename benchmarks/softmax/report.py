#!/usr/bin/env python3
"""
Aggregate softmax benchmark CSV and print a comparison table.

Input CSV format (no header):
  BENCH,<size>,<kernel>,<run>,<time_ms>

Usage:
  python3 report.py results/combined.csv
"""

import sys
import csv
from collections import defaultdict


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def main():
    if len(sys.argv) < 2:
        print("Usage: report.py <combined.csv>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    # Collect times: {(size, kernel): [time_ms, ...]}
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 5 or row[0] != "BENCH":
                continue
            size = int(row[1])
            kernel = row[2]
            time_ms = float(row[4])
            data[(size, kernel)].append(time_ms)

    sizes = sorted({k[0] for k in data})
    kernels = ["rust_vector", "cpp_naive", "cpp_opt"]
    present_kernels = [k for k in kernels if any((s, k) in data for s in sizes)]

    if not present_kernels:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    # Compute stats
    stats = {}
    for key, times in data.items():
        stats[key] = {
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / len(times),
            "median": median(times),
            "n": len(times),
        }

    # Print table
    # Header
    cols = ["Size"]
    for k in present_kernels:
        cols.append(f"{k} med(ms)")
    if "rust_vector" in present_kernels and "cpp_naive" in present_kernels:
        cols.append("Rust/Naive")
    if "rust_vector" in present_kernels and "cpp_opt" in present_kernels:
        cols.append("Rust/Opt")

    widths = [max(8, len(c) + 2) for c in cols]

    def fmt_row(values):
        parts = []
        for v, w in zip(values, widths):
            parts.append(str(v).rjust(w))
        return " | ".join(parts)

    print()
    print(fmt_row(cols))
    print("-+-".join("-" * w for w in widths))

    for size in sizes:
        row = [str(size)]
        medians = {}
        for k in present_kernels:
            key = (size, k)
            if key in stats:
                med = stats[key]["median"]
                medians[k] = med
                row.append(f"{med:.4f}")
            else:
                row.append("-")

        if "rust_vector" in medians and "cpp_naive" in medians and medians["cpp_naive"] > 0:
            ratio = medians["rust_vector"] / medians["cpp_naive"]
            row.append(f"{ratio:.2f}x")
        elif "Rust/Naive" in cols:
            row.append("-")

        if "rust_vector" in medians and "cpp_opt" in medians and medians["cpp_opt"] > 0:
            ratio = medians["rust_vector"] / medians["cpp_opt"]
            row.append(f"{ratio:.2f}x")
        elif "Rust/Opt" in cols:
            row.append("-")

        print(fmt_row(row))

    print()

    # Print detailed stats
    print("Detailed statistics:")
    print(f"  {'Kernel':<12} {'Size':>6} {'N':>3} {'Min':>10} {'Median':>10} {'Mean':>10} {'Max':>10}")
    print("  " + "-" * 65)
    for size in sizes:
        for k in present_kernels:
            key = (size, k)
            if key in stats:
                s = stats[key]
                print(
                    f"  {k:<12} {size:>6} {s['n']:>3} "
                    f"{s['min']:>10.4f} {s['median']:>10.4f} "
                    f"{s['mean']:>10.4f} {s['max']:>10.4f}"
                )
    print()


if __name__ == "__main__":
    main()
