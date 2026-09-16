#!/usr/bin/env python3
"""
Generate golden reference data for softmax benchmark verification.

Creates input and expected output binary files for each benchmark size.
Uses the same LCG PRNG as the Rust/C++ benchmarks for reproducibility.

Usage:
  python3 gen_data.py [output_dir]
"""

import sys
import os
import struct
import numpy as np

SIZES = [256, 1024, 4096, 16384]


def lcg_generate(n, seed):
    """Match the LCG PRNG used in the Rust/C++ benchmarks."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    values = []
    for _ in range(n):
        state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        # Map to [-1.0, 1.0]
        val = ((state >> 33) / 4294967295.0) * 2.0 - 1.0
        values.append(np.float32(val))
    return np.array(values, dtype=np.float32)


def softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / np.sum(e)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "golden_data"
    os.makedirs(out_dir, exist_ok=True)

    for size in SIZES:
        inp = lcg_generate(size, size)
        out = softmax(inp)

        inp_path = os.path.join(out_dir, f"input_{size}.bin")
        out_path = os.path.join(out_dir, f"expected_{size}.bin")

        inp.tofile(inp_path)
        out.tofile(out_path)

        print(f"Size {size:>5}: sum(softmax) = {np.sum(out):.8f}  "
              f"files: {inp_path}, {out_path}")

    print(f"\nGolden data written to: {out_dir}/")


if __name__ == "__main__":
    main()
