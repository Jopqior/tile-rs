#!/usr/bin/env python3
"""Generate golden-value test cases for kernel_correctness using PyTorch.

Usage:
    python generate.py            # generate all golden values
    python generate.py conv       # generate only conv golden values
    python generate.py index      # generate only index golden values

Outputs JSON files in this directory that can be loaded by Rust tests.
"""

import json
import sys
import os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not available, using NumPy-only implementations")


def to_list(t):
    """Convert tensor/ndarray to flat list of floats."""
    if HAS_TORCH and isinstance(t, torch.Tensor):
        return t.detach().cpu().flatten().tolist()
    return np.array(t).flatten().tolist()


# ========================================================================
# Convolution
# ========================================================================

def gen_conv():
    cases = []

    if HAS_TORCH:
        rng = torch.manual_seed(42)

        # Conv1d cases
        for in_ch, out_ch, in_len, k, stride, pad, dilation in [
            (1, 1, 8, 3, 1, 0, 1),
            (2, 3, 10, 3, 1, 0, 1),
            (1, 1, 8, 3, 2, 0, 1),
            (1, 1, 10, 3, 2, 1, 2),  # dilated+strided
            (3, 4, 16, 5, 1, 0, 1),
        ]:
            x = torch.randn(1, in_ch, in_len)
            w = torch.randn(out_ch, in_ch, k)
            y = F.conv1d(x, w, stride=stride, padding=pad, dilation=dilation)
            cases.append({
                "name": f"conv1d_{in_ch}x{out_ch}_k{k}_s{stride}_p{pad}_d{dilation}",
                "op": "conv1d",
                "input": to_list(x), "weight": to_list(w), "output": to_list(y),
                "in_ch": in_ch, "out_ch": out_ch, "in_len": in_len,
                "k_size": k, "stride": stride, "padding": pad, "dilation": dilation,
            })

        # Conv2d cases
        for in_ch, out_ch, ih, iw, kh, kw, stride, pad, dilation in [
            (1, 1, 5, 5, 3, 3, 1, 0, 1),
            (3, 8, 7, 7, 3, 3, 1, 0, 1),
            (1, 1, 6, 6, 3, 3, 2, 0, 1),
            (1, 1, 8, 8, 3, 3, 1, 1, 2),  # dilated+padded
            (2, 4, 5, 7, 3, 5, 1, 0, 1),  # asymmetric
        ]:
            x = torch.randn(1, in_ch, ih, iw)
            w = torch.randn(out_ch, in_ch, kh, kw)
            y = F.conv2d(x, w, stride=stride, padding=pad, dilation=dilation)
            cases.append({
                "name": f"conv2d_{in_ch}x{out_ch}_{ih}x{iw}_k{kh}x{kw}_s{stride}",
                "op": "conv2d",
                "input": to_list(x), "weight": to_list(w), "output": to_list(y),
                "in_ch": in_ch, "out_ch": out_ch, "ih": ih, "iw": iw,
                "kh": kh, "kw": kw, "stride": stride, "padding": pad, "dilation": dilation,
            })

        # Conv3d cases
        for in_ch, out_ch, id_, ih, iw, kd, kh, kw, stride in [
            (1, 1, 4, 4, 4, 3, 3, 3, 1),
            (2, 3, 5, 5, 5, 3, 3, 3, 1),
        ]:
            x = torch.randn(1, in_ch, id_, ih, iw)
            w = torch.randn(out_ch, in_ch, kd, kh, kw)
            y = F.conv3d(x, w, stride=stride)
            cases.append({
                "name": f"conv3d_{in_ch}x{out_ch}_{id_}x{ih}x{iw}_k{kd}",
                "op": "conv3d",
                "input": to_list(x), "weight": to_list(w), "output": to_list(y),
                "in_ch": in_ch, "out_ch": out_ch,
                "id": id_, "ih": ih, "iw": iw,
                "kd": kd, "kh": kh, "kw": kw, "stride": stride,
            })

        # Depthwise conv2d
        for ch, ih, iw, kh, kw, stride in [
            (3, 5, 5, 3, 3, 1),
            (4, 7, 7, 3, 3, 2),
        ]:
            x = torch.randn(1, ch, ih, iw)
            w = torch.randn(ch, 1, kh, kw)
            y = F.conv2d(x, w, stride=stride, groups=ch)
            cases.append({
                "name": f"depthwise_{ch}ch_{ih}x{iw}_k{kh}x{kw}_s{stride}",
                "op": "depthwise_conv2d",
                "input": to_list(x), "weight": to_list(w), "output": to_list(y),
                "ch": ch, "ih": ih, "iw": iw,
                "kh": kh, "kw": kw, "stride": stride,
            })

        # Transposed conv2d
        for in_ch, out_ch, ih, iw, kh, kw, stride, pad in [
            (1, 1, 3, 3, 3, 3, 1, 0),
            (2, 3, 4, 4, 3, 3, 2, 1),
        ]:
            x = torch.randn(1, in_ch, ih, iw)
            w = torch.randn(in_ch, out_ch, kh, kw)
            y = F.conv_transpose2d(x, w, stride=stride, padding=pad)
            cases.append({
                "name": f"conv_transpose2d_{in_ch}x{out_ch}_{ih}x{iw}_s{stride}",
                "op": "conv_transpose2d",
                "input": to_list(x), "weight": to_list(w), "output": to_list(y),
                "in_ch": in_ch, "out_ch": out_ch, "ih": ih, "iw": iw,
                "kh": kh, "kw": kw, "stride": stride, "padding": pad,
            })
    else:
        print("  Skipping conv (needs PyTorch)")

    out_path = os.path.join(os.path.dirname(__file__), "conv_golden.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Wrote {len(cases)} conv cases to {out_path}")


# ========================================================================
# Index operations
# ========================================================================

def gen_index():
    cases = []
    rng = np.random.RandomState(42)

    # argmax/argmin
    for n in [8, 16, 32]:
        data = rng.randn(n).astype(np.float32)
        cases.append({
            "name": f"argmax_{n}", "op": "argmax",
            "input": to_list(data), "output": [int(np.argmax(data))],
        })
        cases.append({
            "name": f"argmin_{n}", "op": "argmin",
            "input": to_list(data), "output": [int(np.argmin(data))],
        })

    # gather
    for n in [8, 16]:
        data = rng.randn(n).astype(np.float32)
        idx = rng.permutation(n).tolist()
        out = [float(data[i]) for i in idx]
        cases.append({
            "name": f"gather_{n}", "op": "gather",
            "input": to_list(data), "index": idx, "output": out,
        })

    # scatter
    for n_src, n_dst in [(4, 8), (6, 10)]:
        src = rng.randn(n_src).astype(np.float32).tolist()
        idx = sorted(rng.choice(n_dst, n_src, replace=False).tolist())
        out = [0.0] * n_dst
        for i, ix in enumerate(idx):
            out[ix] = src[i]
        cases.append({
            "name": f"scatter_{n_src}to{n_dst}", "op": "scatter",
            "input": src, "index": idx, "output": out, "n_dst": n_dst,
        })

    # scatter_add
    src = rng.randn(6).astype(np.float32).tolist()
    idx = [0, 1, 0, 2, 1, 0]
    out = [0.0] * 3
    for i, ix in enumerate(idx):
        out[ix] += src[i]
    cases.append({
        "name": "scatter_add_6to3", "op": "scatter_add",
        "input": src, "index": idx, "output": out, "n_dst": 3,
    })

    # embedding
    weight = rng.randn(5, 3).astype(np.float32)
    indices = [3, 0, 4, 1]
    out = weight[indices].flatten().tolist()
    cases.append({
        "name": "embedding_5x3", "op": "embedding",
        "weight": to_list(weight), "index": indices, "output": out,
        "embed_dim": 3,
    })

    # index_select (row-based)
    data = rng.randn(6, 4).astype(np.float32)
    idx = [3, 1, 5]
    out = data[idx].flatten().tolist()
    cases.append({
        "name": "index_select_6x4", "op": "index_select",
        "input": to_list(data), "index": idx, "output": out,
        "inner_dim": 4,
    })

    # masked_fill
    data = rng.randn(8).astype(np.float32).tolist()
    mask = [True, False, True, True, False, False, True, False]
    fill_val = -999.0
    out = [fill_val if m else v for v, m in zip(data, mask)]
    cases.append({
        "name": "masked_fill_8", "op": "masked_fill",
        "input": data, "mask": mask, "output": out, "fill_value": fill_val,
    })

    out_path = os.path.join(os.path.dirname(__file__), "index_golden.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Wrote {len(cases)} index cases to {out_path}")


# ========================================================================
# Pooling
# ========================================================================

def gen_pooling():
    cases = []

    if HAS_TORCH:
        rng = torch.manual_seed(42)

        # 1D pooling
        for n, k, stride in [(10, 3, 1), (12, 4, 2), (16, 3, 3)]:
            x = torch.randn(1, 1, n)
            y_max = F.max_pool1d(x, k, stride=stride)
            y_avg = F.avg_pool1d(x, k, stride=stride)
            cases.append({
                "name": f"max_pool1d_{n}_k{k}_s{stride}", "op": "max_pool1d",
                "input": to_list(x), "output": to_list(y_max),
                "in_len": n, "k_size": k, "stride": stride,
            })
            cases.append({
                "name": f"avg_pool1d_{n}_k{k}_s{stride}", "op": "avg_pool1d",
                "input": to_list(x), "output": to_list(y_avg),
                "in_len": n, "k_size": k, "stride": stride,
            })

        # 2D pooling
        for ch, ih, iw, k, stride in [(1, 6, 6, 2, 2), (3, 8, 8, 3, 2)]:
            x = torch.randn(1, ch, ih, iw)
            y_max = F.max_pool2d(x, k, stride=stride)
            y_avg = F.avg_pool2d(x, k, stride=stride)
            cases.append({
                "name": f"max_pool2d_{ch}ch_{ih}x{iw}_k{k}_s{stride}", "op": "max_pool2d",
                "input": to_list(x), "output": to_list(y_max),
                "ch": ch, "ih": ih, "iw": iw, "kh": k, "kw": k, "stride": stride,
            })
            cases.append({
                "name": f"avg_pool2d_{ch}ch_{ih}x{iw}_k{k}_s{stride}", "op": "avg_pool2d",
                "input": to_list(x), "output": to_list(y_avg),
                "ch": ch, "ih": ih, "iw": iw, "kh": k, "kw": k, "stride": stride,
            })

        # 3D pooling
        x = torch.randn(1, 1, 4, 4, 4)
        y_max = F.max_pool3d(x, 2, stride=2)
        y_avg = F.avg_pool3d(x, 2, stride=2)
        cases.append({
            "name": "max_pool3d_4x4x4_k2_s2", "op": "max_pool3d",
            "input": to_list(x), "output": to_list(y_max),
            "ch": 1, "id": 4, "ih": 4, "iw": 4,
            "kd": 2, "kh": 2, "kw": 2, "stride": 2,
        })
        cases.append({
            "name": "avg_pool3d_4x4x4_k2_s2", "op": "avg_pool3d",
            "input": to_list(x), "output": to_list(y_avg),
            "ch": 1, "id": 4, "ih": 4, "iw": 4,
            "kd": 2, "kh": 2, "kw": 2, "stride": 2,
        })
    else:
        print("  Skipping pooling (needs PyTorch)")

    out_path = os.path.join(os.path.dirname(__file__), "pooling_golden.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Wrote {len(cases)} pooling cases to {out_path}")


# ========================================================================
# Matmul variants
# ========================================================================

def gen_matmul():
    cases = []
    rng = np.random.RandomState(42)

    # Transposed A: C = A^T @ B, A stored as (k x m)
    for m, k, n in [(3, 4, 2), (5, 3, 4), (2, 6, 3)]:
        a = rng.randn(k, m).astype(np.float32)
        b = rng.randn(k, n).astype(np.float32)
        c = a.T @ b  # (m x n)
        cases.append({
            "name": f"transA_{m}x{k}x{n}", "op": "transposed_a",
            "a": to_list(a), "b": to_list(b), "output": to_list(c),
            "m": m, "k": k, "n": n,
        })

    # Transposed B: C = A @ B^T, B stored as (n x k)
    for m, k, n in [(3, 4, 2), (2, 5, 3)]:
        a = rng.randn(m, k).astype(np.float32)
        b = rng.randn(n, k).astype(np.float32)
        c = a @ b.T  # (m x n)
        cases.append({
            "name": f"transB_{m}x{k}x{n}", "op": "transposed_b",
            "a": to_list(a), "b": to_list(b), "output": to_list(c),
            "m": m, "k": k, "n": n,
        })

    # Transposed both: C = A^T @ B^T, A=(k x m), B=(n x k)
    for m, k, n in [(3, 4, 2), (2, 3, 4)]:
        a = rng.randn(k, m).astype(np.float32)
        b = rng.randn(n, k).astype(np.float32)
        c = a.T @ b.T  # (m x n)
        cases.append({
            "name": f"transBoth_{m}x{k}x{n}", "op": "transposed_both",
            "a": to_list(a), "b": to_list(b), "output": to_list(c),
            "m": m, "k": k, "n": n,
        })

    # Lower triangular: C = tril(A) @ B
    for n_ in [3, 4, 5]:
        a = rng.randn(n_, n_).astype(np.float32)
        b = rng.randn(n_, n_).astype(np.float32)
        a_tril = np.tril(a)
        c = a_tril @ b
        cases.append({
            "name": f"lower_tri_{n_}", "op": "lower_triangular",
            "a": to_list(a), "b": to_list(b), "output": to_list(c),
            "m": n_, "k": n_, "n": n_,
        })

    # Upper triangular: C = triu(A) @ B
    for n_ in [3, 4, 5]:
        a = rng.randn(n_, n_).astype(np.float32)
        b = rng.randn(n_, n_).astype(np.float32)
        a_triu = np.triu(a)
        c = a_triu @ b
        cases.append({
            "name": f"upper_tri_{n_}", "op": "upper_triangular",
            "a": to_list(a), "b": to_list(b), "output": to_list(c),
            "m": n_, "k": n_, "n": n_,
        })

    out_path = os.path.join(os.path.dirname(__file__), "matmul_golden.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Wrote {len(cases)} matmul cases to {out_path}")


# ========================================================================
# Resize / interpolation
# ========================================================================

def gen_resize():
    cases = []

    if HAS_TORCH:
        rng = torch.manual_seed(42)

        # Bilinear upsample 2D
        for ch, ih, iw, oh, ow in [(1, 3, 3, 5, 5), (2, 4, 4, 8, 8), (1, 2, 2, 4, 4)]:
            x = torch.randn(1, ch, ih, iw)
            y = F.interpolate(x, size=(oh, ow), mode='bilinear', align_corners=True)
            cases.append({
                "name": f"bilinear_{ch}ch_{ih}x{iw}_to_{oh}x{ow}", "op": "bilinear_upsample_2d",
                "input": to_list(x), "output": to_list(y),
                "ch": ch, "ih": ih, "iw": iw, "oh": oh, "ow": ow,
            })

        # Nearest upsample 2D
        for ch, ih, iw, oh, ow in [(1, 3, 3, 6, 6), (2, 2, 2, 4, 4)]:
            x = torch.randn(1, ch, ih, iw)
            y = F.interpolate(x, size=(oh, ow), mode='nearest')
            cases.append({
                "name": f"nearest_{ch}ch_{ih}x{iw}_to_{oh}x{ow}", "op": "nearest_upsample_2d",
                "input": to_list(x), "output": to_list(y),
                "ch": ch, "ih": ih, "iw": iw, "oh": oh, "ow": ow,
            })

        # Trilinear upsample 3D
        x = torch.randn(1, 1, 2, 2, 2)
        y = F.interpolate(x, size=(4, 4, 4), mode='trilinear', align_corners=True)
        cases.append({
            "name": "trilinear_1ch_2x2x2_to_4x4x4", "op": "trilinear_upsample_3d",
            "input": to_list(x), "output": to_list(y),
            "ch": 1, "id": 2, "ih": 2, "iw": 2, "od": 4, "oh": 4, "ow": 4,
        })

        # Downsample bilinear 2D
        for ch, ih, iw, oh, ow in [(1, 6, 6, 3, 3), (1, 8, 8, 4, 4)]:
            x = torch.randn(1, ch, ih, iw)
            y = F.interpolate(x, size=(oh, ow), mode='bilinear', align_corners=True)
            cases.append({
                "name": f"downsample_bilinear_{ch}ch_{ih}x{iw}_to_{oh}x{ow}",
                "op": "downsample_bilinear_2d",
                "input": to_list(x), "output": to_list(y),
                "ch": ch, "ih": ih, "iw": iw, "oh": oh, "ow": ow,
            })
    else:
        print("  Skipping resize (needs PyTorch)")

    out_path = os.path.join(os.path.dirname(__file__), "resize_golden.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Wrote {len(cases)} resize cases to {out_path}")


# ========================================================================
# Misc: broadcast, math, loss, optimizer
# ========================================================================

def gen_misc():
    cases = []
    rng = np.random.RandomState(42)

    # where_broadcast
    for n in [8, 16]:
        x = rng.randn(n).astype(np.float32)
        y = rng.randn(n).astype(np.float32)
        mask = rng.randint(0, 2, n).astype(bool)
        out = np.where(mask, x, y).tolist()
        cases.append({
            "name": f"where_{n}", "op": "where_broadcast",
            "x": x.tolist(), "y": y.tolist(), "mask": mask.tolist(), "output": out,
        })

    # logic_and_broadcast
    for n in [8]:
        a = rng.randn(n).astype(np.float32)
        b = rng.randn(n).astype(np.float32)
        out = [1.0 if (av != 0 and bv != 0) else 0.0 for av, bv in zip(a.tolist(), b.tolist())]
        cases.append({
            "name": f"logic_and_{n}", "op": "logic_and_broadcast",
            "a": a.tolist(), "b": b.tolist(), "output": out,
        })

    # power_broadcast
    base = np.abs(rng.randn(8).astype(np.float32)) + 0.1  # positive bases
    exp = rng.randn(8).astype(np.float32)
    out = np.power(base, exp).tolist()
    cases.append({
        "name": "power_8", "op": "power_broadcast",
        "base": base.tolist(), "exp": exp.tolist(), "output": out,
    })

    # masked_cumsum
    for n in [8, 12]:
        data = rng.randn(n).astype(np.float32)
        mask = rng.randint(0, 2, n).astype(bool)
        masked = data * mask
        out = np.cumsum(masked).tolist()
        cases.append({
            "name": f"masked_cumsum_{n}", "op": "masked_cumsum",
            "input": data.tolist(), "mask": mask.tolist(), "output": out,
        })

    # triplet_margin_loss
    for dim in [4, 8]:
        a = rng.randn(dim).astype(np.float32)
        p = rng.randn(dim).astype(np.float32)
        n_ = rng.randn(dim).astype(np.float32)
        margin = 1.0
        dp = float(np.sum((a - p) ** 2))
        dn = float(np.sum((a - n_) ** 2))
        loss = max(0.0, dp - dn + margin)
        cases.append({
            "name": f"triplet_loss_{dim}", "op": "triplet_margin_loss",
            "anchor": a.tolist(), "positive": p.tolist(), "negative": n_.tolist(),
            "margin": margin, "output": [loss],
        })

    # lamb_update
    for dim in [4]:
        param = rng.randn(dim).astype(np.float32)
        grad = rng.randn(dim).astype(np.float32)
        m = np.zeros(dim, dtype=np.float32)
        v = np.zeros(dim, dtype=np.float32)
        lr, beta1, beta2, eps = 0.01, 0.9, 0.999, 1e-8

        # one step
        m_new = (1 - beta1) * grad
        v_new = (1 - beta2) * (grad ** 2)
        m_hat = m_new / (1 - beta1)
        v_hat = v_new / (1 - beta2)
        update = m_hat / (np.sqrt(v_hat) + eps)

        w_norm = float(np.linalg.norm(param))
        u_norm = float(np.linalg.norm(update))
        trust = w_norm / u_norm if (w_norm > 0 and u_norm > 0) else 1.0

        param_new = param - lr * trust * update

        cases.append({
            "name": f"lamb_{dim}", "op": "lamb_update",
            "param": param.tolist(), "grad": grad.tolist(),
            "param_out": param_new.tolist(), "m_out": m_new.tolist(), "v_out": v_new.tolist(),
            "lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps,
        })

    out_path = os.path.join(os.path.dirname(__file__), "misc_golden.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  Wrote {len(cases)} misc cases to {out_path}")


# ========================================================================
# Main
# ========================================================================

GENERATORS = {
    "conv": gen_conv,
    "index": gen_index,
    "pooling": gen_pooling,
    "matmul": gen_matmul,
    "resize": gen_resize,
    "misc": gen_misc,
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(GENERATORS.keys())
    for name in targets:
        if name not in GENERATORS:
            print(f"Unknown target: {name}. Available: {list(GENERATORS.keys())}")
            sys.exit(1)
        print(f"Generating {name}...")
        GENERATORS[name]()
    print("Done.")
