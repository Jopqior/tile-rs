#!/usr/bin/env python3
"""Softmax Benchmark on Ascend NPU (aclnn baseline).

Measures softmax using aclnnSoftmax API on Ascend 910B.
Shapes match ascend-rs tile submissions for head-to-head comparison.

Requirements:
  - CANN 8.5+ with aclnn API
  - Ascend NPU device (tested on 910B2)
  - numpy

Usage:
  python3 bench_softmax_ascend.py [--csv results.csv]

Environment:
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
"""

import ctypes
import numpy as np
import sys
import os
import time

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

CANN_PATH = os.environ.get("ASCEND_HOME_PATH",
    os.environ.get("ACLRS_CANN_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))

lib_path = os.path.join(CANN_PATH, "lib64")
libascendcl = ctypes.CDLL(os.path.join(lib_path, "libascendcl.so"))
libopapi = ctypes.CDLL(os.path.join(lib_path, "libopapi.so"))

# ---------------------------------------------------------------------------
# ACL type definitions
# ---------------------------------------------------------------------------

ACL_FLOAT = 0
ACL_FORMAT_ND = 2
aclDataType = ctypes.c_int
aclFormat = ctypes.c_int

# ---------------------------------------------------------------------------
# ACL function signatures
# ---------------------------------------------------------------------------

libascendcl.aclInit.argtypes = [ctypes.c_char_p]
libascendcl.aclInit.restype = ctypes.c_int
libascendcl.aclrtSetDevice.argtypes = [ctypes.c_int]
libascendcl.aclrtSetDevice.restype = ctypes.c_int
libascendcl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
libascendcl.aclrtMalloc.restype = ctypes.c_int
libascendcl.aclrtFree.argtypes = [ctypes.c_void_p]
libascendcl.aclrtFree.restype = ctypes.c_int
libascendcl.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libascendcl.aclrtMemcpy.restype = ctypes.c_int
libascendcl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
libascendcl.aclrtCreateStream.restype = ctypes.c_int
libascendcl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
libascendcl.aclrtSynchronizeStream.restype = ctypes.c_int
libascendcl.aclrtCreateEvent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
libascendcl.aclrtCreateEvent.restype = ctypes.c_int
libascendcl.aclrtRecordEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libascendcl.aclrtRecordEvent.restype = ctypes.c_int
libascendcl.aclrtEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
libascendcl.aclrtEventElapsedTime.restype = ctypes.c_int

# aclnn tensor
libopapi.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
    aclDataType, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
    aclFormat, ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
    ctypes.c_void_p
]
libopapi.aclCreateTensor.restype = ctypes.c_void_p

# aclnnSoftmax
libopapi.aclnnSoftmaxGetWorkspaceSize.argtypes = [
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p),
]
libopapi.aclnnSoftmaxGetWorkspaceSize.restype = ctypes.c_int
libopapi.aclnnSoftmax.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
]
libopapi.aclnnSoftmax.restype = ctypes.c_int

# ---------------------------------------------------------------------------
# Configuration — shapes match ascend-rs tile submissions
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 1024),
    (1, 4096),
    (16, 1024),
    (16, 4096),
    (64, 1024),
    (64, 4096),
    (256, 1024),
    (256, 4096),
    (1024, 1024),
    (1024, 4096),
    (4096, 1024),
    (4096, 4096),
]

WARMUP = 3
TIMED = 20
SOFTMAX_DIM = -1  # last dimension

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

ret = libascendcl.aclInit(None)
assert ret == 0, f"aclInit failed: {ret}"
ret = libascendcl.aclrtSetDevice(0)
assert ret == 0, f"aclrtSetDevice failed: {ret}"

stream = ctypes.c_void_p()
ret = libascendcl.aclrtCreateStream(ctypes.byref(stream))
assert ret == 0, f"aclrtCreateStream failed: {ret}"

print("Ascend NPU initialized")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEM_MALLOC_HUGE_FIRST = 0


def acl_malloc(size):
    ptr = ctypes.c_void_p()
    ret = libascendcl.aclrtMalloc(ctypes.byref(ptr), size, ACL_MEM_MALLOC_HUGE_FIRST)
    assert ret == 0, f"aclrtMalloc failed: {ret}"
    return ptr


def acl_h2d(dev_ptr, host_arr):
    ret = libascendcl.aclrtMemcpy(dev_ptr, host_arr.nbytes,
        host_arr.ctypes.data_as(ctypes.c_void_p), host_arr.nbytes, ACL_MEMCPY_HOST_TO_DEVICE)
    assert ret == 0


def make_tensor(shape, dev_ptr, dtype=ACL_FLOAT):
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    strides = []
    s = 1
    for d in reversed(shape):
        strides.insert(0, s)
        s *= d
    strides_arr = (ctypes.c_int64 * ndim)(*strides)
    storage_shape = (ctypes.c_int64 * ndim)(*shape)
    return libopapi.aclCreateTensor(shape_arr, ndim, dtype,
        strides_arr, 0, ACL_FORMAT_ND, storage_shape, ndim, dev_ptr)


def bench_softmax(rows, cols):
    """Benchmark softmax using aclnnSoftmax."""
    elements = rows * cols
    byte_size = elements * 4

    x_host = np.random.randn(rows, cols).astype(np.float32)
    x_dev = acl_malloc(byte_size)
    out_dev = acl_malloc(byte_size)
    acl_h2d(x_dev, x_host)

    x_t = make_tensor([rows, cols], x_dev)
    out_t = make_tensor([rows, cols], out_dev)

    get_ws = libopapi.aclnnSoftmaxGetWorkspaceSize
    run_fn = libopapi.aclnnSoftmax

    # Initial workspace query
    ws_size = ctypes.c_uint64()
    executor = ctypes.c_void_p()
    dim = ctypes.c_int64(SOFTMAX_DIM)
    ret = get_ws(x_t, dim, out_t, ctypes.byref(ws_size), ctypes.byref(executor))
    assert ret == 0, f"aclnnSoftmaxGetWorkspaceSize failed: {ret}"

    ws_ptr = acl_malloc(max(ws_size.value, 16)) if ws_size.value > 0 else ctypes.c_void_p(0)

    # Create events
    ev_start = ctypes.c_void_p()
    ev_end = ctypes.c_void_p()
    libascendcl.aclrtCreateEvent(ctypes.byref(ev_start))
    libascendcl.aclrtCreateEvent(ctypes.byref(ev_end))

    # Warmup
    for _ in range(WARMUP):
        executor2 = ctypes.c_void_p()
        ws_size2 = ctypes.c_uint64()
        get_ws(x_t, dim, out_t, ctypes.byref(ws_size2), ctypes.byref(executor2))
        run_fn(ws_ptr, ws_size2, executor2, stream)
        libascendcl.aclrtSynchronizeStream(stream)

    # Timed runs
    times = []
    for _ in range(TIMED):
        executor2 = ctypes.c_void_p()
        ws_size2 = ctypes.c_uint64()
        get_ws(x_t, dim, out_t, ctypes.byref(ws_size2), ctypes.byref(executor2))

        libascendcl.aclrtRecordEvent(ev_start, stream)
        run_fn(ws_ptr, ws_size2, executor2, stream)
        libascendcl.aclrtRecordEvent(ev_end, stream)
        libascendcl.aclrtSynchronizeStream(stream)
        elapsed = ctypes.c_float()
        libascendcl.aclrtEventElapsedTime(ctypes.byref(elapsed), ev_start, ev_end)
        times.append(elapsed.value * 1000)  # ms -> us

    times.sort()
    import statistics
    median = statistics.median(times)
    # softmax: read input + write output = 2 * elements * 4 bytes
    gbps = 2.0 * elements * 4.0 / (median * 1e-6) / 1e9

    libascendcl.aclrtFree(x_dev)
    libascendcl.aclrtFree(out_dev)

    return median, min(times), max(times), gbps


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

results = []
print(f"\n{'Shape':>18s} {'Median(us)':>10s} {'Min(us)':>10s} {'Max(us)':>10s} {'GB/s':>10s}")
print("-" * 60)

for rows, cols in SHAPES:
    label = f"({rows},{cols})"
    try:
        median, min_t, max_t, gbps = bench_softmax(rows, cols)
        print(f"{label:>18s} {median:10.1f} {min_t:10.1f} {max_t:10.1f} {gbps:10.1f}")
        results.append((rows, cols, label, median, min_t, max_t, gbps, "PASS"))
    except Exception as e:
        print(f"{label:>18s} FAIL  {e}")
        results.append((rows, cols, label, None, None, None, None, f"FAIL: {e}"))

# ---------------------------------------------------------------------------
# Submission CSV
# ---------------------------------------------------------------------------

sub_lines = [
    "device_id,kernel_id,dtype,input_shape,batch_size,impl_lang,latency_us,throughput_gbps,driver_version,toolchain,git_sha,submitter"
]
for rows, cols, label, median, min_t, max_t, gbps, status in results:
    if median is None:
        continue
    shape_str = f"[{rows}, {cols}]"
    elements = rows * cols
    med_gbps = 2.0 * elements * 4.0 / (median * 1e-6) / 1e9
    sub_lines.append(
        f'huawei-910b,softmax,f32,"{shape_str}",1,aclnn,{median:.1f},{med_gbps:.1f},'
        f'CANN 8.5,aclnnSoftmax,,bench_softmax_ascend.py'
    )

submission_csv = "\n".join(sub_lines) + "\n"
submission_file = "huawei-910b_softmax_aclnn_submission.csv"
with open(submission_file, "w") as f:
    f.write(submission_csv)
print(f"\nSubmission CSV written to {submission_file}")

# ---------------------------------------------------------------------------
# Head-to-head comparison table (if ascend-rs tile data available)
# ---------------------------------------------------------------------------

tile_data = {
    (1, 1024): 6.160,
    (1, 4096): 7.840,
    (16, 1024): 5.500,
    (16, 4096): 6.680,
    (64, 1024): 10.920,
    (64, 4096): 12.960,
    (256, 1024): 29.380,
    (256, 4096): 28.860,
    (1024, 1024): 104.300,
    (1024, 4096): 109.320,
    (4096, 1024): 430.640,
    (4096, 4096): 359.300,
}

print(f"\n{'Shape':>18s} {'ascend-rs(us)':>14s} {'aclnn(us)':>10s} {'ratio':>8s}")
print("-" * 55)
for rows, cols, label, median, min_t, max_t, gbps, status in results:
    if median is None:
        continue
    tile_us = tile_data.get((rows, cols))
    if tile_us is not None:
        ratio = tile_us / median
        print(f"{label:>18s} {tile_us:14.1f} {median:10.1f} {ratio:7.2f}x")
    else:
        print(f"{label:>18s} {'--':>14s} {median:10.1f} {'--':>8s}")

print("\nratio > 1.0 = ascend-rs slower than aclnn baseline")
print("ratio < 1.0 = ascend-rs faster than aclnn baseline")
