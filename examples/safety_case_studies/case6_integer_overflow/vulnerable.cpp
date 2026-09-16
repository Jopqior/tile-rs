// Case 6: Silent Integer Overflow in Multi-Block Offset Calculation
//
// Multi-block kernels distribute work across NPU cores by computing
// offset = blockIdx * perBlockLen. With uint32_t arithmetic, this
// multiplication silently wraps on overflow. For large tensors
// (e.g., blockIdx=8192, perBlockLen=524288), the product exceeds
// 2^32 and wraps to a small value — causing the kernel to read/write
// from the wrong memory region.
//
// In C/C++, unsigned overflow is defined behavior (wraps modulo 2^32),
// so no warning or error is generated. The kernel silently computes
// on the wrong data and produces garbage output. If two blocks wrap
// to the same offset, they will race on the same memory region.

#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 256;

class KernelVecAddOverflow {
public:
    __aicore__ inline KernelVecAddOverflow() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf) {
        uint32_t perBlockLen = *((__gm__ uint32_t *)len_buf);

        // BUG: Silent uint32_t overflow when blockIdx * perBlockLen > 2^32.
        //
        // Example: With 8192 blocks and perBlockLen = 524288 (512K elements),
        // total tensor size is 4GB of half-precision data. Block 8192 computes:
        //   offset = 8192 * 524288 = 4294967296 = 0x100000000
        // But uint32_t wraps: offset = 0. This block now aliases block 0's data.
        //
        // C++ provides no warning — unsigned overflow is well-defined as modular
        // arithmetic. The kernel silently reads the wrong data.
        uint32_t offset = AscendC::GetBlockIdx() * perBlockLen;

        xGm.SetGlobalBuffer((__gm__ half *)x + offset, perBlockLen);
        yGm.SetGlobalBuffer((__gm__ half *)y + offset, perBlockLen);
        zGm.SetGlobalBuffer((__gm__ half *)z + offset, perBlockLen);

        uint32_t alignedTile = ((TILE_SIZE + 15) / 16) * 16;
        pipe.InitBuffer(inQueueX, 1, alignedTile * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, alignedTile * sizeof(half));
        pipe.InitBuffer(outQueueZ, 1, alignedTile * sizeof(half));
        totalLen = perBlockLen;
    }

    __aicore__ inline void Process() {
        int32_t tileCount = (totalLen + TILE_SIZE - 1) / TILE_SIZE;
        for (int32_t i = 0; i < tileCount; i++) {
            int32_t off = i * TILE_SIZE;
            int32_t len = TILE_SIZE;
            if (off + len > totalLen) len = totalLen - off;
            int32_t alignedLen = ((len + 15) / 16) * 16;
            CopyIn(off, alignedLen);
            Compute(alignedLen);
            CopyOut(off, alignedLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t len) {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t len) {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t len) {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    int32_t totalLen;
};

extern "C" __global__ __aicore__ void vec_add_overflow(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf) {
    KernelVecAddOverflow op;
    op.Init(x, y, z, len_buf);
    op.Process();
}
