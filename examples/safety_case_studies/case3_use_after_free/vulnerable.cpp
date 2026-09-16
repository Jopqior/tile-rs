// Case 3: Use-After-Free of LocalTensor
//
// AscendC requires manual FreeTensor() calls to return SRAM buffers to
// the queue's free pool. After FreeTensor(), the LocalTensor handle
// remains valid at the C++ type level — it still holds the original
// buffer address. Any subsequent GetValue() or SetValue() compiles and
// runs, but reads/writes memory that has been returned to the pool and
// may already be reallocated for a different tensor.
//
// This is particularly insidious in tiled kernels where the freed buffer
// is reallocated in the next tile iteration, causing intermittent data
// corruption that depends on tile count and allocation order.

#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 256;

class KernelVecAddUAF {
public:
    __aicore__ inline KernelVecAddUAF() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf) {
        uint32_t perBlockLen = *((__gm__ uint32_t *)len_buf);
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
            int32_t offset = i * TILE_SIZE;
            int32_t len = TILE_SIZE;
            if (offset + len > totalLen) len = totalLen - offset;
            int32_t alignedLen = ((len + 15) / 16) * 16;
            CopyIn(offset, alignedLen);
            Compute(alignedLen);
            CopyOut(offset, alignedLen);
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

        // Return buffers to the free pool
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);

        // BUG: xLocal was freed above, but the C++ handle still compiles.
        // The SRAM region has been returned to inQueueX's free list.
        // In a multi-tile kernel, this buffer may already be reallocated
        // by the next iteration's AllocTensor() call.
        // Reading from it returns stale or corrupted data.
        half check = xLocal.GetValue(0);  // use-after-free!

        // The stale value may cause incorrect control flow decisions,
        // silent data corruption, or NPU exceptions — depending on
        // whether the buffer has been reallocated yet.
        if ((float)check > 100.0f) {
            // Branch taken based on garbage data
            AscendC::Muls(zLocal, zLocal, (half)0.5f, len);
        }

        outQueueZ.EnQue<half>(zLocal);
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

extern "C" __global__ __aicore__ void vec_add_uaf(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf) {
    KernelVecAddUAF op;
    op.Init(x, y, z, len_buf);
    op.Process();
}
