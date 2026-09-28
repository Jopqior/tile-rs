// Case 5: Double-Free of Tensor Buffers
//
// AscendC's FreeTensor() returns a LocalTensor's SRAM buffer to the
// queue's internal free list. Calling FreeTensor() twice on the same
// tensor inserts the same buffer address into the free list twice.
// The next two AllocTensor() calls will both return the same buffer,
// causing two "different" tensors to alias the same SRAM region.
//
// This manifests as intermittent data corruption: one tensor's
// computation silently overwrites another's data. The bug is
// tile-count-dependent — it only triggers when enough allocations
// happen to consume the duplicated free list entry.

#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 256;

class KernelVecAddDoubleFree {
public:
    __aicore__ inline KernelVecAddDoubleFree() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf) {
        uint32_t perBlockLen = *((__gm__ uint32_t *)len_buf);
        uint32_t offset = AscendC::GetBlockIdx() * perBlockLen;
        xGm.SetGlobalBuffer((__gm__ half *)x + offset, perBlockLen);
        yGm.SetGlobalBuffer((__gm__ half *)y + offset, perBlockLen);
        zGm.SetGlobalBuffer((__gm__ half *)z + offset, perBlockLen);

        uint32_t alignedTile = ((TILE_SIZE + 15) / 16) * 16;
        pipe.InitBuffer(inQueueX, 2, alignedTile * sizeof(half));
        pipe.InitBuffer(inQueueY, 2, alignedTile * sizeof(half));
        pipe.InitBuffer(outQueueZ, 2, alignedTile * sizeof(half));
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

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        outQueueZ.EnQue<half>(zLocal);

        // BUG: Copy-paste error from a refactoring — FreeTensor called again.
        // xLocal's buffer is now in inQueueX's free list TWICE.
        // On the next two tile iterations, AllocTensor will return the same
        // buffer address for two "different" tensors, causing them to alias.
        // One tile's DMA load will silently overwrite another tile's data.
        inQueueX.FreeTensor(xLocal);  // double-free! Corrupts queue free list
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t len) {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    int32_t totalLen;
};

extern "C" __global__ __aicore__ void vec_add_double_free(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf) {
    KernelVecAddDoubleFree op;
    op.Init(x, y, z, len_buf);
    op.Process();
}
