#include "kernel_operator.h"

// Parameterized vec_add kernel with multi-block and tiling.
// len_buf contains per-block element count. Each AICore block processes
// its slice starting at GetBlockIdx() * per_block_len.
constexpr int32_t TILE_SIZE = 256;

class KernelVecAdd {
public:
    __aicore__ inline KernelVecAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf)
    {
        uint32_t perBlockLen = *((__gm__ uint32_t *)len_buf);
        uint32_t offset = AscendC::GetBlockIdx() * perBlockLen;
        totalLen = perBlockLen;
        xGm.SetGlobalBuffer((__gm__ half *)x + offset, perBlockLen);
        yGm.SetGlobalBuffer((__gm__ half *)y + offset, perBlockLen);
        zGm.SetGlobalBuffer((__gm__ half *)z + offset, perBlockLen);
        // Align tile size to 32-byte boundary (16 half elements)
        uint32_t alignedTile = ((TILE_SIZE + 15) / 16) * 16;
        pipe.InitBuffer(inQueueX, 1, alignedTile * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, alignedTile * sizeof(half));
        pipe.InitBuffer(outQueueZ, 1, alignedTile * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t tileCount = (totalLen + TILE_SIZE - 1) / TILE_SIZE;
        for (int32_t i = 0; i < tileCount; i++) {
            int32_t offset = i * TILE_SIZE;
            int32_t len = TILE_SIZE;
            if (offset + len > totalLen) {
                len = totalLen - offset;
            }
            // Align to 16 elements for DMA
            int32_t alignedLen = ((len + 15) / 16) * 16;
            CopyIn(offset, alignedLen);
            Compute(alignedLen);
            CopyOut(offset, alignedLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t len)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t len)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t offset, int32_t len)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[offset], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    int32_t totalLen;
};

extern "C" __global__ __aicore__ void vec_add_bench(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR len_buf)
{
    KernelVecAdd op;
    op.Init(x, y, z, len_buf);
    op.Process();
}

extern "C" void vec_add_bench_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *len_buf)
{
    vec_add_bench<<<blockDim, nullptr, stream>>>(x, y, z, len_buf);
}
