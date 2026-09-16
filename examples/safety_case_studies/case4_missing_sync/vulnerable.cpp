// Case 4: Missing Synchronization Between Pipeline Stages
//
// Ascend NPUs execute DMA (MTE2/MTE3), vector (V), and scalar (S)
// pipelines concurrently. A pipe_barrier() is required between a DMA
// load and a subsequent vector/scalar operation to ensure the data has
// actually arrived in local SRAM before computation begins.
//
// Forgetting this barrier is the single most common NPU bug. The kernel
// compiles and runs without error, but the vector unit reads whatever
// was previously in SRAM (stale data or zeros) instead of the loaded
// input. The result is silently wrong output.
//
// The AscendC queue model (EnQue/DeQue) is meant to handle this, but
// it only works correctly when the queue depth matches the pipeline
// structure. Direct DataCopy without proper queue usage — a common
// pattern in simple kernels — requires explicit barriers.

#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 256;

class KernelSigmoidNoSync {
public:
    __aicore__ inline KernelSigmoidNoSync() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf) {
        uint32_t n = *((__gm__ uint32_t *)len_buf);
        inputGm.SetGlobalBuffer((__gm__ float *)input, n);
        outputGm.SetGlobalBuffer((__gm__ float *)output, n);

        uint32_t alignedTile = ((TILE_SIZE + 7) / 8) * 8;
        pipe.InitBuffer(inQueue, 1, alignedTile * sizeof(float));
        pipe.InitBuffer(outQueue, 1, alignedTile * sizeof(float));
        totalLen = n;
    }

    __aicore__ inline void Process() {
        int32_t tileCount = (totalLen + TILE_SIZE - 1) / TILE_SIZE;
        for (int32_t i = 0; i < tileCount; i++) {
            int32_t offset = i * TILE_SIZE;
            int32_t len = TILE_SIZE;
            if (offset + len > totalLen) len = totalLen - offset;
            int32_t alignedLen = ((len + 7) / 8) * 8;
            CopyIn(offset, alignedLen);
            Compute(alignedLen);
            CopyOut(offset, alignedLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t len) {
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(xLocal, inputGm[offset], len);
        // BUG: Missing pipe_barrier() between DMA load and EnQue.
        // The EnQue only marks the tensor as "available" in the queue,
        // but does NOT ensure the DMA transfer has completed.
        // If the DMA pipeline (MTE2) is slower than the scalar pipeline (S),
        // the subsequent DeQue + vector operations will read stale SRAM data.
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t len) {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        // Sigmoid = 1 / (1 + exp(-x))
        // Each of these vector operations may execute before the DMA load
        // completes, reading uninitialized or stale data from SRAM.
        AscendC::Muls(yLocal, xLocal, -1.0f, len);       // -x (may read stale data)
        AscendC::Exp(yLocal, yLocal, len);                // exp(-x)
        AscendC::Adds(yLocal, yLocal, 1.0f, len);         // 1 + exp(-x)
        AscendC::Reciprocal(yLocal, yLocal, len);          // 1 / (1 + exp(-x))

        outQueue.EnQue<float>(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t offset, int32_t len) {
        AscendC::LocalTensor<float> yLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[offset], yLocal, len);
        outQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<float> inputGm, outputGm;
    int32_t totalLen;
};

extern "C" __global__ __aicore__ void sigmoid_no_sync(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf) {
    KernelSigmoidNoSync op;
    op.Init(input, output, len_buf);
    op.Process();
}
