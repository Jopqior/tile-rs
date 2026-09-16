// Case 2: Buffer Overflow via Unchecked Tensor Index
//
// AscendC's GetValue(i) and SetValue(i, v) perform no bounds checking.
// If the loop bound is wrong — using the wrong length variable, an
// off-by-one error, or confusing input/output sizes — the kernel reads
// or writes out of bounds on local SRAM. This corrupts adjacent tensor
// data in the unified buffer or triggers an NPU exception.
//
// This is especially dangerous because local SRAM is shared across all
// tensor allocations within a tile — an OOB write silently overwrites
// a neighboring tensor's data.

#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 256;

class KernelScalarSoftmax {
public:
    __aicore__ inline KernelScalarSoftmax() {}

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
            Compute(len, alignedLen);
            CopyOut(offset, alignedLen);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t offset, int32_t len) {
        AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(xLocal, inputGm[offset], len);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t len, int32_t alignedLen) {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        // Step 1: Find max (scalar loop)
        float maxVal = xLocal.GetValue(0);
        for (int32_t i = 1; i < len; i++) {
            float v = xLocal.GetValue(i);
            if (v > maxVal) maxVal = v;
        }

        // Step 2: Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int32_t i = 0; i < len; i++) {
            float v = xLocal.GetValue(i) - maxVal;
            // Simplified: in real code this would use the vector Exp
            yLocal.SetValue(i, v);
            sum += v;
        }

        // Step 3: Normalize
        float invSum = 1.0f / sum;

        // BUG: Off-by-one error — loop condition uses <= instead of <.
        // When i == len, SetValue writes one element past the allocated buffer.
        // This overwrites whatever is adjacent in SRAM (another tensor's data,
        // queue metadata, etc.) with no error or warning.
        for (int32_t i = 0; i <= len; i++) {  // should be i < len
            yLocal.SetValue(i, yLocal.GetValue(i) * invSum);  // OOB at i==len
        }

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

extern "C" __global__ __aicore__ void scalar_softmax_overflow(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf) {
    KernelScalarSoftmax op;
    op.Init(input, output, len_buf);
    op.Process();
}
