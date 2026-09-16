// Case 1: Type Confusion via GM_ADDR Type Erasure
//
// AscendC kernel entry points receive all tensor pointers as GM_ADDR,
// which is typedef'd to uint8_t*. The kernel must manually cast to the
// correct element type. If the host passes f16 data but the kernel casts
// to float*, the tensor is reinterpreted with the wrong element size:
// each 4-byte "float" actually spans two f16 values, producing garbage.
//
// This is not a hypothetical bug — it occurs whenever a kernel is reused
// for a different dtype without updating the cast, or when a host wrapper
// passes the wrong tensor format.

#include "kernel_operator.h"

constexpr int32_t TILE_SIZE = 256;

class KernelSoftmaxConfused {
public:
    __aicore__ inline KernelSoftmaxConfused() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf) {
        uint32_t n = *((__gm__ uint32_t *)len_buf);

        // BUG: Host passed half-precision (f16) data, but we cast to float.
        // Each "float" element reads 4 bytes instead of 2, so we get:
        //   - Half the expected number of meaningful values
        //   - Each value is garbage (two f16 bit patterns reinterpreted as one float)
        // The compiler cannot catch this because GM_ADDR is just uint8_t*.
        inputGm.SetGlobalBuffer((__gm__ float *)input, n);
        outputGm.SetGlobalBuffer((__gm__ float *)output, n);

        uint32_t alignedTile = ((TILE_SIZE + 7) / 8) * 8;
        pipe.InitBuffer(inQueue, 1, alignedTile * sizeof(float));
        pipe.InitBuffer(outQueue, 1, alignedTile * sizeof(float));
        pipe.InitBuffer(workQueue, 1, alignedTile * sizeof(float));
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
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t len) {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueue.AllocTensor<float>();

        // All computation operates on garbage values due to the type confusion above.
        // The kernel produces silently wrong output — no crash, no error, just
        // incorrect results that may not be caught until much later in the pipeline.
        AscendC::Exp(yLocal, xLocal, len);
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
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> workQueue;
    AscendC::GlobalTensor<float> inputGm, outputGm;
    int32_t totalLen;
};

// The entry point uses GM_ADDR (= uint8_t*) for all tensor arguments.
// The caller can pass any data type — there is no type checking at this boundary.
extern "C" __global__ __aicore__ void softmax_confused(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf) {
    KernelSoftmaxConfused op;
    op.Init(input, output, len_buf);
    op.Process();
}
