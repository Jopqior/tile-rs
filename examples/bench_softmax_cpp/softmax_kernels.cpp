// =============================================================================
// C++ Softmax Kernels for Ascend NPU Benchmarking
// =============================================================================
//
// Two implementations:
//   1. softmax_naive  — Scalar loop using DataCopy + GetValue/SetValue
//   2. softmax_opt    — AscendC vector pipeline (optimized baseline)

#include "kernel_operator.h"

// =============================================================================
// Kernel 1: Naive scalar softmax (apples-to-apples comparison with Rust)
// =============================================================================
//
// Uses DataCopy to move data to local memory, then scalar GetValue/SetValue
// for element-wise processing. This isolates compute from DMA and gives a
// fair comparison against scalar Rust kernel code.

class KernelSoftmaxNaive {
public:
    __aicore__ inline KernelSoftmaxNaive() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t n)
    {
        this->n = n;
        uint32_t aligned_n = ((n + 7) / 8) * 8;

        inputGm.SetGlobalBuffer((__gm__ float *)input, aligned_n);
        outputGm.SetGlobalBuffer((__gm__ float *)output, aligned_n);

        pipe.InitBuffer(inQueue, 1, aligned_n * sizeof(float));
        pipe.InitBuffer(outQueue, 1, aligned_n * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Copy input from GM to local
        AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(inLocal, inputGm, n);
        inQueue.EnQue(inLocal);
        inLocal = inQueue.DeQue<float>();

        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

        // Step 1: Find max (scalar loop)
        float max_val = inLocal.GetValue(0);
        for (uint32_t i = 1; i < n; i++) {
            float val = inLocal.GetValue(i);
            if (val > max_val) {
                max_val = val;
            }
        }

        // Step 2: exp(x - max) and accumulate sum (scalar loop)
        // Using a simple polynomial approximation for exp
        float sum = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            float x = inLocal.GetValue(i) - max_val;

            // Clamp for numerical stability
            if (x < -88.0f) x = -88.0f;
            if (x > 88.0f) x = 88.0f;

            // exp via range reduction + polynomial
            const float LOG2E  = 1.44269504088896340736f;
            const float LN2_HI = 0.693359375f;
            const float LN2_LO = -2.12194440e-4f;

            float z = x * LOG2E;
            int nn = (int)(z + (z >= 0.0f ? 0.5f : -0.5f));
            float fn = (float)nn;
            float r = x - fn * LN2_HI - fn * LN2_LO;
            float r2 = r * r;
            float e = 1.0f + r + r2 * (0.5f + r * (1.0f/6.0f + r * (1.0f/24.0f + r * (1.0f/120.0f))));
            while (nn > 0) { e *= 2.0f; nn--; }
            while (nn < 0) { e *= 0.5f; nn++; }

            outLocal.SetValue(i, e);
            sum += e;
        }

        // Step 3: Normalize (scalar loop)
        float inv_sum = 1.0f / sum;
        for (uint32_t i = 0; i < n; i++) {
            outLocal.SetValue(i, outLocal.GetValue(i) * inv_sum);
        }

        inQueue.FreeTensor(inLocal);
        outQueue.EnQue(outLocal);

        // Copy output from local to GM
        outLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outputGm, outLocal, n);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t n;
};

extern "C" __global__ __aicore__ void softmax_naive(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf)
{
    uint32_t n = *((__gm__ uint32_t *)len_buf);
    KernelSoftmaxNaive op;
    op.Init(input, output, n);
    op.Process();
}

extern "C" void softmax_naive_do(uint32_t blockDim, void *stream,
                                  uint8_t *input, uint8_t *output, uint8_t *len_buf)
{
    softmax_naive<<<blockDim, nullptr, stream>>>(input, output, len_buf);
}

// =============================================================================
// Kernel 2: Optimized AscendC vector-pipeline softmax
// =============================================================================
//
// Uses GlobalTensor, LocalTensor, DataCopy, ReduceMax, Sub, Exp, ReduceSum,
// Muls vector intrinsics for high throughput.

constexpr int32_t BUFFER_NUM = 1;

class KernelSoftmaxOpt {
public:
    __aicore__ inline KernelSoftmaxOpt() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t n)
    {
        this->n = n;
        // Align to 32 bytes (8 floats) for vector operations
        uint32_t aligned_n = ((n + 7) / 8) * 8;

        inputGm.SetGlobalBuffer((__gm__ float *)input, aligned_n);
        outputGm.SetGlobalBuffer((__gm__ float *)output, aligned_n);

        pipe.InitBuffer(inQueue, BUFFER_NUM, aligned_n * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, aligned_n * sizeof(float));
        // Work buffers for ReduceMax/ReduceSum destination and workspace
        pipe.InitBuffer(workBuf, aligned_n * sizeof(float));
        pipe.InitBuffer(reduceBuf, aligned_n * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(inLocal, inputGm, n);
        inQueue.EnQue(inLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> inLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> work = workBuf.Get<float>();
        AscendC::LocalTensor<float> reduceWork = reduceBuf.Get<float>();

        // ReduceMax → work[0]
        AscendC::ReduceMax(work, inLocal, reduceWork, n);
        float max_val = work.GetValue(0);

        // outLocal = inLocal - max_val
        AscendC::Adds(outLocal, inLocal, -max_val, n);

        // outLocal = exp(outLocal)
        AscendC::Exp(outLocal, outLocal, n);

        // ReduceSum → work[0]
        AscendC::ReduceSum(work, outLocal, reduceWork, n);
        float sum_val = work.GetValue(0);

        // outLocal = outLocal / sum  (via Muls with 1/sum)
        AscendC::Muls(outLocal, outLocal, 1.0f / sum_val, n);

        inQueue.FreeTensor(inLocal);
        outQueue.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outputGm, outLocal, n);
        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t n;
};

extern "C" __global__ __aicore__ void softmax_opt(GM_ADDR input, GM_ADDR output, GM_ADDR len_buf)
{
    uint32_t n = *((__gm__ uint32_t *)len_buf);
    KernelSoftmaxOpt op;
    op.Init(input, output, n);
    op.Process();
}

extern "C" void softmax_opt_do(uint32_t blockDim, void *stream,
                                uint8_t *input, uint8_t *output, uint8_t *len_buf)
{
    softmax_opt<<<blockDim, nullptr, stream>>>(input, output, len_buf);
}
