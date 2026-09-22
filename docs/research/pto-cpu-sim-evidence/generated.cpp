#include "pto/pto-inst.hpp"
using namespace pto;

template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}


enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

template <typename Ptr>
static AICORE inline void PTOAS__DCCI_SINGLE_CACHE_LINE(Ptr ptr) {
  dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
}

AICORE void add_generated(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  using T = float;
  const int64_t v4 = 2048;
  const int64_t v5 = 1024;
  const int64_t v6 = 0;
  // pto: %one
  const int64_t v7 = 1;
  // pto: %n
  const int64_t v8 = 256;
  // pto: %av
  const int64_t v9 = 1;
  // pto: %av
  const int64_t v10 = 1;
  // pto: %av
  const int64_t v11 = 1;
  // pto: %av
  int64_t v12 = v7 * v8;
  // pto: %av
  int64_t v13 = v11 * v12;
  // pto: %av
  pto::Shape<1, 1, 1, -1, -1> v14 = pto::Shape<1, 1, 1, -1, -1>(v9, v10, v11, v7, v8);
  // pto: %av
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v8, v7);
  // pto: %av
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %bv
  const int64_t v17 = 1;
  // pto: %bv
  const int64_t v18 = 1;
  // pto: %bv
  const int64_t v19 = 1;
  // pto: %bv
  int64_t v20 = v7 * v8;
  // pto: %bv
  int64_t v21 = v19 * v20;
  // pto: %bv
  pto::Shape<1, 1, 1, -1, -1> v22 = pto::Shape<1, 1, 1, -1, -1>(v17, v18, v19, v7, v8);
  // pto: %bv
  pto::Stride<-1, -1, -1, -1, -1> v23 = pto::Stride<-1, -1, -1, -1, -1>(v18 * v21, v21, v20, v8, v7);
  // pto: %bv
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v22, v23);
  // pto: %ov
  const int64_t v25 = 1;
  // pto: %ov
  const int64_t v26 = 1;
  // pto: %ov
  const int64_t v27 = 1;
  // pto: %ov
  int64_t v28 = v7 * v8;
  // pto: %ov
  int64_t v29 = v27 * v28;
  // pto: %ov
  pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v25, v26, v27, v7, v8);
  // pto: %ov
  pto::Stride<-1, -1, -1, -1, -1> v31 = pto::Stride<-1, -1, -1, -1, -1>(v26 * v29, v29, v28, v8, v7);
  // pto: %ov
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v30, v31);
  // pto: %ap
  __gm__ float* v33 = PTOAS__GLOBAL_TENSOR_DATA(v16);
  // pto: %ap
  pto::Shape<1, 1, 1, 1, 256> v34 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %ap
  pto::Stride<256, 256, 256, 256, 1> v35 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %ap
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v33, v34, v35);
  // pto: %bp
  __gm__ float* v37 = PTOAS__GLOBAL_TENSOR_DATA(v24);
  // pto: %bp
  pto::Shape<1, 1, 1, 1, 256> v38 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %bp
  pto::Stride<256, 256, 256, 256, 1> v39 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %bp
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v37, v38, v39);
  // pto: %op
  __gm__ float* v41 = PTOAS__GLOBAL_TENSOR_DATA(v32);
  // pto: %op
  pto::Shape<1, 1, 1, 1, 256> v42 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %op
  pto::Stride<256, 256, 256, 256, 1> v43 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %op
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v41, v42, v43);
  // pto: %x
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45;
  // pto: %x
  uint64_t v46 = (uint64_t) v6;
  TASSIGN(v45, v46);
  // pto: %y
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47;
  // pto: %y
  uint64_t v48 = (uint64_t) v5;
  TASSIGN(v47, v48);
  // pto: %z
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49;
  // pto: %z
  uint64_t v50 = (uint64_t) v4;
  TASSIGN(v49, v50);
  TLOAD(v45, v36);
  TLOAD(v47, v40);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v49, v45, v47);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v44, v49);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}