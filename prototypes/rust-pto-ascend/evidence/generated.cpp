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
  // pto: %c1
  const int64_t v7 = 1;
  // pto: %c256
  const int64_t v8 = 256;
  // pto: %pto0
  const int64_t v9 = 1;
  // pto: %pto0
  const int64_t v10 = 1;
  // pto: %pto0
  const int64_t v11 = 1;
  // pto: %pto0
  int64_t v12 = v7 * v8;
  // pto: %pto0
  int64_t v13 = v11 * v12;
  // pto: %pto0
  pto::Shape<1, 1, 1, -1, -1> v14 = pto::Shape<1, 1, 1, -1, -1>(v9, v10, v11, v7, v8);
  // pto: %pto0
  pto::Stride<-1, -1, -1, -1, -1> v15 = pto::Stride<-1, -1, -1, -1, -1>(v10 * v13, v13, v12, v8, v7);
  // pto: %pto0
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v16 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v1, v14, v15);
  // pto: %pto1
  __gm__ float* v17 = PTOAS__GLOBAL_TENSOR_DATA(v16);
  // pto: %pto1
  pto::Shape<1, 1, 1, 1, 256> v18 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %pto1
  pto::Stride<256, 256, 256, 256, 1> v19 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %pto1
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v20 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v17, v18, v19);
  // pto: %pto2
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21;
  // pto: %pto2
  uint64_t v22 = (uint64_t) v6;
  TASSIGN(v21, v22);
  TLOAD(v21, v20);
  // pto: %pto3
  const int64_t v23 = 1;
  // pto: %pto3
  const int64_t v24 = 1;
  // pto: %pto3
  const int64_t v25 = 1;
  // pto: %pto3
  int64_t v26 = v7 * v8;
  // pto: %pto3
  int64_t v27 = v25 * v26;
  // pto: %pto3
  pto::Shape<1, 1, 1, -1, -1> v28 = pto::Shape<1, 1, 1, -1, -1>(v23, v24, v25, v7, v8);
  // pto: %pto3
  pto::Stride<-1, -1, -1, -1, -1> v29 = pto::Stride<-1, -1, -1, -1, -1>(v24 * v27, v27, v26, v8, v7);
  // pto: %pto3
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v2, v28, v29);
  // pto: %pto4
  __gm__ float* v31 = PTOAS__GLOBAL_TENSOR_DATA(v30);
  // pto: %pto4
  pto::Shape<1, 1, 1, 1, 256> v32 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %pto4
  pto::Stride<256, 256, 256, 256, 1> v33 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %pto4
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v31, v32, v33);
  // pto: %pto5
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35;
  // pto: %pto5
  uint64_t v36 = (uint64_t) v5;
  TASSIGN(v35, v36);
  TLOAD(v35, v34);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  // pto: %pto6
  Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37;
  // pto: %pto6
  uint64_t v38 = (uint64_t) v4;
  TASSIGN(v37, v38);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v37, v21, v35);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  // pto: %pto7
  const int64_t v39 = 1;
  // pto: %pto7
  const int64_t v40 = 1;
  // pto: %pto7
  const int64_t v41 = 1;
  // pto: %pto7
  int64_t v42 = v7 * v8;
  // pto: %pto7
  int64_t v43 = v41 * v42;
  // pto: %pto7
  pto::Shape<1, 1, 1, -1, -1> v44 = pto::Shape<1, 1, 1, -1, -1>(v39, v40, v41, v7, v8);
  // pto: %pto7
  pto::Stride<-1, -1, -1, -1, -1> v45 = pto::Stride<-1, -1, -1, -1, -1>(v40 * v43, v43, v42, v8, v7);
  // pto: %pto7
  GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND> v46 = GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, -1>, pto::Layout::ND>(v3, v44, v45);
  // pto: %pto8
  __gm__ float* v47 = PTOAS__GLOBAL_TENSOR_DATA(v46);
  // pto: %pto8
  pto::Shape<1, 1, 1, 1, 256> v48 = pto::Shape<1, 1, 1, 1, 256>();
  // pto: %pto8
  pto::Stride<256, 256, 256, 256, 1> v49 = pto::Stride<256, 256, 256, 256, 1>();
  // pto: %pto8
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND> v50 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>, pto::Layout::ND>(v47, v48, v49);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v50, v37);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}