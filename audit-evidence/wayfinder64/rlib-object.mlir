module {
  llvm.func @__rust_eh_personality(i32) attributes {def, sym_visibility = "private"}
  llvm.func @softmax_1d(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, sym_visibility = "private"} {
    %0 = llvm.mlir.addressof @get_block_idx : !llvm.ptr<1>
    %1 = llvm.call @get_block_idx() : () -> i64
    %2 = llvm.mlir.constant(1024 : i64) : i64
    %3 = llvm.bitcast %2 : i64 to i64
    %4 = arith.muli %1, %3 : i64
    %5 = llvm.mlir.constant(8 : i64) : i32
    %6 = llvm.alloca %5 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr<1>
    %7 = llvm.getelementptr %arg0[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %7, %6 {alignment = 8 : i64} : !llvm.ptr<1>, !llvm.ptr<1>
    %8 = llvm.load %6 {alignment = 8 : i64} : !llvm.ptr<1> -> !llvm.ptr<1>
    %9 = llvm.mlir.constant(8 : i64) : i32
    %10 = llvm.alloca %9 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr<1>
    %11 = llvm.getelementptr %arg1[%4] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %11, %10 {alignment = 8 : i64} : !llvm.ptr<1>, !llvm.ptr<1>
    %12 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr<1> -> !llvm.ptr<1>
    %13 = llvm.mlir.constant(1 : i64) : i32
    %14 = llvm.bitcast %13 : i32 to i32
    %15 = llvm.mlir.constant(1024 : i64) : i32
    %16 = llvm.bitcast %15 : i32 to i32
    %17 = llvm.mlir.addressof @__tile_load_f32 : !llvm.ptr<1>
    %18 = llvm.call @__tile_load_f32(%8, %14, %16) : (!llvm.ptr<1>, i32, i32) -> i32
    %19 = llvm.mlir.constant(0 : i64) : i32
    %20 = llvm.bitcast %19 : i32 to i32
    %21 = llvm.mlir.constant(1 : i64) : i32
    %22 = llvm.bitcast %21 : i32 to i32
    %23 = llvm.mlir.constant(1024 : i64) : i32
    %24 = llvm.bitcast %23 : i32 to i32
    %25 = llvm.mlir.addressof @__tile_softmax_f32 : !llvm.ptr<1>
    %26 = llvm.call @__tile_softmax_f32(%20, %18, %22, %24) : (i32, i32, i32, i32) -> i32
    %27 = llvm.mlir.constant(1 : i64) : i32
    %28 = llvm.bitcast %27 : i32 to i32
    %29 = llvm.mlir.constant(1024 : i64) : i32
    %30 = llvm.bitcast %29 : i32 to i32
    %31 = llvm.mlir.addressof @__tile_store_f32 : !llvm.ptr<1>
    llvm.call @__tile_store_f32(%12, %26, %28, %30) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
  llvm.func @get_block_idx() -> i64 attributes {def, hacc.function_kind = #hacc.function_kind<DEVICE>, sym_visibility = "private"}
  llvm.func @__tile_load_f32(!llvm.ptr<1>, i32, i32) -> i32 attributes {def, hacc.function_kind = #hacc.function_kind<DEVICE>, sym_visibility = "private"}
  llvm.func @__tile_softmax_f32(i32, i32, i32, i32) -> i32 attributes {def, hacc.function_kind = #hacc.function_kind<DEVICE>, sym_visibility = "private"}
  llvm.func @__tile_store_f32(!llvm.ptr<1>, i32, i32, i32) attributes {def, hacc.function_kind = #hacc.function_kind<DEVICE>, sym_visibility = "private"}
}
