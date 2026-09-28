func.func @scatter_i64(%arg0: !pto.ptr<i64>, %arg1: !pto.ptr<i64>, %arg2: !pto.ptr<i64>, %arg3: !pto.ptr<i64>) {
  %first = pto.make_tensor_view %arg0
  %second = pto.make_tensor_view %arg1
  %destination = pto.make_tensor_view %arg2
  %extra = pto.make_tensor_view %arg3
  %lhs = pto.tload %first
  %rhs = pto.tload %second
  %product = pto.tmul %lhs, %rhs
  pto.tstore %product, %destination
}
