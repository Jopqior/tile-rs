func.func @blend_tiles(%arg0: !pto.ptr<f16>, %arg1: !pto.ptr<f16>) {
  %input = pto.make_tensor_view %arg0
  %output = pto.make_tensor_view %arg1
  %value = pto.tload %input
  pto.tstore %value, %output
  // pto.tadd %ignored_comment_only
}
