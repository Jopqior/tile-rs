# PyTorch、TileLang、Triton 的结果比较实践

服务于[确定首批 Metal 验证用例的结果比较规则](https://github.com/Jopqior/tile-rs/issues/11)。本报告保存研究事实；本项目的选择以[最终决议](https://github.com/Jopqior/tile-rs/issues/11#issuecomment-5692509661)为准，不在报告中另存一份设计。

## 范围与结论

核对官方测试源码中的参考位置、数据类型与比较方式，关注前向计算结果，不讨论梯度检查或性能。未执行这些测试，也未运行本项目 Metal 数值用例。

- CPU 和 GPU 参考都有直接先例。Triton 同时有主机 NumPy 参考与同设备 PyTorch 参考；TileLang 的 Metal 矩阵乘测试使用 MPS 上的 PyTorch。
- 下列 f32 实例的参考与被测计算类型一致。这是具体测试实践，不是三个项目所有测试的统一政策。
- 这些测试使用比较 API 默认值或调用处指定的容差，没有在所列比较路径中要求先证明参考实现自身的误差界。
- 来源中的阈值适用于各自测试，不能直接视为本项目 Metal kernel 的精度保证。源码写了一个容差，也不意味着已经说明它的标定方法。

## 固定源码版本

| 项目 | 本次核对的提交 |
| --- | --- |
| PyTorch | `2b3ec34829036a65cd9d1398ea72a0167dc37470` |
| TileLang | `96815790f53d510a17387239581fc3d9f837fb31` |
| Triton | `570e5b4dd1e2d1daf785a894d19268fe7b281cb5` |

## 代表性测试

| 测试 | 参考位置与类型 | 比较方式 |
| --- | --- | --- |
| PyTorch `test_compare_cpu` | 同一批输入移到 CPU，保留 dtype，用同一算子求值 | 浮点/复数显式 `atol=rtol=1e-3` |
| TileLang `test_gemm_float32` | PyTorch MPS 上的 `a @ b`，输入、累加及输出均为 f32 | `torch.allclose(a @ b, c, atol=1e-8)`，rtol 使用 API 默认值 |
| Triton `test_softmax` | 当前测试设备上的 `torch.softmax(x, dim=dim)`，与被测输入同为 f32 | `torch.testing.assert_close(..., rtol=1e-5, atol=1e-6)` |
| Triton `test_reduce1d` 的 f32 sum | 主机 NumPy `np.sum`，结果保留输入 dtype | `np.testing.assert_allclose(z_ref, z_tri, rtol=0.01)` |

### PyTorch：CPU 对照与 API 默认值是两回事

[`test_compare_cpu`](https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/test/test_ops.py#L490-L522) 通过 `arg.to(device="cpu")` 构造 CPU 输入，分别调用设备算子和 CPU 算子后比较。该版本的装饰器包含 `onlyAccelerator`、`skipIfMPS`、`skipCUDAIfNotRocm`，且只选择没有 NumPy 参考的相关算子。它证明存在 CPU 对照实践，不能说这个测试在所有 GPU 后端都执行，尤其不能称它为 MPS 测试。

此测试显式传入的容差不是公开 `torch.testing.assert_close` 的 dtype 默认值。后者的[源码表](https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/testing/_comparison.py#L93-L104)给出 f32 默认 `rtol=1.3e-6`、`atol=1e-5`。[API 文档与实现](https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/testing/_comparison.py#L1415-L1544)定义逐元素比较：

```text
|actual - expected| <= atol + rtol * |expected|
```

公开 API 默认检查 shape、dtype 与 device，跨设备结果需要先移到同一设备或明确关闭设备一致性检查。不能将 PyTorch 内部测试的 `self.assertEqual` 与公开 `assert_close` 的所有默认参数混为一谈；`torch.allclose` 也有自己的默认容差。

### TileLang：Metal kernel 对照 GPU PyTorch

[`test_metal_codegen.py`](https://github.com/tile-ai/tilelang/blob/96815790f53d510a17387239581fc3d9f837fb31/testing/python/metal/test_metal_codegen.py#L8-L68) 的 `test_gemm_float32` 运行 1024×1024×1024 矩阵乘，默认输入与累加类型为 f32。`a`、`b`、`c` 都在 `device="mps"` 上，参考直接调用 `a @ b`。

这是与当前主题直接相关的 Metal 测试先例：参考是 GPU，不是 CPU。它仍用容差比较，而不是要求两种实现逐位一致。源码没有给该容差附上数值证明；本次也没有复现其运行结果。

### Triton：同设备参考和主机参考并存

[`test_standard.py::test_softmax`](https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/test/unit/language/test_standard.py#L158-L182) 从 f32 NumPy 输入构造测试设备上的张量，比较 Triton softmax 与该设备上的 PyTorch softmax。正常 GPU 执行时，这是同 GPU 参考；测试也带有 interpreter 标记，不能把源码中的每一种执行模式都说成 GPU 实测。

[`test_core.py::test_reduce1d`](https://github.com/triton-lang/triton/blob/570e5b4dd1e2d1daf785a894d19268fe7b281cb5/python/test/unit/language/test_core.py#L3021-L3090) 则在主机调用 NumPy，执行 Triton kernel 后通过 `to_numpy` 比较结果。浮点求和用固定 `rtol=0.01`；整数求和及其他一些精确语义采用精确比较。单个项目里也不存在只有一种参考位置、只有一套容差的规则。

## 如何使用这份证据

可以据此说明：同类型参考、CPU/GPU 两种参考位置、默认或专用容差，都是实际项目采用的测试方式。不能据此声称 CPU 天然比 GPU 正确，也不能说 GPU 参考必然与被测结果一致。

这里的测试验证选定样本在选定规则下是否接近，不是对所有输入的数值正确性证明。容差之外仍需检查类型、形状、非有限结果、索引与漏写等问题。具体 CI 政策只在本项目最终决议中定义。
