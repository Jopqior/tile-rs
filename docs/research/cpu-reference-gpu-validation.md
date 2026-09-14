# CPU 数值参考用于 Metal GPU 正确性验证的适用条件

研究产物，对应研究票 [查证 CPU 数值参考用于 Metal GPU 正确性验证的适用条件](https://github.com/Jopqior/tile-rs/issues/10)，
服务于地图 [Metal 正确性 CI：为精简重构建立可信验证设计](https://github.com/Jopqior/tile-rs/issues/2)
下的决策票 [确定 Metal 正确性判据与首批覆盖规则](https://github.com/Jopqior/tile-rs/issues/6)。

范围：已选定的 f32 基础 kernel（add/sub/mul、exp、(a+b)*c、行求和、softmax、f32 转置矩阵乘、
gather、transpose），有限正常数与零。只整理一手资料（官方文档、官方源码、第一方测试），
**不代替用户决定 Q10**，不制定容差数值，不实施代码或 CI 改动。文中数值一律是**被引来源自己使用
的值**，用于说明实践与量级，不是本文提出的判据。

## 0. 直接回答

**用 PyTorch CPU 结果作为 Metal kernel 的正确性参考，在明确条件下站得住；它只证明一部分命题。**

1. **第一方确实把 CPU 结果当设备结果的期望值用**：PyTorch 的 `test_compare_cpu` 逐一比较同一算子的
   GPU 与 CPU 输出；CUTLASS 的 device GEMM 测试用 CPU host reference 校验；Triton 与 JAX 的通用
   测试用 NumPy（CPU）参考。（§3）
2. **运行位置不决定参考的语义角色**：独立的成熟 GPU 库实现同样可当语义参考（CUTLASS 把 cuBLAS 列为
   GEMM 的 verification provider），CPU 实现同样可能只是行为基线；独立性是共享多少代码/数学库/
   规范/硬件的程度问题，不是 CPU/GPU 二分。（§2）
3. **“不保证 bitwise 相同”不等于不可能相同**：官方原话是 *not guaranteed*；当两边在各自算术下都
   精确时相等可达到，第一方测试在这些用例上确实用零容差。（§4.1、§3.4）
4. **参考本身有误差，且差异来源不能靠参考本身区分**：可能来自求值顺序、函数近似，也可能来自实现
   bug。（§4.2、§4.4）
5. **判据按算子族给、按各自自由度推**：单次四则在 MSL 中是正确舍入；`exp` 规范给的是 ulp 上界
   （关闭 fast math 时 `<= 4 ulp`）；`a*b+c` 默认被收缩成 FMA；`(a+b)*c` 受影响的是重结合而非收缩；
   固定有界输入域上可据此推固定保守界。（§4.3、§5、§6）
6. **Metal 还有编译选项层自由度**：`safe` 关掉规范列出的那批不安全优化，但它仍设 `FP contract=on`，
   也不承诺完整 IEEE 行为；规范与 Apple 文档的默认值说法分别针对 CLI 与 `MTLCompileOptions`，
   tile-rs 路径上的实际生效值需实测。（§5）

## 1. 证据级别与实测边界

| 级别 | 含义 | 例子 |
|---|---|---|
| 规范/文档声明 | 官方规范或文档明文 | MSL 规范 4.1 的 ulp 表；PyTorch numerical accuracy 文档 |
| 源码事实 | 第一方仓库可读到的代码与默认值 | `torch/testing/_comparison.py` 容差表；CUTLASS GEMM testbed |
| 第一方测试实践 | 第一方测试实际怎么用参考 | PyTorch `test_compare_cpu`；Triton `_test_unary`；JAX `_CheckAgainstNumpy` |
| 未验证/未复现 | 本文无硬件或环境去跑 | **全篇 Metal 行为**：无 Apple GPU、无 Metal 工具链、无任何实验 |

本环境无 macOS、无 Metal 工具链、无 Apple GPU，也未运行 PyTorch/CUDA；§5 全部是规范文本。
MSL §8.2 的舍入模式自由度、`xcrun metal` 的选项支持范围、默认 math mode 的实际生效值，需要在
[实测免费托管 runner 的最小 Metal 编译与执行链路](https://github.com/Jopqior/tile-rs/issues/9)
或本地 Apple 机器上确认。

## 2. 参考的语义角色 ≠ 参考的运行位置

区分看的是**被比较的计算是什么**，而不是它在哪跑：

| 角色 | 判据 | 第一方实例 |
|---|---|---|
| **语义参考** | 从算子数学定义出发、在另一条实现路径上求值，估计离数学值多远 | PyTorch `test_compare_cpu`（CPU 算子 vs 设备算子）；CUTLASS host reference；Triton/JAX 的 NumPy 参考；CUTLASS profiler 把 cuBLAS 列为 GEMM 的 verification provider |
| **行为基线** | 与既有实现或同一实现的另一后端比较，判“行为是否改变” | PyTorch dynamo 的 `same(ref, res)`（eager vs compiled，同设备）；Triton 解释器 vs GPU 后端；与 kernel 共享同一 libm/同一后端的 CPU 实现 |

因此：CPU 参考既可能是语义参考，也可能只是行为基线；GPU 参考可以是语义参考（cuBLAS 之于 GEMM 即
一例）。GPU 上的参考与被测 kernel 共享 GPU 编译器/驱动/库，这一层的错误在它那里不可见；CPU 参考
共享更少。**这是相对程度，不是“GPU 参考不能验证数值正确性”的结论，第一方也不要求必须有 GPU
参考**（CUTLASS 的 provider 是可选集合，host 与 device 并列）。

## 3. 最直接的一手证据

### 3.1 PyTorch：`test_compare_cpu` 拿 CPU 结果当设备结果的期望值

`test/test_ops.py`（tag `v2.8.0`）L373-L396：

```python
for sample in op.reference_inputs(device, dtype):
    cpu_sample = sample.transform(to_cpu)        # 同一批数值搬到 CPU
    cuda_results = op(sample.input, ...)         # 设备上算
    cpu_results  = op(cpu_sample.input, ...)     # CPU 上算
    # Lower tolerance because we are running this as a `@slowTest`
    self.assertEqual(cuda_results, cpu_results, atol=1e-3, rtol=1e-3)
```

配套默认值：测试基类 `TestCase.assertEqual` 的 `exact_device=False` 是默认（`common_utils.py` L4071，
旁注 `# TODO: default this to True`），透传为 `check_device=exact_device`（L4131-L4133）；公开 API
`torch.testing.assert_close` 默认 `check_device=True`（`_comparison.py` L1325），跨设备比较要显式关闭。
判定式为 `|actual - expected| <= atol + rtol * |expected|`（L1337），f32 默认 `rtol=1.3e-6, atol=1e-5`
（L54-L60）。来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_ops.py#L373-L396>、
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/common_utils.py#L4060-L4135>。

### 3.2 MPS：f64 不可用；`exp` 的 CPU/MPS 差异有第一方记录

`test/test_mps.py`（v2.8.0）：`test_double_error`（L11895）断言在 MPS 上构造 `torch.float64` 抛
`TypeError: the MPS framework doesn't support float64`（`a.double()` 同）——**f64 参考不可能在 MPS 上
求值**。`test_exp`（L728）用 `compare_with_numpy`（`common_utils.py` L4010-L4042：把输入拷到 CPU 跑
`np.exp`，再与 MPS 输出的 CPU 拷贝 `assertEqual`），即 **MPS 结果对 NumPy（CPU）参考**。
`test_exp1`（L759-L768）直接对 CPU 结果比较，并把一次真实差异写进注释：

```python
self.assertEqual(output, output_cpu, atol=1e-8, rtol=1e-8)
# If exponentWithTensor: MPS call is used on M1 running 14.5 test will fail with
# Greatest absolute difference: 1.1920928955078125e-07 ... (up to 1e-08 allowed)
# Greatest relative difference: 1.0786502002702036e-07 ... (up to 1e-08 allowed)
```

第一方记录了一个配置下 `exp` 的 MPS/CPU 差异超出其自设界约一个数量级——这是“超越函数需要单独定
判据”的直接证据，而不是“无法定判据”。来源：
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_mps.py#L728-L775>、
#L11895-L11903。

### 3.3 CUTLASS：CPU host reference、可配比较模式、provider 可以是 GPU 库

`test/unit/gemm/device/gemm_testbed_3x.hpp`（CUTLASS `main`）：`verify()` →
`cutlass::reference::host::Gemm3x(...)`（L2802）在 **CPU** 上算 `reference_D`，再
`equality_check(reference_D, tensor_D)`（L1788）。该 host reference 使用 kernel 声明的
`ElementAccumulator` 类型（`gett.hpp` L837 起）——**这只能确认累加类型被复刻；它推不出累加顺序相同，
这些源码也没有解释 EXACT 默认为何能通过**，本文记为未确认（§8）。比较模式默认
`CheckEquality::EXACT`（L388 等）；`RELATIVE` 是显式选项，走
`TensorRelativelyEquals(lhs, rhs, epsilon, nonzero_floor)`，其中
`epsilon = static_cast<Element>(0.1f)`，判据为 `diff < epsilon * (|a| + |b|)`
（`include/cutlass/relatively_equal.h` L58-L78）——本文只描述该公式，不把它读成“接受某个百分比误差”。
`media/docs/cpp/profiler.md` 把性能与校验分成两个开关：`--mode=profile`、
`--verification-enabled=<bool>`、`--epsilon=<error>`（“Setting to zero (default) requires bit-level
equivalence”）、`--verification-providers`（GEMM：`{cublas*}`；Conv2d：`{cudnn*, device*, host}`）。
来源：<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/test/unit/gemm/device/gemm_testbed_3x.hpp#L583-L605>、
#L2802、<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/util/include/cutlass/util/reference/host/gett.hpp#L837-L895>、
<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/profiler.md#L260-L290>。

### 3.4 Triton / JAX：通用测试的参考是 NumPy（CPU）；PyTorch 编译器测试用 f64 仲裁

Triton `python/test/unit/language/test_core.py`：`_test_unary`（L165-L190）在主机上
`z_ref = eval(expr)`（NumPy 输入、NumPy 求值），再 `np.testing.assert_allclose(..., rtol=0.01)`；
同文件也有同 GPU 参考（`z_ref = torch.erf(x)`，L1133 → L1136）与精确语义的零容差用例
（L476、L2265-L2266 的 `atol=0, rtol=0`；L506 的 `np.testing.assert_array_equal`）。
JAX `jax/_src/test_util.py`：`_CheckAgainstNumpy`（L1467-L1475）比较 NumPy 与当前后端结果；
`_CompileAndCheck`（L1420-L1464）比较 jit 与逐 op（同设备）结果。默认容差表见
`jax/_src/public_test_util.py` L46-L91（f32 `1e-6`、f16 `1e-3`、bf16 `1e-2`、f64 `1e-15`、
整数/bool 为 `0`），`_assert_numpy_close` 按数组 size 放大（L164）；**该放大的理由源码未说明，
本文不据此推断误差模型**；`tests/lax_numpy_test.py` 的 `testDot`（L574-L585）与 `testMatmul`
（L622-L632）各自手写 f32 容差（`2e-5` 与 `2e-2`），**本文不据此推断标定方法**。
PyTorch `torch/_dynamo/utils.py::same(ref, res, fp64_ref=None, tol=1e-4, ...)`（L2780）先试同设备
`torch.allclose`（L2913）；失败后进入 `# Check error from fp64 version`（L2922），比较
`rmse(fp64_ref, ref)` 与 `rmse(fp64_ref, res)`，判据 `res_error <= multiplier * ref_error + tol / 10.0`
（L2990）；f64 参考的构造见 `test/inductor/test_layout_optim.py` L95-L117（模型与输入都
`.to(torch.float64)`）。来源：<https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/test/unit/language/test_core.py#L165-L190>、
<https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/jax/_src/test_util.py#L1420-L1475>、
<https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/jax/_src/public_test_util.py#L46-L91>、
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_dynamo/utils.py#L2773-L3010>。

## 4. 为什么 CPU 参考不是真值，但仍可用

**4.1 “不保证 bitwise”不是“不可能 bitwise”。** PyTorch 官方（`docs/source/notes/numerical_accuracy.rst`
L13-L38）：“PyTorch is not guaranteed to produce bitwise identical results … In particular, **CPU and GPU
results can be different even for bitwise-identical inputs and even after controlling for the sources of
randomness.**” 该文档的两条具体例子（`(A@B)[0]` vs `A[0]@B[0]`；对张量切片施加同一运算 vs 对完整结果
取切片）针对的是**改变了求值结构**的场景。当两边在各自算术下都精确时，相等是可达到的，第一方测试
在这些用例上确实用零容差（§3.4）。来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/notes/numerical_accuracy.rst#L13-L38>。

**4.2 求值顺序是差异的正当来源。** IEEE 754 的加法/乘法不结合，归约长度与结合顺序都会改变结果；
PyTorch 文档也把 batched vs 非 batched、整张量 vs 切片列为结果可以不同的正道理由。同类第一方证据：
cuBLAS “Results Reproducibility” 说明同一 toolkit 版本、同架构、同 SM 数下有 bitwise 稳定性，并说明
多流或 atomic 路径下该保证不成立（<https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>）。
对行求和、softmax 分母、f32 矩阵乘的 k 维累加，参考的用途是给“允许不同顺序”留出界。

**4.3 超越函数的 ulp 值是上界，不是必然误差。** MSL 规范 4.1 §8.4 明确 Table 8.1 给的是**最小精度**
（ULP），且“The reference value used to compute the ULP value of an arithmetic operation is the
infinitely precise result”——个别结果完全可能正确舍入。关闭 fast math 时 `exp` 为 `<= 4 ulp`；开启
fast math（Table 8.2，规范注明默认）为 `<= 3 + floor(fabs(2 * x)) ulp`。**在固定有界输入域上两者都是
有限上界，据此推一个固定保守界是可辩护的**，与“不能要求与 CPU `exp` bitwise 相同”不矛盾。

**4.4 f32 升 f64：同一数值、不同 bit；也不会消掉 bug。** Rust Reference：“Casting from an f32 to an
f64 is perfect and lossless”（`[expr.as.numeric.float-widening]`）；C++ 工作草案 `[conv.double]`：
“If the source value can be exactly represented in the destination type, the result of the conversion is
that exact representation.” 因此加宽不引入舍入，参考看到的**数值**与被测 kernel 一致（编码 bit 串
不同）。这对输入、权重、常量同样成立——把 f32 权重升 f64 与把 f32 输入升 f64 是同一件无损的事，
不构成两条不同的输入路线。但**差异不只来自舍入**：也可能来自求值顺序、函数近似或实现 bug，参考
本身不能把这些来源分开；这正是判据要逐算子族给、失败时还要保留可复现材料的原因（产物契约见
[定义可追溯可重放的 Metal 验证产物契约](https://github.com/Jopqior/tile-rs/issues/7)）。另外 f64 不是
无穷精度，n 项累加的参考自身仍有误差；“差 ≤ 常数”隐含了“参考误差可忽略”的假设，需要被承认或补偿。
来源：<https://doc.rust-lang.org/reference/expressions/operator-expr.html>、<https://eel.is/c++draft/conv.double>。

**4.5 条件数与动态范围会影响差异量级。** 同一份 PyTorch 文档指出病态输入下结果可能随设备或后端变化
（建议用 `torch.linalg.svdvals` / `torch.linalg.cond` 检测），其 “Extremal values” 一节给出 f32 下
`a.norm()` 溢出为 `inf` 而 `a.double().norm()` 正常的例子。因此判据要么按输入分布定，要么用相对量
（`atol + rtol*|expected|`；CUTLASS 的 `TensorRelativeErrorMetric = ||A-B||/||B||`，见 `error_metrics.h` L45-L60）。

## 5. Metal 侧已知自由度（规范级，未实测）

MSL 规范 4.1 §1.6.3（p.15-16）列出 fast math 的优化项：`No NaNs`、`No INFs`、`No Signed Zeroes`、
`Allow Reciprocal`、`Allow Reassociation`、**`Allow Contract`**。相关开关：
`-fmetal-math-fp32-functions=<fast|precise>`（规范：“The default is fast.”）、
`-fmetal-math-mode=<fast|relaxed|safe>`（规范：“The default is fast.”）、legacy `-ffast-math` ≡
`fast + fast`、`-fno-fast-math` ≡ `precise + safe`。规范对 `safe` 的原文是“it disables unsafe
floating-point optimizations … **This sets the FP contract to on.**”——即 **`safe` 不关闭收缩**，关闭
收缩要另用 `-ffp-contract=off` 或 `#pragma METAL fp contract(off)`。**`safe` 也不等于完整 IEEE 行为**：
§8.1 说非规格化数“may be flushed to zero”，§8.2 说单精度舍入模式“either round ties to even or round
toward zero”。默认值方面，MSL 规范说 CLI 选项默认 `fast`；Apple Developer 文档说
`MTLMathMode.relaxed` 是 Apple silicon 的默认（`.fast` 是 Intel/AMD 默认），
`MTLMathFloatingPointFunctions.fast`“This is the default behavior.”，`MTLCompileOptions.mathMode` 的
Discussion 给出 `fastMathEnabled=false` ⇒ `safe`+`precise` 的映射。**这些描述的是不同接口层，本文不把
差异判定为矛盾**；tile-rs 走 `xcrun metal` 时的实际生效值需实测（§1）。来源：MSL 规范 4.1 §1.6.3、
§8.1–§8.4；<https://developer.apple.com/documentation/metal/mtlmathmode>、
<https://developer.apple.com/documentation/metal/mtlmathmode/relaxed>、
<https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode>、
<https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions/fast>。

## 6. 本票算子范围内的含义

下表只给“与参考相关的主要自由度”及其一手依据，**不给任何具体数值**。

| 算子 | 自由度 |
|---|---|
| `add` / `sub` / `mul` | 单次运算在 MSL Table 8.1/8.2 中都是 “Correctly rounded”；但**不结合**，且可能被重结合。 |
| `a*b+c`（收缩） | 默认允许收缩为一条 FMA（§1.6.3 原文即 “a*b+c may be converted to a single fused-multiply-add”）。 |
| `(a+b)*c` | **收缩不适用于该形式**；影响它的是 `Allow Reassociation`。 |
| `exp` | 关闭 fast math `<= 4 ulp`、开启时 `<= 3 + floor(\|2x\|) ulp`（均为上界）；固定有界输入域上仍可给出固定保守界。 |
| 行求和 | 非结合 + 不同归约顺序；参考给的是“允许偏离”的界。 |
| `softmax` | 叠加 `exp` 的上界、行求和顺序自由度与除法（fast math 下 `x/y` 另有上界，Table 8.2）。 |
| f32 转置矩阵乘 | k 维累加顺序；是否走 tensor-core/低精度路径（PyTorch 文档列 TF32 与 reduced-precision reduction 为额外变量）。MPS 上不适用 TF32，但不能假定不存在其他低精度路径。 |
| `gather` / `transpose` | 无算术；两边都精确时相等可达到，第一方对此类语义用零容差（§3.4）。 |

## 7. Q10 的未决点与已确认项

**已确认、本文不重开**（见决策票 [确定 Metal 正确性判据与首批覆盖规则](https://github.com/Jopqior/tile-rs/issues/6)）：
Q11 —— 关闭 fast-math、按算子族给误差界。

**Q10 唯一暂停项**：是否采用 **PyTorch CPU** 作为参考。本文只给它的适用条件与边界（§0、§4）。

**未被选中但不是必需**：是否再加一条 GPU 参考。第一方并不要求“必须有 GPU 参考”（§2、§3.3）；
若加入，与 GPU 编译链共模的部分需要在判定说明里写明。

与 Q10 相关的子问题（只列，不选）：① 参考在多大程度上是语义参考而非行为基线（§2）；
② f64 参考只能在 CPU/其他设备（MPS 无 f64，§3.2）；③ 公开 `assert_close` 默认 `check_device=True`，
跨设备比较需显式关闭（§3.1）；④ 判据是否逐算子族给界、界按规范上界推还是按相对基线给（§3.4）；
⑤ 纯数据搬运算子是否单列零容差（§3.4 已有先例）；⑥ 参考自身误差是否显式留出（§4.4）。

## 8. 未确认 / 不假设

- **未确认**：CUTLASS host reference 与 device kernel 的累加顺序关系，及其 EXACT 默认能通过的原因；
  §3.3 只确认累加**类型**相同。
- **未确认**：MSL §8.2 的舍入模式自由度对 Apple GPU 上 `add/mul` 的实际影响；Table 8.1 的
  “Correctly rounded” 相对哪种舍入模式。
- **未确认**：`xcrun metal` 在实际使用的 macOS 版本上是否接受 `-fmetal-math-mode` /
  `-fmetal-math-fp32-functions`，以及默认 math mode 的实际生效值（§5）。
- **未确认**：PyTorch MPS 后端实际传给 Metal 的编译选项（本文未读 MPS 源码，不能从测试容差反推）。
- **不假设**：不假设 CPU 与 GPU 必须或必然不同；不假设 CPU 参考是真值或必然更高精度（JAX 关闭 x64
  时其 NumPy 参考同为 f32）；不假设 GPU 参考无法验证数值正确性；不假设第一方的容差数值适用于本项目。
- **不在范围**：特殊值（NaN/INF）、非规格化数、低精度类型（f16/bf16/fp8）、性能基准与具体容差数值。

## Sources

pinned 链接；行号对应上引修订。Apple 的 MSL 规范与 Developer 文档无 commit，标版本与抓取日。

- **PyTorch**（tag `v2.8.0` = `ba56102387ef21a3b04b357e5b183d48f0afefc7`）：`docs/source/notes/numerical_accuracy.rst`；`test/test_ops.py`（`test_compare_cpu`）；`torch/testing/_internal/common_utils.py`（`assertEqual`、`compare_with_numpy`）；`torch/testing/_comparison.py`；`test/test_mps.py`（`test_exp`、`test_exp1`、`test_double_error`）；`torch/_dynamo/utils.py`（`same` 的 fp64 仲裁）；`test/inductor/test_layout_optim.py`。基址 <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/>
- **CUTLASS**（`main` @ `147295a3d4b75f3aeff247c25b8927cea9a7006a`）：`test/unit/gemm/device/gemm_testbed_3x.hpp`；`tools/util/include/cutlass/util/reference/host/gett.hpp`；`include/cutlass/relatively_equal.h`；`tools/util/include/cutlass/util/reference/host/error_metrics.h`；`media/docs/cpp/profiler.md`。基址 <https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/>
- **Triton / JAX**（`main` @ `59eaaff969e731cdea71188fc4aedbd33fa393fe` / `adb0562417371429beddf7d575a0753dc957de19`）：Triton <https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/test/unit/language/test_core.py#L165-L190>；JAX `jax/_src/test_util.py`（L1420-L1475）、`jax/_src/public_test_util.py`（L46-L91）、`tests/lax_numpy_test.py`（L574-L585、L622-L632）
- **Apple**：Metal Shading Language Specification Version 4.1（2026-06-04）<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>（§1.6.3 p.15-16；§8.1–§8.4 与 Table 8.1/8.2 p.368-372）；<https://developer.apple.com/documentation/metal/mtlmathmode>；<https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode>；<https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions>
- **其它**：cuBLAS 13.4 “Results Reproducibility” <https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>；Rust Reference `[expr.as.numeric.float-widening]` <https://doc.rust-lang.org/reference/expressions/operator-expr.html>；C++ 工作草案 `[conv.double]` <https://eel.is/c++draft/conv.double>
