# CPU 数值参考用于 Metal GPU 正确性验证的适用条件

Research findings for [Jopqior/tile-rs#10](https://github.com/Jopqior/tile-rs/issues/10)
under map [#2](https://github.com/Jopqior/tile-rs/issues/2).

研究问题：tile-rs 的 kernel 在 GPU（Metal）执行时，用 PyTorch CPU 计算结果作为正确性参考是否
合理，以及这种参考在什么条件下成立、在哪里失效。本文只整理一手资料（官方文档、源码、第一方测试
实践）并区分事实层级；**不代替用户决定 Q10**，也不制定任何容差数值，不实施代码或 CI 改动。

- 仓库惯例说明：本仓库已有研究产物放在 `docs/research/`（见
  [`metal-validation-chain.md`](https://github.com/Jopqior/tile-rs/blob/research/metal-validation-chain/docs/research/metal-validation-chain.md)、
  [`metal-runner-capability.md`](https://github.com/Jopqior/tile-rs/blob/research/metal-runner-capability/docs/research/metal-runner-capability.md)），
  本文沿用该位置与结构。
- 本文所有“容差数值”都是**被引来源自己使用的值**，用于说明一手实践的存在与量级，**不是本文提出的
  判据**。

## 0. 简短回答与证据级别

### 0.1 直接答案

**合理，但是有条件的合理，而且“合理”只覆盖一部分命题。** 一手资料支持的结论如下：

1. **CPU 结果作为 GPU 结果的期望值，是主流第一方测试的常规做法。** PyTorch 自己的 op 测试
   `test_compare_cpu` 就把同一算子在 CPU 上的输出当作 CUDA 输出的期望值比较（并且为此显式放宽到
   `atol=1e-3, rtol=1e-3`）。CUTLASS 的 device GEMM 测试用 CPU 上的 host reference 校验；Triton
   的逐元素测试用 **NumPy（CPU）** 参考；JAX 的 `_CheckAgainstNumpy` 用 NumPy（CPU）参考。
   证据见 §3.3、§4.1、§4.3、§4.4。
2. **“合理”依赖四个前提**（缺一即不成立，逐条证据见 §5）：
   - 参考必须比被测实现更精确或至少是独立实现。**“CPU”本身不等于“更高精度”**：JAX 在关闭 x64 时
     的 NumPy 参考同样以 f32 运行（§4.4），此时它只是“另一个 f32 实现”。
   - CPU 与 GPU 不可能也不必 bitwise 相同（PyTorch 官方明确写出，§3.1），所以判据必须是误差界，
     不能是相等。
   - 判据必须按算子语义分别定：`add/sub/mul` 是正确舍入，`exp` 不是（Metal 规范给出 ulp 上界，
     §5.3）；`gather/transpose` 是数据搬运，应当按精确一致处理（§4.3、§6）。
   - 参考本身也不是真值：f64 参考有自身舍入误差，且它不覆盖输入条件数带来的放大（§5.6、§5.7）。
3. **“CPU 参考”和“GPU 参考”证明的东西不同，互为补充而非替代**（§7）：
   - CPU/f64 参考能回答“结果离数学值多远”，因此是**数值正确性**的合适依据。
   - 同 GPU 参考能回答“新实现是否复现了既有实现的行为”，因此是**回归**的合适依据，但它与被测实现
     共享编译器/库，无法独立发现共享层里的错误。cuBLAS 官方“同一 toolkit 版本 + 同一架构 + 同 SM
     数 → bitwise 相同”的声明（§4.5）说明同设备一致性强，**恰恰说明它不能证明正确性**。
4. **f32 输入先定稿、再升 f64 是数值上等价且成立的**：Rust 参考明确规定 `f32 → f64` 转换是
   “perfect and lossless”，C++ 工作草案规定“若能精确表示，结果就是该精确表示”，而被测 f32 输入
   在 f64 中全能精确表示。因此 f64 参考里的输入与被测 kernel 看到的输入是同一批数
   （§5.6）。这条同时说明：**这个参考测的是 kernel 的舍入/求值顺序误差，不测输入本身的表示误差**。
5. **性能基准与数值参考必须分开**：CUTLASS profiler 文档把 profiling 与 verification 做成两个
   开关，并说明 epsilon=0 时要求 bit-level equivalence（§4.2）；Triton 的 `do_bench` 与
   `assert_close` 是同一个模块里两个互不相干的函数（§4.3）。本文不覆盖性能基准，见 §9。

### 0.2 证据级别

本文每条结论标注来源层级，全文只用前三级：

| 级别 | 含义 | 例子 |
|---|---|---|
| **规范/文档声明** | 官方文档、语言/接口规范明文写出 | Apple MSL 规范 §8.4 的 ulp 表；PyTorch numerical accuracy 文档 |
| **源码事实** | 第一方仓库中可读到的代码、测试、参数默认值 | `torch/testing/_comparison.py` 的容差表；CUTLASS `gemm_testbed_3x.hpp` |
| **第一方测试实践** | 第一方项目自己的测试怎么用参考 | PyTorch `test_compare_cpu`；Triton `_test_unary`；JAX `_CheckAgainstNumpy` |
| 未验证/未复现 | 本文没有对应硬件或环境去跑 | **全篇的 Metal 行为**：本文没有任何 Apple GPU、没有 Metal 工具链、没有运行任何实验。所有 Metal 结论都是规范文本，不是实测 |

**本文的实测边界（重要）**：本工作区无 macOS、无 Metal 工具链、无 Apple GPU，也没有运行
PyTorch/CUDA。因此 §5.2–§5.4 关于 Metal 的一切都是**规范声明**，尚待 runner 探针或本地 Apple 机器
实测确认（对应 sibling 票 #9）。本文也没有验证 `xcrun metal` 在各 macOS 版本上是否接受
`-fmetal-math-mode`（§5.2 已标注该不确定性）。

## 1. 修订与抓取对象

| 对象 | 修订 | 说明 |
|---|---|---|
| PyTorch | tag `v2.8.0` = `ba56102387ef21a3b04b357e5b183d48f0afefc7` | 文档与测试源码均按此修订引用 |
| CUTLASS | `main` @ `147295a3d4b75f3aeff247c25b8927cea9a7006a` | `test/unit/gemm/device/`、`tools/util/include/cutlass/util/reference/`、`media/docs/cpp/profiler.md` |
| Triton | `main` @ `59eaaff969e731cdea71188fc4aedbd33fa393fe` | `python/triton/testing.py`、`python/test/unit/language/test_core.py` |
| JAX | `main` @ `adb0562417371429beddf7d575a0753dc957de19` | `jax/_src/test_util.py`、`jax/_src/public_test_util.py`、`tests/lax_numpy_test.py` |
| OpenXLA | `main` @ `6c5e7717f9af43bb5256d667c9e910be450345ae` | `xla/error_spec.h`、`xla/tests/literal_test_util.h` |
| Apple MSL 规范 | **Version 4.1**，文档日期 2026-06-04，共 383 页 | PDF：`developer.apple.com/metal/Metal-Shading-Language-Specification.pdf`（本地 `pdftotext` 后按页/节引用） |
| Apple Developer 文档 | 抓取日 2026-09-14（文档无版本号，含 “Copyright © 2026”） | `MTLMathMode`、`MTLCompileOptions.mathMode`、`MTLMathFloatingPointFunctions` |
| CUDA C++ Programming Guide | 12.6.0 archive（静态 HTML 全文） | 现行 `latest` 页面已改为无内容的 SPA 外壳，故引用 archive 12.6.0 |
| cuBLAS | 13.4 文档（抓取日 2026-09-14） | Results Reproducibility 一节 |
| Rust Reference / C++ 工作草案 / NumPy 文档 | 抓取日 2026-09-14 | §5.6 的加宽转换依据 |

## 2. 先区分三种“参考”，否则讨论会串味

一手资料里“reference”至少指三种不同东西，混用会导致“CPU 参考是否合理”这个问题的答案漂移。
仓库 `CONTEXT.md` 已经区分了“正确性验证用例”和“预期输出依据”，下面把“依据”再分三类：

| 类型 | 定义 | 第一方实例 | 它能证明什么 |
|---|---|---|---|
| **语义参考**（本文关注对象） | 对同一**计算含义**在更精确或更独立的环境下求值，作为期望输出 | PyTorch `test_compare_cpu` 的 CPU 输出；CUTLASS host reference；Triton 的 NumPy 参考 | 被测实现的数值误差有多大 |
| **行为基线** | 既有实现/同设备另一实现的实际输出 | PyTorch dynamo 的 `same(ref, res)`（eager vs compiled，同设备）；Triton 用同 GPU 的 `torch.erf` 作参考 | 新实现是否复现了既有行为 |
| **性能基准** | 只测时间/吞吐，不判定数值 | CUTLASS profiler 的 profiling 开关；Triton `do_bench` | 速度，与正确性无关 |

**本票问的是第一类**。第二类在 map #2 已被明确排除为正确性标准（map Notes：“不能把旧实现自动当作
正确性标准”）。

## 3. PyTorch 自己怎么用 CPU 结果

### 3.1 官方文档：CPU 与 GPU 结果可以不同，且不保证 bitwise

`docs/source/notes/numerical_accuracy.rst`（v2.8.0）：

> “Because of this, PyTorch is not guaranteed to produce bitwise identical results for floating point
> computations that are mathematically identical. Similarly, bitwise identical results are not guaranteed
> across PyTorch releases, individual commits, or different platforms. **In particular, CPU and GPU
> results can be different even for bitwise-identical inputs and even after controlling for the sources
> of randomness.**”

同一文件还给出两条与本票算子直接相关的例子：

> “``(A@B)[0]`` ... is not guaranteed to be bitwise identical to ``A[0]@B[0]``”（批处理 matmul）
> “``A.sum(-1)[0]`` is not guaranteed to be bitwise equal to ``A[:,0].sum()``”（归约/切片）

来源（pinned）：
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/notes/numerical_accuracy.rst#L13-L19>、
同文件 L30-L38。

同一文档也说明它的精度模型有例外（对本票范围是“超纲”边界，仅记录）：TF32 只读输入尾数前 10 位；
FP16/BF16 可开启 reduced-precision reduction；AMD MI200 上 FP16/BF16 的 V_DOT2/MFMA 会把非规格化数
清零。

`docs/source/notes/randomness.rst` 补充：某些 GPU 归约（`index_add_` 等）在 CUDA 上本身就不可复现
（atomic），即使在**同一设备**上跑两次也不是 bitwise 相同。

### 3.2 `torch.testing.assert_close` 的默认值与设备检查

源码 `torch/testing/_comparison.py`（v2.8.0）：

- 判定式：`|actual - expected| <= atol + rtol * |expected|`
  （L1337，docstring；实现 L619 `tolerance = self.atol + self.rtol * abs(self.expected)`）。
- 默认容差表 `_DTYPE_PRECISIONS`（L54-L60）：f16 `(1e-3, 1e-5)`、bf16 `(1.6e-2, 1e-5)`、
  f32 `(1.3e-6, 1e-5)`、f64 `(1e-7, 1e-7)`（顺序为 `(rtol, atol)`）。表在 docstring L1419 起复述。
- **默认 `check_device=True`**（L1325）：即公开 API 默认要求两个张量在同一设备上，跨设备比较必须
  显式 `check_device=False`；此时文档说会“moved to the CPU before being compared”。
- 非有限值不是“近似相等”而是必须相等；NaN 只有在 `equal_nan=True` 时才相等（L1330-L1336）。
- 文档明确建议用户用 `functools.partial` 定制，并给出“需要相等就用 `rtol=0, atol=0`”的官方示例。

来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_comparison.py#L54-L60>、
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_comparison.py#L1317-L1340>、
同文件 #L1419-L1460。

**对本票的直接含义**：如果参考库用 `torch.testing.assert_close` 比较 CPU 参考与 MPS 输出，默认会因
设备不同而报错；必须显式关闭设备检查（或先把参考搬到 MPS，但那会改变“参考在哪算”的语义）。

### 3.3 `test_compare_cpu`：PyTorch 把 CPU 结果当作设备结果的期望值

`test/test_ops.py`（v2.8.0）L373-L396：

```python
def test_compare_cpu(self, device, dtype, op):
    def to_cpu(arg): ...
    samples = op.reference_inputs(device, dtype)
    for sample in samples:
        cpu_sample = sample.transform(to_cpu)
        cuda_results = op(sample.input, *sample.args, **sample.kwargs)      # 设备上算
        cpu_results  = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)  # CPU 上算，输入同值
        ...
        # Lower tolerance because we are running this as a `@slowTest`
        self.assertEqual(cuda_results, cpu_results, atol=1e-3, rtol=1e-3)
```

这是本票问题最直接的一手证据：**PyTorch 自己就用 CPU 结果作为设备结果的期望值，并且明确承认必须
放宽容差**（注释与 `atol=rtol=1e-3` 都在同一段）。

配套事实（决定这段代码为什么不会因设备不同而失败）：内部测试基类 `TestCase.assertEqual` 的签名里
`exact_device=False` 是**默认值**（`torch/testing/_internal/common_utils.py` L4071，旁边还有
`# TODO: default this to True`），并且它把该值透传成 `check_device=exact_device`（L4131-L4133）。
也就是说：**PyTorch 内部测试框架默认允许跨设备比较，而公开 API `assert_close` 默认不允许**。

来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_ops.py#L373-L396>、
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/common_utils.py#L4060-L4135>。

同一基类还有“按需放宽容差”的机制 `precisionOverride`（L1475）与
`toleranceOverride`（L1512），后者注释写明“Specifies per-dtype tolerance overrides tol(atol, rtol)…
it also controls the behavior of functions like self.assertEqual()”。**逐设备/逐 dtype 覆盖容差是
第一方内建能力，而不是 hack。**

### 3.4 fp64 参考：PyTorch 编译器测试用 f64 当仲裁者

`torch/_dynamo/utils.py`（v2.8.0）的 `same(ref, res, fp64_ref=None, tol=1e-4, ...)`（L2780）是
dynamo/inductor “编译结果是否正确”的判据。逻辑：

1. 先试普通 `torch.allclose(ref, res, atol=tol, rtol=tol)`（L2913）——此处的 `ref` 是**同设备**的
   eager 结果。
2. 普通比较失败时，进入注释为 `# Check error from fp64 version` 的分支（L2922）：
   - `ref_error = rmse(fp64_ref, ref)`、`res_error = rmse(fp64_ref, res)`（`rmse` = L2773
     `torch.sqrt(torch.mean(torch.square(ref - res)))`）；
   - 判据 `passes_test = res_error <= (multiplier * ref_error + tol / 10.0)`（L2990），
     multiplier 视 dtype/张量大小取 2/3/8/10；
   - 特殊情形：若 eager 有 NaN 而 f64 参考与编译结果都没 NaN，注释写明 “res is 'BETTER' than ref so
     we count it pass”（L2934-L2945）。

具体使用例（f64 参考如何构造）：`test/inductor/test_layout_optim.py` L95-L117

```python
fp64_mod = copy.deepcopy(mod).to(torch.float64)
fp64_inp = [t.to(torch.float64) for t in copy.deepcopy(inp)]
fp64_out = wrap_mod(fp64_mod)(*fp64_inp)
...
self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))
```

这是**同模型、输入升 f64、在 f64 下求值**的 f64 参考模式；`benchmarks/dynamo/common.py` 里也有
`cast_to(torch.float64, model, inputs)`（L1594）与 check_accuracy 路径。

**这三点直接回答本票的核心机制问题**：(a) 第一方承认同设备参考（eager）不够，需要 f64 仲裁；
(b) 仲裁方式是“被测实现相对 f64 的误差 不大于 基线实现相对 f64 的误差”，而不是“与 f64 参考相等”；
(c) 容差本身就是逐 op/逐规模挑选的经验值（multiplier 与 `numel` 相关），没有从中推导出的常数。

来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_dynamo/utils.py#L2773-L2793>、
#L2913-L3010；<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/inductor/test_layout_optim.py#L95-L117>。

### 3.5 OpInfo 的 `ref` 是 NumPy（CPU）参考

`torch/testing/_internal/opinfo/core.py`（v2.8.0）顶部注释列出 OpInfo 自动保证的性质，其中：

```
#   - that the operator produces the same results as a NumPy reference (if ref is defined)
#   - that the operator produces the same results as a NumPy reference on an extended
#       set of "reference inputs" (if both ref and reference_inputs_func are defined)
```

（L538-L542）。也就是说 PyTorch 把“与 NumPy 参考一致”写成了算子契约的一部分，NumPy 是纯 CPU 实现。

进一步，`test/test_ops.py::_ref_test_helper`（L448-L588）在比较失败时采用的判据是**“谁离精确计算更近”**：

```python
precise_dtype = torch.double          # f32/f16 算子的“更精确计算”就是 f64
...
ref_distance   = Σ |ref   - precise_result|
torch_distance = Σ |torch - precise_result|
self.assertTrue(ref_distance <= torch_distance,
                msg="Reference result was farther ... than the torch result was ...")
```

（L507-L585）。这是第一方把“参考实现应当比被测实现更准”写成断言的实例，也是“CPU 参考不是绝对真值、
它只是另一条有误差的计算路径”的源码级确认。

来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/opinfo/core.py#L538-L542>、
<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_ops.py#L448-L588>。

### 3.6 MPS 特定：不支持 f64；exp 的 CPU/MPS 实测差异有记录

`test/test_mps.py`（v2.8.0）：

- `test_double_error`（L11895）断言在 MPS 上构造 `torch.float64` 必须抛
  `TypeError: the MPS framework doesn't support float64`；`a.double()` 同样抛。
  **这意味着在 MPS 设备上不可能算 f64 参考**——f64 参考只能在 CPU（或别的设备）上求值。
- `test_exp`（L728）用 `self.compare_with_numpy(torch.exp, np.exp, a)`，即 MPS 结果对 **NumPy
  （CPU）** 参考（`compare_with_numpy` 由 `common_utils` 提供，内部走 `assertEqual`）。
- `test_exp1`（L759-L768）直接对 CPU 结果比较，并且把一次真实差异写进了注释：

  ```python
  self.assertEqual(output, output_cpu, atol=1e-8, rtol=1e-8)
  # If exponentWithTensor: MPS call is used on M1 running 14.5 test will fail with
  # Mismatched elements: 3 / 4 (75.0%)
  # Greatest absolute difference: 1.1920928955078125e-07 at index (3,) (up to 1e-08 allowed)
  # Greatest relative difference: 1.0786502002702036e-07 at index (3,) (up to 1e-08 allowed)
  ```

  第一方自己的记录显示：**同一个 `exp` 在 MPS 与 CPU 上的差异会超出它自己设的 1e-8 判据约 12 倍**，
  且取决于 macOS 版本与 GPU。这正是“超越函数必须单独定判据、不能沿用逐元素四则运算的直觉”的
  一手证据。

来源：<https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_mps.py#L728-L775>、
#L11895-L11903。

## 4. 其它 GPU 编译器/库的第一方实践

### 4.1 CUTLASS：CPU host reference，默认按精确相等比较，相对比较是 opt-in

- device GEMM 测试的校验路径（`test/unit/gemm/device/gemm_testbed_3x.hpp`）：
  `verify()` → `cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params)`（L2802）在 **CPU**
  上算 `reference_D`，再 `compare_reference` → `equality_check(reference_D, tensor_D)`（L1788）。
- 参考实现的累加类型是 **kernel 自己的 `ElementAccumulator`**
  （`tools/util/include/cutlass/util/reference/host/gett.hpp` L837 起，`Gemm3x` 用
  `typename MainloopParams::ElementAccumulator` 重建 `GettMainloopParams`）。
  **即 CUTLASS 的 host reference 是“同一算法契约的 CPU 实现”，不是更高精度的真值近似。**
- 比较方式默认 **EXACT**：`CheckEquality check_relative_equality = CheckEquality::EXACT;`
  （L388、L675、L1007、L1315、L1662、L1970 …）。只有显式选择 `RELATIVE` 时才走
  `cutlass::reference::host::TensorRelativelyEquals(lhs, rhs, epsilon, nonzero_floor)`，
  而该路径里 `Element epsilon(static_cast<Element>(0.1f))`（L592、L1754、L2232），
  `nonzero_floor = std::numeric_limits<Element>::min()`（L593）。
- 相对判据本身不看 ULP：`include/cutlass/relatively_equal.h` L58-L78 的
  `relatively_equal_float` 用 `diff < epsilon * (|a| + |b|)`（若两者都接近 0 则用
  `diff < epsilon * nonzero_floor`），注释说明它“inspired by floating-point-gui.de”。
- 另有独立的误差度量工具 `tools/util/include/cutlass/util/reference/host/error_metrics.h`：
  `TensorRelativeErrorMetric` = `||A_computed - B_reference|| / ||B_reference||`，
  `ComputeType` 默认 `double`。

**关键含义**：CUTLASS 的 CPU 参考之所以能按 EXACT 比较，是因为它是**同一舍入/累加契约的 CPU 复刻**；
它的 `epsilon=0.1f` 相对比较则承认跨实现差异可以大到 10%。这两条并存，说明“CPU 参考”在 CUTLASS 里
是**算法模型**，而不是“正确性真值”。这与本票设想的“f32 输入 + f64 精确参考”是**不同种类**的参考。

来源：<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/test/unit/gemm/device/gemm_testbed_3x.hpp#L583-L605>、
#L2802、<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/util/include/cutlass/util/reference/host/gett.hpp#L837-L895>、
<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cutlass/relatively_equal.h#L58-L78>、
<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/util/include/cutlass/util/reference/host/error_metrics.h#L51-L60>。

### 4.2 CUTLASS profiler：性能与校验分离，且校验可以拿 vendor 库当参考

`media/docs/cpp/profiler.md`（CUTLASS main）明列：

```
--mode=profile     regular verification and profiling (default)
--verification-enabled=<bool>   Whether to perform verification checks.
--epsilon=<error>  Error threshold. Setting to zero (default) requires bit-level equivalence.
--verification-providers=<providers>  List of providers used to verify result. (default: '*')
      Gemm   verification-providers {cublas*}
      Conv2d verification-providers {cudnn*, device*, host}
```

（L186、L266-L282）；正文 L330-L331：“The CUTLASS Profiler can be built with cuBLAS enabled to use as
a reference implementation.”；示例输出里出现 `cuBLAS: Passed`（L489）与
`reference_device: Passed`（L729）。

**含义**：(a) 性能工具与数值校验是两个独立开关，性能基准不承担正确性职责；(b) 第一方把
**vendor 库（cuBLAS）**、**设备侧参考**、**host（CPU）参考**并列为 verification provider，
说明“参考放哪算”在第一方是可选项而不是教条；(c) 默认 `epsilon=0` 即为 bit-level equivalence，与
Triton 的零容差用例（§4.3）呼应。

来源：<https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/profiler.md#L260-L290>。

### 4.3 Triton：逐元素用 NumPy（CPU）参考；部分算子用同 GPU 的 torch 参考；精确算子用零容差

`python/test/unit/language/test_core.py`（Triton main）：

- 通用逐元素测试生成器 `_test_unary`（L165-L190）：输入用 `numpy_random(...)`，参考是
  **CPU 上的 NumPy 表达式求值** `z_ref = eval(expr ...)`（L184），然后
  `np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)`（L190）。`_test_binary`（L224-L297）
  同构，`rtol=0.01`。
- 也有**同 GPU** 的 torch 参考：`test_erf` 一类走
  `z_ref = torch.erf(x)`（L1133，`x` 在 `device=device` 即 GPU 上）→
  `torch.testing.assert_close(z_tri, z_ref)`（L1136）。**这是“同设备参考”的实例**。
- **零容差**用例存在，用于精确语义：
  `torch.testing.assert_close(z, z_ref, atol=0, rtol=0)`（L476）、
  `torch.testing.assert_close(out_min, inp, atol=0, rtol=0)`（L2265-L2266）、
  `np.testing.assert_array_equal(...)`（如 L506、L2604、L2661）用于位级/整数语义。
- `python/triton/testing.py` 同一文件里既有：`assert_close(x, y, atol=None, rtol=None, ...)`
  （L438，文档写明默认 `atol=1e-2, rtol=0`，float 比较经 `np.testing.assert_allclose`，bf16 先升
  f32，且 `equal_nan=True`），又有 `do_bench(...)`（L288）。两者互不调用。
- Triton 还自带一个第一方解释器作为语义参考实现：`python/triton/runtime/interpreter.py`（数据用
  `numpy.ndarray` 承载，见 L29-L38 的 `TensorHandle`），其测试用
  `np.testing.assert_array_equal` 等做精确比较（`python/test/unit/runtime/test_interpreter.py` L27、
  L35、L70、L123）。**这是“同一实现的两个执行后端互为参考”**，其独立性比 CPU/GPU 更低。

来源：<https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/test/unit/language/test_core.py#L165-L190>、
#L1128-L1136、#L2263-L2267、<https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/triton/testing.py#L438-L489>、
同文件 #L288。

### 4.4 JAX / XLA：NumPy（CPU）参考 + 按 dtype/按算子的经验容差；并区分两类检查

`jax/_src/test_util.py`（JAX main）：

- `_CheckAgainstNumpy(self, numpy_reference_op, lax_op, args_maker, ...)`（L1467-L1475）：
  `numpy_ans = numpy_reference_op(*args)`，`lax_ans = lax_op(*args)`，再 `assertAllClose`。
  **NumPy 在 CPU 上求值，lax_op 在 JAX 的当前后端（GPU/TPU/CPU）上求值。**
- `_CompileAndCheck(self, fun, args_maker, ...)`（L1420-L1464）：比较 `api.jit(fun)` 的结果与
  `fun` 在 Python/逐 op 执行的结果，两者都在**同一设备**上。**这是同设备自洽，不是数值真值检查。**
- 默认容差是 dtype 表，来自 `jax/_src/public_test_util.py`：
  f32 `1e-6`、f16 `1e-3`、bf16 `1e-2`、f64 `1e-15`、整数与 bool 为 `0`（`_default_tolerance`，
  L46-L91）。`_assert_numpy_close` 把容差**按元素个数放大**：
  `atol=atol * a.size, rtol=rtol * b.size`（L164）。**这是第一方承认误差随归约长度增长。**
- 逐算子容差差异很大，且是手写的（`tests/lax_numpy_test.py`）：
  `testDot` L574-L585 `tol = {float16: 1e-2, bfloat16: 1e-1, float32: 2e-5, float64: 1e-14, complex128: 1e-14}`；
  `testMatmul` L622-L632 `tol = {float16: 1e-2, float32: 2e-2, float64: 1e-12, complex128: 1e-12, bfloat16: 1e-1}`。
  **同一个 f32，dot 用 2e-5，matmul 用 2e-2，相差 1000 倍**，说明容差是按算子形态实测标定的，
  而不是从 ulp 推导的常数。
- 重要细节：NumPy 参考函数被 `jtu.with_jax_dtype_defaults(...)` 包装（如 L466、L491、L2283），
  其输出 dtype 跟随 JAX 的默认 dtype。因此**在关闭 x64 时，NumPy 参考也是 f32**——此时“CPU 参考”
  只是“另一个 f32 实现”，并不更精确。

XLA 侧提供了与“误差界是逐算子参数”一致的基础设施：`xla/error_spec.h` 定义
`ErrorSpec{double abs; double rel; ...}`（L23-L31），并包含 `expect_inexact` 之类开关，其中注释
写明“integer comparisons in Near() allow a maximum difference of 1 between expected and actual values.
This accounts for differences between round-to-nearest semantics in fused hardware operations
(e.g. cuDNN fused convolutions) and standard round-to-zero (truncation) semantics in reference
evaluation.”（L54-L58）。**第一方显式承认参考求值与硬件融合运算的舍入语义可以不同。**

来源：<https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/jax/_src/test_util.py#L1420-L1475>、
<https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/jax/_src/public_test_util.py#L46-L91>、
同文件 #L159-L165、<https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/tests/lax_numpy_test.py#L574-L585>、
#L622-L632、<https://github.com/openxla/xla/blob/6c5e7717f9af43bb5256d667c9e910be450345ae/xla/error_spec.h#L23-L60>。

### 4.5 cuBLAS 的“可复现”声明说明了什么、没说明什么

cuBLAS 文档 “Results Reproducibility”：

> “By design, all cuBLAS API routines from a given toolkit version, generate the same bit-wise results at
> every run when executed on GPUs with the same architecture and the same number of SMs. However,
> bit-wise reproducibility is not guaranteed across toolkit versions… This guarantee no longer holds when
> multiple CUDA streams are active or fixed-point emulation is used.”

同一节说明绕开多流不确定性的手段（每流独立 workspace / 每流一个 handle / 用 `cublasLtMatmul`
配合用户 workspace / 设 `CUBLAS_WORKSPACE_CONFIG`），并说明 `symv`/`hemv` 走 atomic 路径时结果不保证
bitwise 可复现。

**含义**：同设备/同库的 bitwise 一致是**稳定性**保证，它在定义上不能发现该库自身的系统性错误。
把它当作正确性依据，等于假设“既有实现是对的”——而 map #2 明确排除了这一假设。

来源：<https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>（cuBLAS 13.4，
抓取日 2026-09-14），及同页 “GEMM Algorithms Numerical Behavior”。

## 5. 数值机理：为什么 CPU 参考不是绝对真值

### 5.1 IEEE 754 层面：不结合、不保证 bitwise

已在 §3.1 引用 PyTorch 官方表述（“floating point addition and multiplication are not associative, so
the order of the operations affects the results”），并有 `A.sum(-1)[0] != A[:,0].sum()` 的具体例子。
同一段还明确说 CPU 与 GPU 即使输入 bitwise 相同、随机性受控也可以不同。**因此“CPU 参考 = 真值、
GPU 必须与之相等”这一命题在一手资料中没有支持。**

### 5.2 Metal 侧的实际自由度（规范原文）

Apple MSL 规范 **4.1** 的相关条款：

- §1.6.3 “Math Intrinsics Compiler Options”（p.15-16）列出 fast math 包含的优化：
  `No NaNs`、`No INFs`、`No Signed Zeroes`、`Allow Reciprocal`、`Allow Reassociation`
  **`Allow Contract`（把乘法+加法融合成 FMA）**。并给出三个开关：
  - `-fmetal-math-fp32-functions=<fast|precise>`，**“The default is fast.”**
  - `-fmetal-math-mode=<fast, relaxed, safe>`，**“The default is fast.”**
    （`relaxed` 保留 INF/NaN 但仍允许 no-signed-zeros / reciprocal / reassociation / contract；
    `safe` “disables unsafe floating-point optimizations … This sets the FP contract to on.”）
  - legacy：`-ffast-math` ≡ `fast + fast`；`-fno-fast-math` ≡ `precise + safe`。
  - 收缩控制：**默认允许收缩**（`a*b+c` 可变成一条 FMA）；`-ffp-contract=off` 关闭；也可用
    `#pragma METAL fp contract([off|on|fast])` 与 `#pragma METAL fp math_mode([relaxed|safe|fast])`
    在源码段级控制。
- §8.1（p.368）：INF 必须支持；**NaN 只在关闭 fast math 时必须支持，开启 fast math 时 NaN/INF 行为
  未定义**；非规格化数“may be flushed to zero”。
- §8.2（p.368）：**单精度舍入模式“either round ties to even or round toward zero”**——即舍入模式本身
  由实现选择。这是 Table 8.1 “Correctly rounded” 的一个未定项，本文把它记为**未确认**（§9）。
- §8.4（p.368）：“Table 8.1 describes the minimum accuracy of single-precision floating-point basic
  arithmetic operations and math functions given as ULP values. **The reference value used to compute
  the ULP value of an arithmetic operation is the infinitely precise result.**”
  ——Apple 自己就是用“无限精确结果”定义精度，这与“更高精度参考”的思路一致。

**Apple 官方 API 侧的默认值与规范文本存在分歧，必须两个都记**：

- `MTLMathMode` 的 abstract：**“An indication of whether the compiler can perform optimizations for
  floating-point arithmetic that may violate the IEEE 754 standard.”**
- `MTLMathMode.fast`：“This is the default for Intel and AMD devices.”
- `MTLMathMode.relaxed`：“**This is the default for Apple silicon devices.**”
- `MTLMathFloatingPointFunctions.fast`：“This is the default behavior.”
- `MTLCompileOptions.mathMode` 的 Discussion 给出与旧 `fastMathEnabled` 的映射：
  `true → mathMode=fast + mathFloatingPointFunctions=fast`；
  `false → mathMode=safe + mathFloatingPointFunctions=precise`。

即：**规范说 CLI 默认是 `fast`，而 Apple 文档说 Apple silicon 设备的默认是 `relaxed`**；二者都允许
reassociation 与 contraction。**要在 Metal 上得到 IEEE-754 一致的行为，必须显式设置
`safe` + `precise`（等价于旧的 `-fno-fast-math`），不能依赖默认值。** 这一条对 tile-rs 直接相关，
因为被测对象是 Apple GPU 上的生成 MSL。

来源：MSL 规范 4.1 §1.6.3（p.15-16）、§8.1–§8.4（p.368 起）、Table 8.1/8.2（p.368-372）；
<https://developer.apple.com/documentation/metal/mtlmathmode>、
<https://developer.apple.com/documentation/metal/mtlmathmode/fast>、
<https://developer.apple.com/documentation/metal/mtlmathmode/relaxed>、
<https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode>、
<https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions/fast>。

### 5.3 超越函数不是正确舍入，且 fast math 下误差随参数放大

MSL 规范 **Table 8.1**（“Accuracy of single-precision floating-point operations and functions”，
p.368-370，fast math 关闭时的最小精度）：

| 函数 | 最小精度 |
|---|---|
| `x + y` / `x - y` / `x * y` / `1.0 / x` / `x / y` | Correctly rounded |
| `fma` | Correctly rounded |
| `exp` / `exp2` / `exp10` / `log` / `log2` / `log10` | `<= 4 ulp` |
| `sin` / `cos` / `sinh` / `cosh` | `<= 4 ulp` |
| `tan` / `atan2` / `tanpi` | `<= 6 ulp` |
| `pow` / `powr` | `<= 16 ulp` |

MSL 规范 **Table 8.2**（“with fast math enabled (which is the default unless you specify
`-fno-fast-math`)”，p.370-372）：

| 函数 | fast math 下的最小精度 |
|---|---|
| `x + y` / `x - y` / `x * y` | Correctly rounded |
| `1.0 / x` | `<= 1 ulp for x in … 2^-126 to 2^126` |
| `x / y` | `<= 2.5 ulp for y in … 2^-126 to 2^126` |
| `exp` / `exp2` | **`<= 3 + floor(fabs(2 * x)) ulp`** |
| `cos` / `sin` / `cospi` / `sinpi` | 在 `[-pi, pi]`（或 `[-1,1]`）内最大绝对误差 `<= 2^-13`，区间外更大 |
| `rsqrt` | `<= 2 ulp`（Table 8.1 中是 correctly rounded） |
| `pow` / `powr` | 实现为 `exp2(y * log2(x))`，`x=0, y=0` 未定义 |
| `fmod` | **Undefined**（Table 8.1 中为 `0 ulp`） |

**对本票范围的含义**：

1. `add/sub/mul` 在两种模式下都是 correctly rounded，**但它们不结合**（§5.1、§5.4）。
2. **`exp` 从来不是正确舍入**：关闭 fast math 也有 `<= 4 ulp` 的偏差；开启 fast math 时误差上界
   随 `|x|` 线性增长（`x = 20` 时 43 ulp）。softmax 的输入是 logits，logit 幅度越大，这个上界越松。
   **因此对 `exp` / softmax 不能要求“与 CPU 的 f32 `exp` bitwise 相同”，也不能把一个与 `|x|` 无关的
   固定容差当成有依据的判据。**
3. `pow` 的 16 ulp 与 `fmod` 的 undefined 说明“逐函数 ulp 表”是必须逐函数查的，不能推广。
4. 对照来源（不同厂商的同类声明，用于说明这不是 Apple 特有）：CUDA 12.6 编程指南 **Table 13**
   “Single-Precision Mathematical Standard Library Functions with Maximum ULP Error” 中 `x+y`、`x*y`
   为 `0` ulp，`expf(x)` 为 `2 (full range)` ulp；该表的参照物被明确定义为 “a correctly rounded
   single-precision result obtained according to the round-to-nearest ties-to-even rounding mode”。
   而 intrinsic 版 `__expf(x)` 为 `2 + floor(abs(1.173 * x))` ulp
   （§13.2 Intrinsic Functions 与 Table 15：`-use_fast_math` 把 `expf` 映射到 `__expf`，并
   **“In addition to reducing the accuracy of the affected functions, it may also cause some differences
   in special case handling.”**）。

来源：MSL 规范 4.1 Table 8.1（p.368-370）、Table 8.2（p.370-372）；
<https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html> 之
§13.2、Table 13、Table 15（“Single-Precision Floating-Point Intrinsic Functions”）。

### 5.4 收缩/FMA 直接改变 `(a+b)*c` 这类表达式

- MSL 规范 §1.6.3：**默认允许 contraction**，“For example, a*b+c may be converted to a single
  fused-multiply-add. These contractions could lead to computation differences if other expressions are
  not contracted.”；关闭方式 `-ffp-contract=off` 或 `#pragma METAL fp contract(off)`。
- CUDA 指南同样：`*` 与 `+` 生成的乘加“will frequently be combined into FMADs”，而
  `__fadd_*` / `__fmul_*` 明确“the compiler never merges into FMADs”。
- PyTorch 的 `same()` 也把这种差异纳入判据：`tol` 之外还允许 multiplier 2–10 倍的相对 f64 误差
  （§3.4）。

**含义**：`(a+b)*c` 的 f32 结果与“CPU 上先加再乘”的 f32 结果可以差 1 ulp 量级，**这不是 bug**；
而把它与 f64 精确参考比较时，差异会同时包含“f32 舍入”和“是否收缩”两项。判据必须说明是否允许收缩，
否则同一份 MSL 在不同编译选项下会得到不同的通过/不通过。

### 5.5 归约顺序与长度

- 归约不结合：PyTorch 官方例子 `A.sum(-1)[0] != A[:,0].sum()`（§3.1）。
- JAX 默认容差按元素个数放大 `atol * a.size`（§4.4）。
- `index_add_` 等 CUDA 归约即使同设备也不可复现（§3.1、§4.5 的 atomic 说明）。
- CUTLASS 的 host reference 之所以能 EXACT 比较，是因为它复刻了同样的累加类型与顺序（§4.1）；
  一旦顺序不同，就必须改用 `RELATIVE`（epsilon 0.1f）。

**含义**：行求和、softmax 的分母、f32 矩阵乘的 k 维累加，都属于“允许不同求值顺序”的算子。
参考可以定“误差上界”，但**必须先确定被测 kernel 的累加精度与顺序契约**，否则上界无从谈起。

### 5.6 f32 输入先定稿、再升 f64：语义上与“用精确输入”等价

- Rust Reference `[expr.as.numeric.float-widening]`：**“Casting from an f32 to an f64 is perfect and
  lossless”**，并给出 `assert_eq!(1_234.5f32 as f64, 1_234.5f64)` 等示例。
  （tile-rs 的 kernel 与参考实现都在 Rust 生态里，这条直接适用。）
- C++ 工作草案 `[conv.double]`：**“If the source value can be exactly represented in the destination
  type, the result of the conversion is that exact representation.”**
- 因此：**任何有限的 f32 值在 f64 中都能精确表示**，`f32 → f64` 不会引入任何舍入。
  “把 f32 输入定稿后升 f64”与“把这些输入当作精确实数”在数值上等价（本票范围内的有限正常数与零
  尤其如此；非规格化数在 f64 中也精确，但 §8.1 说明 Metal 侧可能 FTZ，属于范围外）。

**这条推论的三个直接后果**（本票问题的一部分）：

1. **参考的输入与被测 kernel 的输入是同一批 bit**，所以测出来的差异**只反映 kernel 的运算舍入/顺序/
   函数近似误差**，不掺入输入表示误差。
2. **反过来，f64 参考也不会替被测 kernel 承担“输入本身就不精确”的问题**。如果上游给的是
   `float` 变量、其值本身就是某个物理量的近似，那么 f64 参考与 kernel 都只是在回答同一个精确问题，
   与“输入相对真实世界的误差”无关。这个参考能界定的误差类型，是**舍入误差**，不是**输入/模型误差**。
3. 注意 f64 不是无穷精度：n 项 f64 累加的自身误差约随 n 增长。**把 f64 参考当“零误差真值”是错的**
   （XLA 的 `ErrorSpec` 同时有 `abs` 与 `rel`，JAX/CUTLASS 都提供误差度量工具，§4.1、§4.4）。
   被测 kernel 与参考的差异是两者误差之差，判据若写成“差 ≤ 常数”，其中隐含了“参考误差可忽略”的
   假设——**这个假设需要被显式承认或显式补偿**。

来源：<https://doc.rust-lang.org/reference/expressions/operator-expr.html>（`float-widening` 小节）、
<https://eel.is/c++draft/conv.double>。

### 5.7 参考只能覆盖“算法规定的语义”，不覆盖“条件数”

PyTorch 官方文档对 linalg 的说明（同一 `numerical_accuracy.rst`）：

> “If provided with ill-conditioned inputs, the result of these functions they may vary when using the
> same inputs on different devices or when using different backends…”；并建议用
> `torch.linalg.svdvals` / `torch.linalg.cond` 检测。

同一文档的 “Extremal values” 一节给出 `a.norm()` 在 f32 下溢出为 `inf` 而 `a.double().norm()` 正常的
例子。**含义**：即使参考完全正确，`|kernel - ref|` 的大小也主要由输入的**条件数**与**动态范围**决定，
不由实现质量决定。判据要么按输入分布定，要么用相对量（PyTorch `assert_close` 的 `atol + rtol*|expected|`、
CUTLASS 的 `TensorRelativeErrorMetric`、JAX 的 size 缩放都是这类形式）。

## 6. 本票 8 类算子的逐条对照

下表只给“参考应当是什么、依据在哪、哪些差异是允许的”，**不给任何容差数值**。
“允许差异的来源”一列说明该算子为什么不可能 bitwise。

| 算子 | 语义 | 允许差异的来源（一手依据） | 对“CPU 结果作参考”的直接影响 |
|---|---|---|---|
| `add` / `sub` / `mul` | 逐元素四则 | 单次运算 correctly rounded（MSL Table 8.1/8.2）；但**不结合**，且可能被 reassociate/contract（MSL §1.6.3） | 单算子自身 CPU/GPU 应只差输入表示与是否收缩；若配上 f64 参考，差异上限可被算子的正确舍入性质界定 |
| `exp` | 超越函数 | **从不 correctly rounded**：precise `<= 4 ulp`，fast math `<= 3 + floor(\|2x\|) ulp`（MSL Table 8.1/8.2）；CUDA `expf` 为 2 ulp / `__expf` 随参数增长 | CPU 与 GPU 的 `exp` 实现不同，差异是**实现差异**而非 bug；PyTorch 自己也记录了 MPS 与 CPU 的 `exp` 差异超出 1e-8（§3.6） |
| `(a+b)*c` | 融合表达式 | 默认允许收缩成 FMA（MSL §1.6.3；CUDA 指南同）；`-ffp-contract=off` 可禁 | 判据必须先声明是否允许收缩；与 f64 参考的差异同时含“舍入”和“收缩”两项 |
| 行求和（reduction） | 有序累加 | **不结合**（PyTorch 官方例子）；误差随长度增长（JAX `atol*a.size`）；同设备也可能不可复现（atomic，PyTorch randomness 文档） | CPU 顺序与 GPU（stride loop / workgroup 归约）顺序不同；参考只应定误差上界，且上界要说明累加顺序/精度契约 |
| `softmax` | `exp → 行求和 → 除` | 叠加 `exp` 的非正确舍入与归约的非结合；若走 max-subtraction 还有额外的减法与除法（`x/y` fast math `<= 2.5 ulp`） | 最不适合用单一常数判据；PyTorch MPS 对 `exp` 都要单独写测试（§3.6） |
| f32 转置矩阵乘 | 累加 + 数据布局 | 累加顺序与 k 维分块；PyTorch 官方列 TF32/reduced-precision 为额外变量（§3.1）；JAX 对 `matmul` 的 f32 容差比 `dot` 大 1000 倍（§4.4）；CUTLASS 跨实现比较时用 0.1 相对 epsilon（§4.1） | CPU 参考有用，但差异主要由 k 维累加顺序与是否走 tensor-core/低精度路径决定；MPS 上不存在 TF32 这条，但不能假定其他低精度路径不存在 |
| `gather` | 纯数据搬运 | 无算术（比较见 §4.3 的 `assert_array_equal` / `atol=0, rtol=0`） | **应当按精确一致处理**：差异只能是 bug 或形状/index 错误，不该用容差掩盖 |
| `transpose` | 纯数据搬运 | 同上 | 同上；MSL 侧也没有舍入余地 |

## 7. CPU 参考与 GPU 参考分别能证明什么

| 命题 | CPU/f64 参考 | 同设备 GPU 参考 |
|---|---|---|
| 结果离数学值有多远（数值正确性） | **能**（受参考自身精度限制，§5.6） | 不能：与被测实现共享编译器/库，共模错误不可见（§4.5） |
| 新实现是否复现既有行为（回归） | 间接（行为不同也判失败） | **能**，且更贴近“行为不变”的意图（PyTorch `same` 的默认路径、JAX `_CompileAndCheck`、Triton 的 `torch.erf` 参考） |
| 是否违反算子契约（dtype/形状/别名/视图） | 不能单独覆盖（PyTorch OpInfo 的 `prims.utils.compare_tensor_meta`/view 检查是另一套测试，§3.5） | 不能 |
| 是否能发现 GPU 库/编译器自身的系统错误 | **可以**（不同实现、不同平台） | 不能 |
| 是否要求 bitwise 相等 | 不要求，也不应要求（PyTorch 官方声明，§3.1） | 同版本同架构下 cuBLAS 自己给出 bitwise 保证（§4.5），但这不是正确性 |
| 跨设备可移植性 | 直接给出（同一参考可用于多后端） | 不能，各设备参考不同 |

**独立性不是二值**：CPU 参考 ⊃ 独立于 GPU 编译器与 GPU 库；GPU 参考独立于 CPU 库但与 GPU 编译链
共模。map #2 的 Notes 已经把“既有实现不能自动当作正确性标准”定为约束，本节的表格是这一约束在
第一方资料中的对应证据。

## 8. 对本票尚未决定的选择（供 Q10 重新确认，本文不替用户选）

1. **参考的精度层级**：f32 CPU 参考、f64 CPU 参考、还是“f64 参考 + 与基线比较相对误差”（PyTorch
   `same()` 模式）？三者在第一方资料里都有实例，证明能力不同（§3.4、§4.4、§7）。
2. **f64 参考的输入处理**：沿用“f32 输入 bitwise 定稿后升 f64”（§5.6 证明其精确性），
   还是像 PyTorch `test_layout_optim.py` 那样连模型权重也升 f64（这会改变权重本身的舍入）。
   两者回答的问题不同。
3. **是否要求关闭 fast math**：MSL 的默认值在 CLI 文档与 Apple developer 文档之间不一致
   （§5.2），且 MPS 之外 tile-rs 走的是 `xcrun metal`。是否把 `-fno-fast-math`（`safe` + `precise`）
   作为验证产物契约的一部分？是否同时 `-ffp-contract=off`？
4. **逐算子判据的组织方式**：是每个算子一个界（JAX 的做法），还是按算子族给相对/绝对组合
   （PyTorch `atol + rtol*|expected|`），还是“被测实现相对 f64 的误差不大于基线相对 f64 的误差”
   （PyTorch dynamo）？本文不给数值。
5. **纯数据搬运算子（gather/transpose）是否单列零容差**：Triton 与 CUTLASS 都有零容差路径
   （§4.1、§4.3），但 PyTorch 的 OpInfo 默认比较走的是容差比较。
6. **参考实现的载体**：PyTorch CPU（本票问题原文）、NumPy、还是 tile-rs 自己的 Rust f64 参考？
   PyTorch 的额外约束是 MPS 不支持 f64（§3.6）与 `assert_close` 默认 `check_device=True`（§3.2），
   这两点会直接决定实现方式。
7. **“GPU 参考”是否纳入**：作为回归门禁（与 CPU 数值判据并列），还是明确排除。本文证明它与
   CPU 判据不可互相替代（§7）。
8. **参考本身误差是否需要量化**：例如对 f64 行求和给出自身误差界，或在判据中显式留出参考误差项
   （§5.6 第 3 点）。

## 9. 显式未知 / 不要假设

- **未实测**：本文没有任何 Apple GPU、Metal 工具链或运行实验。MSL §8.2 的“舍入模式可能是 RTZ”
  是否实际影响 Apple GPU 上的 add/mul（Table 8.1 的 “Correctly rounded” 到底相对哪种舍入模式），
  本文**未能确认**。这需要 runner 上的实测（sibling 票 #9）或本地 Apple 机器。
- **未确认**：`xcrun metal` 在 tile-rs 关心的 macOS 版本上是否接受 `-fmetal-math-mode` /
  `-fmetal-math-fp32-functions`（MSL §1.6.3 标注这些选项自 Xcode 16 / “Windows 5（iOS 18 或
  macOS 15 SDK）”起支持），本文只读到规范文本。
- **未知**：`MTLMathMode` 默认值在 CLI（规范说 `fast`）与 `MTLCompileOptions`（Apple 文档说
  Apple silicon 是 `relaxed`）之间不一致，实际生效值本文未能确认；也不确定两者是否等价。
- **未确认**：PyTorch 的 MPS 后端具体以什么编译选项调用 Metal（`fastMathEnabled` / `mathMode`），
  本文没有读 MPS 的相关源码；因此不能从 §3.6 的测试容差反推 MPS 的编译选项。
- **未覆盖（本票范围外，按用户指示）**：特殊值（NaN/INF）、非规格化数、低精度类型（f16/bf16/fp8）、
  性能基准与容差数值本身。
- **不假设**：不假设“CPU 与 GPU 必须 bitwise 相同”；不假设“CPU 参考是绝对真值”；不假设“同设备参考
  更独立”；不假设各第一方来源使用的容差数值对本项目同样适用（§4.4 的 2e-5 与 2e-2 相差 1000 倍
  已说明这些值是各自标定的经验值）。

## Sources

pinned 链接；行号对应 §1 的修订。Apple 的 MSL 规范与 Developer 文档无 commit，标为版本 + 抓取日。

**PyTorch（tag `v2.8.0` = `ba56102`）**

- Numerical accuracy: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/notes/numerical_accuracy.rst>
- Randomness: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/notes/randomness.rst>
- MPS backend: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/docs/source/notes/mps.rst>
- `torch/testing/_comparison.py`（容差表 L54-L60、`assert_close` L1317-L1340）: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_comparison.py#L54-L60>
- `torch/testing/_internal/common_utils.py`（`assertEqual` 的 `exact_device=False` 默认）: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/common_utils.py#L4060-L4135>
- `torch/testing/_internal/common_device_type.py`: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/common_device_type.py#L1475-L1521>
- `torch/testing/_internal/opinfo/core.py`: <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/testing/_internal/opinfo/core.py#L538-L542>
- `torch/_dynamo/utils.py` (`same`, `rmse`): <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/_dynamo/utils.py#L2773-L3010>
- `test/test_ops.py` (`test_compare_cpu`, `_ref_test_helper`): <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_ops.py#L373-L396>
- `test/test_mps.py` (`test_exp`, `test_exp1`, `test_double_error`): <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/test_mps.py#L728-L775>
- `test/inductor/test_layout_optim.py` (fp64_ref 构造): <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/test/inductor/test_layout_optim.py#L95-L117>
- `benchmarks/dynamo/common.py` (`cast_to(torch.float64, …)`): <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/benchmarks/dynamo/common.py#L1594>
- `torch/backends/cuda/__init__.py` (`allow_tf32`): <https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/torch/backends/cuda/__init__.py#L130-L144>

**CUTLASS (`main` @ `147295a`)**

- GEMM testbed（EXACT 默认、epsilon 0.1f、host reference 调用）: <https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/test/unit/gemm/device/gemm_testbed_3x.hpp#L583-L605>
- `Gemm3x` host reference（用 `ElementAccumulator`）: <https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/util/include/cutlass/util/reference/host/gett.hpp#L837-L895>
- `relatively_equal_float`: <https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/include/cutlass/relatively_equal.h#L58-L78>
- `TensorRelativeErrorMetric`: <https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/tools/util/include/cutlass/util/reference/host/error_metrics.h#L45-L60>
- Profiler（verification 开关与 provider）: <https://github.com/NVIDIA/cutlass/blob/147295a3d4b75f3aeff247c25b8927cea9a7006a/media/docs/cpp/profiler.md#L260-L290>

**Triton (`main` @ `59eaaff`)**

- `python/triton/testing.py`（`assert_close` L438、`do_bench` L288）: <https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/triton/testing.py#L438-L489>
- `python/test/unit/language/test_core.py`（NumPy 参考、torch 同设备参考、零容差）: <https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/test/unit/language/test_core.py#L165-L190>
- 解释器参考实现: <https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/triton/runtime/interpreter.py#L29-L38>、
  <https://github.com/triton-lang/triton/blob/59eaaff969e731cdea71188fc4aedbd33fa393fe/python/test/unit/runtime/test_interpreter.py>

**JAX (`main` @ `adb0562`) / OpenXLA (`main` @ `6c5e771`)**

- `jax/_src/test_util.py`（`_CheckAgainstNumpy`、`_CompileAndCheck`）: <https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/jax/_src/test_util.py#L1420-L1475>
- `jax/_src/public_test_util.py`（默认容差表、size 缩放）: <https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/jax/_src/public_test_util.py#L46-L91>
- `tests/lax_numpy_test.py`（`testDot`、`testMatmul`）: <https://github.com/jax-ml/jax/blob/adb0562417371429beddf7d575a0753dc957de19/tests/lax_numpy_test.py#L574-L585>
- `xla/error_spec.h`: <https://github.com/openxla/xla/blob/6c5e7717f9af43bb5256d667c9e910be450345ae/xla/error_spec.h#L23-L60>

**Apple**

- Metal Shading Language Specification, Version 4.1（2026-06-04）：<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
  （§1.6.3 p.15-16；§8.1–§8.4 p.368；Table 8.1 p.368-370；Table 8.2 p.370-372）
- `MTLMathMode`: <https://developer.apple.com/documentation/metal/mtlmathmode>
- `MTLMathMode.fast`: <https://developer.apple.com/documentation/metal/mtlmathmode/fast>
- `MTLMathMode.relaxed`: <https://developer.apple.com/documentation/metal/mtlmathmode/relaxed>
- `MTLCompileOptions.mathMode`: <https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode>
- `MTLMathFloatingPointFunctions`: <https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions>
- `MTLMathFloatingPointFunctions.fast`: <https://developer.apple.com/documentation/metal/mtlmathfloatingpointfunctions/fast>

**NVIDIA / 语言规范**

- CUDA C++ Programming Guide 12.6.0（§13.2 Intrinsic Functions、Table 13、Table 15、§16.3 Floating-Point Standard）: <https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html>
- cuBLAS 13.4 — Results Reproducibility: <https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility>
- Rust Reference — `[expr.as.numeric.float-widening]`: <https://doc.rust-lang.org/reference/expressions/operator-expr.html>
- C++ 工作草案 `[conv.double]`: <https://eel.is/c++draft/conv.double>
- NumPy `testing.assert_allclose`: <https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html>
