# GPU 正确性测试与 CI：同类项目对 Metal 方案的启示

研究票：[#44](https://github.com/Jopqior/tile-rs/issues/44)，所属地图：[#42](https://github.com/Jopqior/tile-rs/issues/42)，待复核方案：[#31](https://github.com/Jopqior/tile-rs/issues/31)。调查日期：2026-09-22。

## 结论与证据边界

1. **TileLang 已有免费标准托管 macOS 上真实执行 Metal 数值测试的公开先例。** 不只是目录里有测试：已追到 PR workflow、安装当前 checkout、pytest 选择器及实际 job 日志。该次 runner 是 `macos-26-arm64`，不是 #31 的 `macos-15`。它证明这类托管路径存在，不证明 tile-rs 当前 emitter、指定系统或全部首批用例可用。
2. 四个项目都把测试资产与运行器分开复用，但没有证据支持“必须采用某个框架”或“必须另外写一套独立用例”。Triton/PyTorch 大规模参数化，TileLang 复用参考函数，cuTile Rust 划分编译和 GPU 脚本。可迁移的是职责与可追溯性，不是它们的依赖栈、并行度或宽容差。
3. **GPU 用例存在、workflow 选择了目录、job 成功、某个数值用例实际执行通过，是四种不同证据。** cuTile Rust 公开 PR CI 在本次固定版本只证明 CPU/assembler 与 compile-only 路径，不能由 `run_gpu_tests.sh` 的存在推断 GPU CI。TileLang 的成功 run 同时含跳过；PyTorch 允许无 MPS 时替换测试基类；Triton 也有硬件条件 skip。
4. #31 的当前源码受测、同一批输入、全量比较、缺设备不能成功、必需清单完整、失败保真，应保留。独立入口的具体形态、PyTorch CPU 是否唯一参考、首批覆盖映射及扩展调度，应结合新增上游资产重审。以下均为**建议，未作架构或框架选型**。

本报告只读一手仓库、文档、Actions API 与已有日志，没有触发 CI、执行对方代码或进行 GPU 实测。下文“源码事实”只表示固定版本的接线；“运行证据”只表示所链接尝试；“建议”不冒充同行共识。研究对象按代表性路径深入追踪，并非穷举四个仓库的所有测试。没有找到公开调用链不等于不存在内部 CI。

## 1. 身份、固定版本与本项目基线

| 对象 | 明确身份与固定源码版本 | 排除歧义 |
| --- | --- | --- |
| Triton | [`triton-lang/triton@ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782`](https://github.com/triton-lang/triton/tree/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782)，语言和编译器仓库 | 不是 NVIDIA Triton Inference Server |
| TileLang | [`tile-ai/tilelang@6679f91387a2a94d858f0789478fb3461ab5a695`](https://github.com/tile-ai/tilelang/tree/6679f91387a2a94d858f0789478fb3461ab5a695)，tile DSL/编译器 | 不是 cuTile、CuTe DSL，也不是当前 tile-rs |
| PyTorch | [`pytorch/pytorch@a402343453f89b31a2ec8372b304accdc5b78e5f`](https://github.com/pytorch/pytorch/tree/a402343453f89b31a2ec8372b304accdc5b78e5f)，重点查 MPS 测试及通用比较器 | 参考库和被研究的 MPS 后端是两种角色，不以 MPS 比较代替 CPU 独立参考 |
| cuTile-rs | [`NVlabs/cutile-rs@3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b`](https://github.com/NVlabs/cutile-rs/tree/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b)，README 自称 **cuTile Rust**，crate 为 `cutile` | 没有替换为 NVIDIA 的 Python cuTile 或另一个 Rust DSL；`cuda-tile-rs` 是该仓库内的成员目录，不是另一研究对象 [C0] |

TileLang 的 `requires_metal` 来自其 TVM 子模块，不是本地同名函数。固定 gitlink 为 [`TileLang/tvm@907a88c8791ccf33b9874821bc875e7abf624367`](https://github.com/TileLang/tvm/tree/907a88c8791ccf33b9874821bc875e7abf624367)，已沿 `.gitmodules` 和 gitlink 追到 `python/tvm/testing/utils.py`。[L4]

本报告分支基于指定上游 `e8d1acdf58f4e2239bf6be9d59333d7a337282b3`。它与已交付入口合并提交 `4e66bc824872a95bfae51f6503bfbdb360199fe3` **互非祖先**，共同祖先是 `c3c8ec0169bd02757b51370ba9c6ec3b116b833d`；后者是规划基线 `b1772be1df05edeb475cb7ee16e644a368c2e04c` 的祖先。故此 checkout 没有 `metal_correctness` 不能解释为未交付。已交付 README 明确只有 8 元素 `add_f32_small`，保持输入与两个输出 pad，关闭 fast-math，使用 PyTorch CPU，且明确不等于 #31 首批完成。[B1]

上游已有真实数值测试资产。例如 `crates/tile_codegen/tests/msl_flash_attn_ref.rs` 经 `TargetRegistry` 生成当前 MSL、Metal 编译/执行/回读，与 Rust CPU 参考比较；也有无设备直接 return。`msl_ported_iquant_contract.rs` 则主要是文本/调度契约检查。这两类不能按同一个“测试数”计入数值覆盖。这里仅用于说明同行做法的迁移落点，不替代上游全量盘点，也不把 attention 纳入旧首批承诺。[B2]

## 2. 从测试追到真实 workflow

### 2.1 Triton：当前源码安装与分层执行，另有 warmup 完整性检查

**源码调用链：** `.github/workflows/ci.yml` → `runner-preparation.yml` → `integration-tests-nvidia.yml` → `make dev-install`、`make NUM_PROCS=24 test-unit` → `Makefile` → `python -m triton._test_runner suite unit` → `_unit()` 在 `python/test/unit` 中启动 pytest → `language/test_core.py`。lit、C++、interpreter、regression、GSan、Gluon 是分开的步骤，不能把 interpreter 或 lit 的通过计为 GPU 数值通过。[T1][T2]

- PR 和手动触发启用 integration；main push 按构建依赖变化或距上次成功至少四小时启用。不是每次 main 更新均跑，也不是一个定时 nightly。NVIDIA matrix 为 A100/H100/GB200；GB200 job 有 `continue-on-error`。这些是资源/稳定性政策，不能直接继承为本项目必需检查的成功规则。[T1]
- `_validate_gpus()` 在可见设备数不足时退出；测试内部仍可按 dtype、SM 能力跳过。完整性步骤调用 `report --require-complete`，检查 warmup 有运行事件、cache hit、已 warmup 的测试是否未执行、是否有未用 specialization。它确实有漏跑检测，但基准集合来自 warmup trace，不等于一份独立声明的“全部必需数值用例且断言成功”清单。[T2][T4]
- **运行证据：** [run 35692993800](https://github.com/triton-lang/triton/actions/runs/35692993800)，API `head_sha=45c57048e60a7981203852d00af6886a42254b3a`；[H100 job 106633786606](https://github.com/triton-lang/triton/actions/runs/35692993800/job/106633786606) 的 API 步骤显示 `Run triton tests`、regression、GSan/interpreter 均成功。本次 H100 日志下载遇 TLS 超时，**不声称已逐用例核实该 run 的 `test_reduce1d`**。运行提交与上面的源码分析 pin 不同，也不把 PR head 当作已核实的 checkout merge SHA。

### 2.2 TileLang：真正的 hosted Metal 路径，同时包含 codegen 与 skip

**源码调用链：** `.github/workflows/ci.yml` 的 `tests` matrix 包含 `[macos-latest]`、`toolkit: Metal` → `USE_METAL=ON` → 安装测试依赖 → `uv pip install -v .` → `cd testing; pytest ... -k metal ./python`。`testing/conftest.py` 将仓库根置于 `sys.path`，固定 Python/Torch/NumPy 随机种子；`test_metal_gemm_v2.py` 使用 `@tilelang.jit`，在 `mps` 张量上执行并比较矩阵乘输出。这不是只编译或下载现成 TileLang 发布包。[L1][L2][L3]

- CI 触发为 PR/手动，owner 限制、draft 条件与 lint 前置可能使数值 job 不执行。`-k metal` 选中的还有 Linux 可运行的 Metal codegen 测试，选择名称不代表每个 item 都执行了 GPU。
- **已查实际日志：** [run 35711502646 / Metal job 106693322060](https://github.com/tile-ai/tilelang/actions/runs/35711502646/job/106693322060)。PR head 是 `885476d418d9dbff3ad6bd7abe8cafd9dae6d24d`，日志 checkout 为 merge **`0baf77d4e5dbb53b1b10d6d11b46fb822ebed032`**，合入的基线正是本报告 TileLang pin。runner image `macos-26-arm64`，OS `26.6.2`。pytest 汇总 **82 passed、4 skipped、88 warnings，107.05 秒**。`test_gemm_v2_16x16x16`、`16x16x8`、`large`、`cooperative_tensor_non_square`、`1024` 均有 PASSED 行；三个 global-C Metal4 测试是 SKIPPED。
- 数字只描述这个混合选择集，不能称为“82 个 GPU 数值测试”。这次 checkout/install 到测试结束约半小时，测试本身不到两分钟，说明构建成本值得单独量测；不将一次记录当作稳定时长预算。日志没有给出足够的 Metal 设备身份，不能补写物理 GPU 型号。
- `dist.yml` 有每日 schedule，另有 self-hosted CUDA/ROCm 的 built-wheel GPU smoke job；该 job 明确不阻塞发布。它是产物验证层，**不是本次 Metal 首批 nightly 全覆盖证据**。“Nightly-Metal”配置词也可能仅指安装 torch nightly，不代表测试调度。[L6]

### 2.3 PyTorch：MPS 的 CPU 对照真正接入了 CI，但不是标准免费 runner

**源码调用链：** `mac-mps.yml` → `_mac-build.yml`/`_mac-test.yml`（下载同一 CI build 产物）→ 环境 `TEST_CONFIG=test_mps` → `.ci/pytorch/macos-test.sh` 的 `test_python_mps()` → `python test/run_test.py --verbose --mps` → 明确选入 `test_mps`、`test_ops`、`test_metal`、NN/Inductor 等测试。另用 `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` 单独运行 `test_add_sub`，用于 buffer address space 诊断。[P1][P2]

- `mac-mps.yml` 是 `ciflow/mps/*` tag push 与手动触发，不是“所有 PR 都直接触发 MPS”。`trunk.yml` 也包含 `config: mps`，且受 job-filter 影响。全库还有 periodic/slow 等层，不意味着每个 MPS 测试每天都执行。需分别核对所宣称层级的选择条件。[P1][P5]
- `macos-m1-stable`、`macos-m1-14`、`macos-m2-15`、`macos-m2-26` 是项目专用 runner label；`_mac-test.yml` 还有 pet runner 清理。不能把它们视为 GitHub 的 `macos-15` 标准托管规格。[P1]
- **已查实际日志：** [run 35703328812 / M2-15 job 106670935381](https://github.com/pytorch/pytorch/actions/runs/35703328812/job/106670935381)，checkout **`eaff5f67eab85ae80ee65d972bfff6f1e066e5c4`**，MPS job 成功；独立 validation 步骤记录 `TestMPS.test_add_sub` 与 `test_add_sub_alpha_cast` 均 `ok`，共 2 tests。这证明具体测试确有执行，不证明无 skips，也不是本报告源码 pin 的全量通过声明。

### 2.4 cuTile Rust：GPU 运行脚本不是公开 GPU PR CI

**公开 CI 调用链：** `.github/workflows/pr.yml` 在 `push: pull-request/[0-9]+` 上运行，`linux-amd64-cpu16` + CUDA devel container，执行 `cargo test --no-run`、`scripts/run_cpu_tests.sh`、assembler 验证和其他 CPU 检查。CUDA 13.4 lane 还取决于 repository variable。本次枚举全部公开 workflow（pr、pages、cargo-deny、codeql），**未找到对 `scripts/run_gpu_tests.sh` 的 workflow 调用**。[C1]

**本地 GPU 调用链：** `run_gpu_tests.sh` 明确列出 integration targets，包括 `slice_non_divisible`、`gpu_execution_ops`、`tensor_and_matrix_ops execute_`、aggregate `gpu`；后者包含 `gpu/add_basic.rs`。调用实际 JIT kernel，经 `.to_host_vec().sync()` 回读再断言。`test_runner_common.sh` 逐步骤积累失败，最后退出 1，不因某个后续步骤成功覆盖先前失败。[C2][C3]

[run 35682712783 / build job 106604559303](https://github.com/NVlabs/cutile-rs/actions/runs/35682712783/job/106604559303)，API head `a8df83823cbce114c1c09856d7290e9f24401cb3`，显示 compile-only、CPU-only 步骤成功，13.4 lane skipped。这里仅查 job API，没有逐条 CPU 测试日志；既不能称其 GPU 回归通过，也不能推断维护者从未在别处运行 GPU 脚本。

## 3. 按关键问题跨项目比较

### 3.1 参考、输入、dtype、容差和数学模式

| 项目与具体路径 | 源码事实 | 迁移判断 |
| --- | --- | --- |
| Triton `test_core.py::test_reduce1d`、`test_dot` | reduction 使用 NumPy 参考，同一输入转设备；整数 sum 精确比较，浮点 sum `rtol=0.01`，bf16 有显式舍入处理。dot 区分 `ieee/tf32/tf32x3/bf16x3/bf16x6`，设置 PyTorch TF32 开关，并按分支选容差；还有结果与汇编检查。`numpy_random` 默认 seed 17。[T3] | 独立参考不等于必须 PyTorch，也不等于必须 CPU；需要记录舍入/累加模式。不能把 reduction 的 1% 容差迁到 Metal f32 默认策略，更不能把汇编检查当数值验证。 |
| TileLang `test_tilelang_kernel_gemm.py` | 参考 callable 复用输入，转换到 torch float 做 matmul，再转输出 dtype；有 tfloat32 特殊处理，打印生成源码。`Profiler.assert_allclose` 默认 `atol=rtol=1e-2`，允许 `max_mismatched_ratio=0.01`，只比较输出。[L5] | callable 接口适合复用；容忍一定比例错误与 #31 全元素合格冲突，应拒绝照搬。此 profiler 还直接同步 CUDA，不是现成 Metal 通用运行器。 |
| TileLang `test_metal_gemm_v2.py` | reference 在 **MPS** 上 `a.to(accum_dtype) @ b.to(accum_dtype)`；一般 `torch.allclose(..., atol=1e-2)`，1024 case `atol=1.0`，Metal4 fp16 case 又有专门容差。不是 PyTorch CPU reference。[L3] | 与 TileLang 编译器不同实现，但共享设备/后端，独立性弱于本项目 CPU 对照目标。不能将这些容差解释为任意 f32 误差预算。 |
| PyTorch `test_mps.py::test_add_sub` | CPU 生成 f16/f32 数据，clone 到 MPS；相同操作 CPU/MPS 比较。f16 alpha 有显式 `2e-3` 容差及待修说明。大量其他操作有独立 dtype 容差/skip/expected failure。[P3] | 最直接支持“同一批已量化输入，跨后端参考”的先例。内部 `TestCase.assertEqual` 不是外部 `torch.testing.assert_close` 的同义接口，不能混同默认值。 |
| cuTile Rust `gpu/add_basic.rs`、`slice_non_divisible.rs` | 简单 add 用常量 1+1=2 锚点；逐项 abs error `<1e-6` 或 `<1e-5`。非整齐长度检查 `host.len()`，打印失败 index。无需大型参考框架。[C3] | 简单算术与搬运可以手算/Rust 参考；常量不能发现全部索引置换，需要补坐标编码/非对称输入。不能据 smoke case 推导整个项目统一数值政策。 |

#31 的 f32 `rtol=1.3e-6, atol=1e-5` 确是所查 PyTorch **公开** `torch.testing.assert_close` 默认值，定义在 `_comparison.py`；`equal_nan=False`，默认检查 dtype/device 等属性。[P4] 这是工具默认值的事实，**不是对任意长 K 或归约树的精度证明**。建议保留默认值作为起点以及“先诊断，后逐例审查例外”，不是先扩大容差求绿。

另需注意，`equal_nan=False` 本身不等于拒绝 Inf；同号 Inf 可以被 close 判定相等。#31 的有限输入域应继续在比较前独立要求参考和实际结果均有限。TileLang 底层 `torch_assert_close` 默认 `equal_nan=True`、允许非零 mismatch 比例，与这条要求不同。[L5][P4]

数学模式应按用例记录：IEEE/fast-math、FMA、累加 dtype、转换位置不能从框架名称推断。同行已经把模式放进参数或分支，支持 #31 显式声明 fast-math 关闭，但没有提供跨所有 Metal 设备的误差保证。独立 CPU 参考也需要手算锚点确认轴、布局与公式，不能只因“用了 torch”就相信参考正确。

### 3.2 形状与边界：针对失败机制，而不是最大化笛卡尔积

- **Triton：** reduction 按 op/dtype/shape/CTA 参数化；32、64、128、512 覆盖实现分组，int64 引入 carry/符号位反例，argmin/max 人造重复极值；dot 的具名 case generators 分开布局、短 K、特殊架构边界等。这种“按原因组织 case 集”的方式比无解释的大表可迁移。CUDA warp/SM/CTA 取值不能替代 Metal 的实际 SIMD/threadgroup 能力。[T3]
- **TileLang：** GEMM 的 block、输入/累加 dtype 与 M/N/K 是不同参数。Metal 样本中的 `cooperative_tensor_non_square` 是 **32×64 tile**，输入仍是 128×128，不应误报为完整非方阵输入覆盖；所查 Metal GEMM 样本不证明任意尾部覆盖。[L3]
- **PyTorch：** add/sub 同时考虑 scalar、不同形状、alpha、in-place 和广播；其他测试可包含大内存、OS 版本或 dtype 条件。可借鉴语义维度，但 #31 当前明确排除 alias/in-place，不因同行覆盖它们就扩大首批范围。[P3]
- **cuTile Rust：** `slice_non_divisible` 明确选择 block 128 前后 127/129、prime 251、多块 1000、带偏移切片及 2D copy。GEMM 注释明确 K 必须整除 BK，是那个测试 kernel 的循环前提，不是本项目 K 尾部可以省略的理由。[C3]

建议保留 #31 的单元素、实际 SIMD/threadgroup 前后、多组/尾部、K=3/4/5、非方阵与较长 K、重复/逆序 gather、坐标编码 transpose、固定 seed 加结构化反例。将每个 case 写成“风险 → 输入 → 预期可观察结果 → GPU 配置”，不借同行庞大矩阵追加预算之外的 dtype/模型。首批与新增上游路径的映射需要单独决策；复杂 attention 通过不能替代独立 exp 或 add/sub/mul 的验证。

### 3.3 无设备、skip 与执行完整性

| 项目 | 所查行为 | 对本项目的含义 |
| --- | --- | --- |
| Triton | GPU 数不足时 runner 失败；特定能力 `pytest.skip`。warmup report 有已标记测试漏执行检查，但检查的是 warmup/runtime trace 关系。[T2][T4] | 借鉴“预期集合与执行集合比对”，改用独立必需 case 清单和最终数值状态；不能仅凭 cache completeness。 |
| TileLang | `requires_metal` 沿 TVM Feature 检查 runtime/target/父 GPU 条件；Metal4 另设 skip。conftest 尝试拒绝全部未执行，但 `executed_count` 将 warnings、xfailed 等也计入，且 perf-only skip 有豁免。[L2][L4] | 这是零执行保护的先例，却不是严格完备门禁。即使代码生成测试通过、数值测试全跳过，也不能满足 #31。 |
| PyTorch | 无 MPS 时 `TestCase`/`NNTestCase` 换成 `NoTest`；按 OS、feature、内存跳过很普遍；`--mps` 只选择特定 suite，不是完整 CPU suite 的复制。[P2][P3] | 适合开发者跨平台收集测试，不适合作为必需 Metal CI 的成功政策。必须先验检查设备，并检查每个必需项是否 GPU 执行。 |
| cuTile Rust | `supports_tile_ir` 发现能力失败用 `expect`，不支持某 feature 输出 SKIP 并返回 false；不吞 JIT/执行错误。脚本汇总 shell 命令成败，不核对每个 test 的必需完成状态。早 return 可由 Rust test harness 报 ok。[C2][C4] | 可复用错误分类，但本项目必需能力缺失必须未验证/非零，不能把返回值或 cargo exit 0 当作数值成功。 |

**建议的最低验收模型：** 对每个版本化 case ID，记录 planned、开始、当前源码生成、GPU 提交/完成、回读、参考比较通过或失败/未验证；最终核对预期集合与实际状态。runner 不可用、fixture 缺失、超时、中途异常、汇总失败均不得成功。自检和纯文本测试有独立类别。对 comparator 注入 NaN/Inf、错索引、漏写、shape/dtype 不符、保持区损坏，并注入缺设备/漏 case，确认失败退出。同行框架都不能自动替我们满足这些合同。

### 3.4 PR、更大测试层与资源

| 层级 | 同行事实 | 迁移建议 |
| --- | --- | --- |
| 必需 PR 层 | Triton 多种专用 GPU，24 pytest workers；TileLang CUDA/ROCm self-hosted 与 hosted Metal，8 workers、120 分钟 job；PyTorch 专用 Mac；cuTile Rust 公开 CI 为 CPU/assembler。[T1][L1][P1][C1] | 保留所有 PR 的小而完整首批，无路径过滤。先测冷构建、依赖安装、GPU 运行与汇总成本，不复制并行数。 |
| 合入后层 | Triton main 按四小时/依赖变化节流；PyTorch trunk 有 MPS 与过滤；TileLang 此 CI 文件没有 main push。[T1][P5][L1] | 同行调度不自动修改 #31 每次 main 验证的承诺。免费不代表无限吞吐，优化应先减重复构建而非默默漏跑。 |
| 扩展/定时层 | TileLang 每日 Dist 验证 wheel，CUDA/ROCm GPU smoke 不阻塞发布；PyTorch periodic 独立 workflow；cuTile GPU script 无公开定时调用证据。[L6][P5][C1] | nightly 只作为未来更多形状/dtype/长 K 的候选，不是现已决定。若扩展层独占某能力，PR 成功声明必须明确不包括它。 |

GitHub 固定版本文档声明公开仓库标准托管 runner 免费；arm64 macOS 标准表列 **3 CPU、7 GB RAM、14 GB SSD**，`macos-15` 与 `macos-latest` 均在标准列表，后者当前指向 macOS 26。此为服务文档规格，不是永久设备或 Metal feature 契约。[G1] TileLang 实际 run 加强了标准托管 Metal 的可行性证据，但没有代替 #31 指定系统的当前源码验收。PyTorch 专用 Mac 和 Triton NVIDIA 集群更不能作为免费 runner 的资源证明。

建议使用最小规模但保持全量输出比较，分别记录 build/cache-hit/cache-miss 成本、峰值内存、磁盘、单 case 时间，控制同时编译和 GPU 任务数。缓存只能缓存构建依赖或能对应源码的产物，不能缓存成功结论。是否使用 pytest、Rust 集成测试、独立二进制或混合 adapter，成本要包含 Python/Torch 安装与 Rust emitter 构建，而不能只计 kernel 运行时间。

### 3.5 失败诊断与失败保真

- Triton 分开 lit/runtime/sanitizer/regression，pytest instafail、可选 JUnit、编译 trace 有助于判断故障阶段，但专用 GPU sanitizer 不能直接移植 Metal。[T1][T2]
- TileLang `--showlocals --durations=0`、`torch.utils.collect_env`、生成源码、GEMM 最大差异与参数说明可直接借鉴；一次测量需区分 build 与 runtime。其全目录 `--maxfail=3` 意味着前面失败后未执行项应记录原因，而不是消失。[L1][L3][L5]
- PyTorch 的 Metal validation 层值得作为**可选诊断试验**，需先确认虚拟设备支持和资源消耗。`macos-test.sh` 对 Metal capture 有最多三次尝试，通用 `run_test.py` 也有 retry 机制；这不是 #31 允许自动重试计算的依据。大项目 artifact 上传方案同样不自动推翻仅日志决定。[P1][P2]
- cuTile Rust 的源码位置/JIT 能力错误、逐步骤失败累积、`print_ir`、逐元素 index/value 对小型 runner 有参考价值；不需要引入整个 CUDA 编译栈。[C2][C3][C4]

建议保留 #31 失败阶段、参数绑定、shape/dtype/layout、dispatch、实际/参考差异位置、有效容差、已生成完整 MSL、源码/依赖/toolchain/device/run attempt。日志-only 首版可继续；若提出附件、capture 或更长期保留，须说明实际诊断缺口与存储成本，不能因同行上传 artifact 就照抄。

## 4. 对旧 spec 的具体保留/重审建议

以下是供地图决策的选项，不修改 spec 或旧票，也不把已有交付撤销。

| #31 决定 | 建议 | 依据、代价与待决问题 |
| --- | --- | --- |
| 当前源码 MLIR → emitter → MSL → GPU → 回读 | **保留保证** | 同行最可信证据都能追到构建和执行。小 add 已交付；上游也有 `TargetRegistry` 接线。[B1][B2] 要证明实际编译源码对应记录提交，不能用发布 dylib/预生成 MSL 代替。 |
| 独立入口直接引用 `convert_mlir_to_msl`，不依赖整套旧测试 | **重审具体形态，不放宽边界** | 可以保留入口、复用已有 integration tests，或用薄 adapter 统一它们。低成本方案是保留既有状态/比较/日志代码并复用 fixture；中等成本是统一 case 契约；整套迁到新框架成本最高且目前证据不足。选择需等上游资产映射，不在报告里替用户决定。 |
| PyTorch CPU 为唯一参考 | **重审唯一性** | PyTorch MPS 测试支持 CPU 对照；Triton NumPy、cuTile 手算和上游 Rust CPU 参考说明框架不是独立性的必要条件。保留 torch 成本主要是安装/版本记录；混合参考需逐类语义审核、舍入一致、锚点与 comparator 共用；全量重写 Rust 数值参考会增加自证负担，不建议仅为去 Python 而做。 |
| 相同 dtype/同一输入；f32 默认容差，搬运精确 | **保留；例外逐例审查** | 默认容差确有一手依据，但不是通用误差界。不要复制 TileLang mismatch ratio、Metal 大 atol 或 Triton reduction 1%。改参考实现时必须重新核对 dtype/累加顺序，不能继承名字相同的保证。 |
| 独立 add/sub/mul/exp/组合、行和、softmax、转置 matmul、gather、transpose 首批 | **保留风险问题，重审资产映射与交付分层** | 先标出每项当前有无 MLIR、reference、GPU adapter、真实 CI 证据，再决定复用或补齐。attention、量化等上游新增路径不能替代基本语义，也不能自动扩范围。补 tail/guard case 成本通常低于迁移完整框架。 |
| 实际 SIMD/threadgroup 边界、固定 seed/结构化输入、输出/保持区全检 | **保留** | 同行展示边界前后与模式特例；常量 smoke 揭示索引检测盲区。额外成本为小输入和 guard buffer，优先级高于增加随机矩阵规模。 |
| 必需无跳过、完整清单、自检；环境不可用非成功 | **保留并作为任何复用方案的准入条件** | 上游/同行都有正常开发者 skip，与指定 Metal CI 合同不同。adapter 需将 required skip/early return 转为未验证非零，并区分纯文本、自检和 GPU 数值结果。 |
| PR/main/manual 全首批，无路径过滤、无 nightly 要求 | **保留初始承诺；扩展层待决** | 同行成本策略不自动适合小项目。先拿实际免费 runner 冷/热成本；只有规模证据出现后才讨论分层，不能以 nightly 替代原本必需 PR case。 |
| `macos-15` 免费标准 runner | **保留约束，重新实测指定环境** | TileLang 是 macOS 26 先例，PyTorch 是专用 Mac。若考虑升系统必须作为明确新决策，而非偷换标签。无设备失败、真实设备记录不可省略。 |
| 仅日志、完整失败 MSL、不自动重试计算 | **保留首版；诊断增强作为提案** | 可借 showlocals/差异位置/阶段摘要；validation 层与 capture 是否有效需试验。不得借 PyTorch retry 或 Triton continue-on-error 弱化必需结果。 |
| 接入完成与首批全量通过分开验收 | **保留** | cuTile CPU CI 与 GPU 脚本分离、TileLang 混合通过/跳过都说明这个区分必要。既有 `add_f32_small` 保证保留，但不外推 #31 全批。 |

## 5. 尚缺证据及后续决策输入

1. 本报告没有对指定 `macos-15` 执行当前 tile-rs 源码。#31 的既有探针只支持它所声明的手写 add；同行 run 也不能替代本项目全链路验收。
2. 上游新增 tests 到旧首批的逐项映射、生成路径是否与生产 emitter 相同、CPU reference 精度和 skip 分类，需要独立盘点。这里仅抽查 [B2]，不以数量做覆盖率结论。
3. 四个项目都未在所查代表路径上给出可直接复用的“当前 Metal 首批全部必需 case 无跳过”的现成机制。Triton trace 检查可借鉴，仍需本项目版本化清单和最终状态合同。
4. 精度预算、较长 K 的正常误差、hosted 虚拟设备能力和 cold-build 资源，都需要后续目标环境证据。没有发现可把 CPU参考/pytest/Rust harness 中某一个直接判为唯一正确方案的依据。
5. Actions 日志会过期。报告固定源码 SHA、run/job ID、checkout 与关键观察；不声称永久保存了外部执行档案。Triton 仅 job API、cuTile 仅 CPU job API 的证据层级不得升级。

可带入下一张人工决策票的问题是：选择哪种复用边界、哪些语义保留 CPU torch 或用已有独立 Rust 参考、必需清单如何区分文本/自检/GPU、已有 add 交付如何接续。是否修订旧票或关闭替代票仍由地图决定，不由本报告代办。

## 一手来源索引（所有源码链接固定 revision）

### 本项目

- [B1] [已交付入口 README](https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/README.md)、[workflow](https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/.github/workflows/metal-correctness.yml)、[规划合并基线](https://github.com/Jopqior/tile-rs/commit/b1772be1df05edeb475cb7ee16e644a368c2e04c)。提交关系通过本地 `git merge-base`/`--is-ancestor` 核对。
- [B2] [当前 `msl_flash_attn_ref.rs`](https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_flash_attn_ref.rs)、[`msl_ported_iquant_contract.rs`](https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_ported_iquant_contract.rs)。

### Triton

- [T1] [`ci.yml`](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/.github/workflows/ci.yml)、[`runner-preparation.yml`](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/.github/workflows/runner-preparation.yml)、[`integration-tests-nvidia.yml`](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/.github/workflows/integration-tests-nvidia.yml)。
- [T2] [`Makefile`](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/Makefile)、[`python/triton/_test_runner.py`](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/python/triton/_test_runner.py)、[`python/test/conftest.py`](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/python/test/conftest.py)。
- [T3] [`test_core.py` reduction L3038–3121](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/python/test/unit/language/test_core.py#L3038-L3121)、[dot cases/modes L3997–4360](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/python/test/unit/language/test_core.py#L3997-L4360)、[`_internal_testing.py` 输入生成](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/python/triton/_internal_testing.py#L155-L182)。
- [T4] [`_compile_warmup.py` completeness L300–369](https://github.com/triton-lang/triton/blob/ee07f7ff302740dbb858b84c7c1ea9f1d1ffc782/python/triton/_compile_warmup.py#L300-L369)。

### TileLang

- [L1] [`.github/workflows/ci.yml`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/.github/workflows/ci.yml)。
- [L2] [`testing/conftest.py`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/testing/conftest.py)。
- [L3] [`testing/python/metal/test_metal_gemm_v2.py`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/testing/python/metal/test_metal_gemm_v2.py)；[实际 run checkout 版本同路径](https://github.com/tile-ai/tilelang/blob/0baf77d4e5dbb53b1b10d6d11b46fb822ebed032/testing/python/metal/test_metal_gemm_v2.py)。
- [L4] [`tilelang/testing/__init__.py`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/tilelang/testing/__init__.py)、[`.gitmodules`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/.gitmodules)、[TVM `python/tvm/testing/utils.py` Feature / requires_metal](https://github.com/TileLang/tvm/blob/907a88c8791ccf33b9874821bc875e7abf624367/python/tvm/testing/utils.py)。
- [L5] [`testing/python/kernel/test_tilelang_kernel_gemm.py`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/testing/python/kernel/test_tilelang_kernel_gemm.py)、[`tilelang/profiler/__init__.py` L104–162](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/tilelang/profiler/__init__.py#L104-L162)、[`tilelang/utils/tensor.py` 比较器](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/tilelang/utils/tensor.py#L205-L289)。
- [L6] [`.github/workflows/dist.yml`](https://github.com/tile-ai/tilelang/blob/6679f91387a2a94d858f0789478fb3461ab5a695/.github/workflows/dist.yml)。

### PyTorch

- [P1] [`mac-mps.yml`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/.github/workflows/mac-mps.yml)、[`_mac-test.yml`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/.github/workflows/_mac-test.yml)。
- [P2] [`.ci/pytorch/macos-test.sh`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/.ci/pytorch/macos-test.sh)、[`test/run_test.py`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/test/run_test.py)。
- [P3] [`test/test_mps.py` 无设备处理](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/test/test_mps.py#L82-L89)、[`test_add_sub`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/test/test_mps.py#L9862-L9910)；[实际 run 版本](https://github.com/pytorch/pytorch/blob/eaff5f67eab85ae80ee65d972bfff6f1e066e5c4/test/test_mps.py)。
- [P4] [`torch/testing/_comparison.py` 默认容差、属性与非有限值政策](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/torch/testing/_comparison.py)。
- [P5] [`trunk.yml`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/.github/workflows/trunk.yml)、[`periodic.yml`](https://github.com/pytorch/pytorch/blob/a402343453f89b31a2ec8372b304accdc5b78e5f/.github/workflows/periodic.yml)。

### cuTile Rust

- [C0] [`README.md` 项目身份、JIT 链及硬件/toolkit 要求](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/README.md)。CUDA 13.3 推荐、Tile IR/GPU capability 分开要求，不支持低于 sm_80 的设备；这些并非 Metal 要求。
- [C1] [`.github/workflows/pr.yml`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/.github/workflows/pr.yml)、[完整 workflow 目录](https://github.com/NVlabs/cutile-rs/tree/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/.github/workflows)、[`scripts/run_cpu_tests.sh`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/scripts/run_cpu_tests.sh)。
- [C2] [`scripts/run_gpu_tests.sh`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/scripts/run_gpu_tests.sh)、[`scripts/test_runner_common.sh`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/scripts/test_runner_common.sh)。
- [C3] [`cutile/tests/gpu.rs`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/cutile/tests/gpu.rs)、[`cutile/tests/gpu/add_basic.rs`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/cutile/tests/gpu/add_basic.rs)、[`cutile/tests/slice_non_divisible.rs`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/cutile/tests/slice_non_divisible.rs)。
- [C4] [`cutile/tests/common/mod.rs`](https://github.com/NVlabs/cutile-rs/blob/3ac6fb9669dd4fb0ebc7d01904033d3d6e417a3b/cutile/tests/common/mod.rs)。

### GitHub runner 服务文档

- [G1] `github/docs@73f38c5edb3f47a01eb3db96fa723775e2bfeaf9`：[`content/actions/reference/runners/github-hosted-runners.md`](https://github.com/github/docs/blob/73f38c5edb3f47a01eb3db96fa723775e2bfeaf9/content/actions/reference/runners/github-hosted-runners.md)、[`data/reusables/actions/supported-github-runners.md`](https://github.com/github/docs/blob/73f38c5edb3f47a01eb3db96fa723775e2bfeaf9/data/reusables/actions/supported-github-runners.md)、[`macos-runner-limitations.md`](https://github.com/github/docs/blob/73f38c5edb3f47a01eb3db96fa723775e2bfeaf9/data/reusables/actions/macos-runner-limitations.md)。服务规格与实际 job 观察分开表述。
