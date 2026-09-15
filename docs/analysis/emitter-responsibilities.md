# 多目标 emitter 职责与重复：源码调查

对应决策票：[核实多后端 emitter 的重复、职责与硬件差异](https://github.com/Jopqior/tile-rs/issues/14)。本文保存调查证据，确认后的判断以该票 resolution comment 为准，不选择分层方案，不制定实施计划。

## 证据边界

- 源码版本：`Jopqior/tile-rs@3bf59222a72c4e293cc19ebbdb72754549d60dea`。与领域地图参照 `1ee797f1f5834ac9ce581ce69e8b8f1775ece076` 相比，`git diff --stat <reference> HEAD -- crates/` 为空。
- 本文表格中省略目录的 `mlir_*.rs` 均位于 `crates/rustc_codegen_tile/src/`。行号相对此固定版本，不随文档后续提交更新。
- 固定版本源码链接格式：`https://github.com/Jopqior/tile-rs/blob/3bf59222a72c4e293cc19ebbdb72754549d60dea/<path>#L<line>`。
- 只读源码与现有测试定义，未运行生成器、测试、目标编译工具链或硬件。下面的输入变化案例是控制流推导，不是已执行实验。
- 不推断历史复制来源，不以文件相似证明复制历史，也不以源码规模估算新增目标所需工时或可删除行数。
- 发布动态库与公开源码的版本对应、Metal 构建产物接线和运行验证缺口沿用[领域与链路调查](../domain/tile-rs-source-map.md)，本票未补齐。
- 不给维护痛点预设优先级。分别检查跨目标同步修改、新增目标所需独立知识、代码体量、内部理解与验证联动。

## 1. 对照选择

不是按 GPU/NPU/TPU 分类选样，而是选能区分共享方式和实现职责的路径：

| 对照 | 区分作用 |
|---|---|
| Metal/MSL | 最大文件；整类 kernel 模板与操作链分析并存；深入核对识别、入口及输出的联动 |
| CUDA/gpu 与 MUSA | 逐操作转换与整生成器委托的对照，反证所有目标都独立写一份 |
| SPIR-V 名称下的 GLSL 生成器 | 与 MSL 各有本地分类，但覆盖范围不同；不能把名称或同步注释当共享设施 |
| NKI、TPU/Pallas | 独立逐操作处理，观察相似输入分析与不同目标原语 |
| PTO | parser 助手再导出、SiLU→Mul 检测、显式 tile 操作；不把导入关系当 Metal 先经 PTO |
| linalg | 保留高层 named operation 的输出，反证所有目标都必须展开同一算法 |

其余文件只做规模和接线概览。没有逐一审计 AIE、BANG、Gaudi 的所有操作，也不据此断言它们与上述样本语义等价。

## 2. 体量，不是可删规模

机械统计总行数，并以首个单独的 `#[cfg(test)]` 为分界。分界前仍包含空行、注释及代码内目标文本，不是精确的生产逻辑行数；分界后也不是独立测试场景数。

| 文件 | 总行 | 首个测试标记前 | 测试标记及其后 |
|---|---:|---:|---:|
| mlir_to_msl.rs | 17127 | 13541 | 3586 |
| mlir_to_pto.rs | 7910 | 6137 | 1773 |
| mlir_to_gaudi.rs | 4377 | 3229 | 1148 |
| mlir_to_bang.rs | 3746 | 2410 | 1336 |
| mlir_to_linalg.rs | 3258 | 1907 | 1351 |
| mlir_to_aie.rs | 3187 | 2172 | 1015 |
| mlir_to_gpu.rs | 3084 | 2097 | 987 |
| mlir_to_nki.rs | 3014 | 2009 | 1005 |
| mlir_to_spirv.rs | 2846 | 1576 | 1270 |
| mlir_to_tpu.rs | 2789 | 1646 | 1143 |
| mlir_to_hexagon.rs | 397 | 306 | 91 |
| mlir_to_ttmetal.rs | 360 | 269 | 91 |
| mlir_to_csl.rs | 322 | 227 | 95 |
| mlir_to_musa.rs | 140 | 41 | 99 |

MSL 的约略职责区段：

- `57–347`：本地 `KernelType`。
- `519–2824`：`generate_func_msl`，包含分析调度、未处理操作检查、入口参数、线程属性与正文分发。
- `2840–3679`：操作链分析及相应代码生成。
- `3729–4143`：`classify_body`。
- `4145–约13480`：大量专用生成函数，包含归约、矩阵乘、量化及特化 kernel。

这些区段显示专用实现的数量与调度联动都占据体量。不能把所有专用正文当跨目标重复，也不能把所有现存正文当硬件唯一允许的实现。

## 3. 已有共享的实际范围

### 文本解析与助手

`mlir_to_pto.rs:103–105` 再导出 `mlir_parse` 的 `parse_module`、调用参数及 SSA 提取等设施。MSL、gpu、spirv、nki、tpu、linalg、aie、bang、gaudi 经该位置导入部分助手。共享的是 MLIR 文本解析结果及文本助手，不是 PTO lowering。

MSL 实际调用 `parse_module`（`432`）、`is_builtin_helper`（`503`），并在链分析和分类中使用调用参数及结果 SSA 提取。函数体仍主要以文本行供各目标分析。

### 整生成器委托

[mlir_to_musa.rs:13–40](https://github.com/Jopqior/tile-rs/blob/3bf59222a72c4e293cc19ebbdb72754549d60dea/crates/rustc_codegen_tile/src/mlir_to_musa.rs#L13) 调用 CUDA 生成器，再替换头文件、平台名称等字符串。它没有独立的 SiLU→Mul 检测。CUDA 的改动和能力缺口也会被这条路径继承。

这证明本树已有整实现共享，不证明字符串改写足以支持任意新目标，亦不证明所有 MUSA 输出都经工具链验证。

### 公共调用 interface

`crates/tile_codegen/src/target.rs:59–69` 定义 `CodegenTarget`；`targets.rs:87–114` 的 `EmitterTarget` 包装普通转换函数。它共享调用和输出包装，不共享各目标内部的操作分析。

`targets.rs:59–80` 登记十三个实际生成器，不含 PTO；`crates/tile_spec/tests/cucumber.rs:58–75` 直接调用十四个，含 PTO。登记、源码存在与执行能力不能混为一谈。

## 4. 重复分析：SiLU→Mul

相同的逻辑输入：

```text
s = silu(gate)
out = s * up
```

| 生成器 | 位置 | 实际检测方式 |
|---|---|---|
| CUDA | `mlir_to_gpu.rs:542–569` | 先收集 SiLU 结果，再扫描所有乘法调用的前三个参数是否引用结果，不要求相邻 |
| PTO | `mlir_to_pto.rs:3464–3497` | 从 SiLU 向后找第一条调用，只在该调用是匹配的乘法时记录融合对 |
| TPU/Pallas | `mlir_to_tpu.rs:617–653` | 在乘法分支使用 `last_silu`，融合分支要求至少五个参数；匹配后改写已积累输出 |
| NKI | `mlir_to_nki.rs:562–633` | 在乘法分支使用本地 SiLU 状态，融合分支要求至少五个参数，并重写已积累输出 |

若在 SiLU 与乘法之间插入无关调用，CUDA 的上述检测仍可能找到引用关系，PTO 的上述检测会停止。这是独立规则差异的证据，不是执行结果或正确性判决。

需区分三件事：识别使用关系、判断合并处理是否合法或值得、输出目标实现。识别重复值得研究，但共享识别不要求所有目标选择同一融合策略。中间值另有使用、参数约定、目标存储约束等条件仍需核对。不能把某份现有检测直接指定为正确的公共实现。

维护含义：修改输入约定或组合识别范围时，有多个独立位置需要核对。没有证据表明所有规则都必须同时改成一致，也没有测量哪一项成本最高。

## 5. Metal 的内部联动与未接入设施

### 组合分析与分类并存

[mlir_to_msl.rs:519–619](https://github.com/Jopqior/tile-rs/blob/3bf59222a72c4e293cc19ebbdb72754549d60dea/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L519) 先执行本地分类，再尝试 matvec，未匹配才尝试逐元素链；随后 gemm 或 partition 分析可以覆盖当前类别。最后部分检查会拒绝未被处理的归约或分区计算。

`analyze_matvec_chain`（`3074–3173`）跟踪 load、乘法、求和、store 的结果关系，检查保存的关系是否满足 `store(reduce(mul(load,load)))`。这是一种受限模式识别，不是完整 MLIR 数据流验证器。

`emit_matvec_chain_msl`（`3616–3669`）据此生成局部乘加、`simd_sum`、threadgroup 部分结果及最终存储。逐元素输出则不具备该归约行为。

扩大某个识别器的接受范围会影响后续识别器是否能接手。此成本来自行为覆盖和调度关系，不能以移走文本模板证明已消除，也不足以单独推出必须增加 IR。

### 独立分类与同步注释

MSL `54–55` 的注释声称复用 SPIR-V 分类，但 MSL `57–347` 和 SPIR-V `65–113` 各自定义 `KernelType`，并各有 `classify_body`。不能把注释当实际共享，也不能假定两份分类覆盖相同能力。

### 有定义不等于转换入口可达

- MSL 导入的 `DEFAULT_K_UNROLL` 未在该文件使用。
- `emit_unrolled_k_accumulation` 在 MSL 的引用出现在测试，生产 `emit_matmul_transposed_msl`（`13427–13442`）仍直接生成四路展开。
- `emit_mul_mm_q4_0_f32_setup_msl`（`9391`）有直接调用它的测试（`16472`），但未见 `convert_mlir_to_msl` 的分类分发接线。

这里仅记录可见调用关系，不断言其历史原因、未来用途或应删除。直接测试生成函数与通过公共转换入口测试不是同一条路径。

## 6. Softmax：共同计算与不同实现职责

| 生成器 | 证据 | 输出承担的职责 |
|---|---|---|
| Metal | `mlir_to_msl.rs:4145–4179` | 线程局部 max/sum、threadgroup 暂存、循环归约、显式屏障及归一化 |
| CUDA | `mlir_to_gpu.rs:1242–1326` | warp/block 归约、共享暂存及同步 |
| NKI | `mlir_to_nki.rs:681–736` | 平台 tensor reduce、减、exp、sum、divide 操作 |
| TPU/Pallas | `mlir_to_tpu.rs:677–733` | JAX 的 max、exp、sum、除法，显式保留 axis 等参数 |
| PTO | `mlir_to_pto.rs:2500–2576` | 显式 rowmax、广播减、exp、rowsum、广播除 tile 操作 |
| linalg | `mlir_to_linalg.rs:1308–1342` | 输出 `linalg.softmax dimension(1)`，本生成器不展开线程与同步 |

这些实现有共同数学意图，但归约轴、工作分配、dtype、尾部及同步条件仍需分别验证。表格不是跨目标数值等价证明。

差异至少包含两类：目标语言/执行模型约束，以及当前选择在哪一步展开、用哪种算法实现。后者不是硬件必然。把 named operation 强行展开成公共逐步实现，可能提前承担原本由后续工具完成的职责。

## 7. 能力与验证缺口：matmul

[CUDA 的普通 matmul 分支](https://github.com/Jopqior/tile-rs/blob/3bf59222a72c4e293cc19ebbdb72754549d60dea/crates/rustc_codegen_tile/src/mlir_to_gpu.rs#L1372)（`1372–1396`）读取参数、登记结果，但返回的正文是 `TODO: matmul ... use cublasSgemm` 注释。

NKI `742–766` 对应分支实际输出 `nisa.nc_matmul`；linalg 对应路径输出 tensor 初始化及 `linalg.matmul`；Metal 还存在专用 simdgroup 矩阵乘及 partition-cell 链分析路径。它们不构成一个已经统一验证的能力全集。

`crates/tile_spec/features/core_op_lowering.feature:22–26` 的 CUDA matmul 示例只要求 `__global__`。`tests/cucumber.rs:272–314` 实现的是转换返回成功和字符串包含检查，没有调用目标工具链或比较矩阵乘结果。因此这项断言不能识别普通 matmul 正文缺失。本文未运行该场景，不宣称整套测试通过或失败，也不把这一项检查代表为全项目的全部验证。

需要分别记录：识别操作、输出计算、目标编译、数值正确性及性能。共享参数读取不会自动补齐计算实现或验证强度。

## 8. 对后续研究的输入

值得研究的共性：输入操作及参数约定、数据依赖与组合关系的理解，以及这些事实能否与目标实现选择分别变化。已有 parser 和 MUSA 委托应作为现状起点，不能忽略后重新论证一遍共享。

必须保留或明确表达的差异：目标原语、存储与同步、入口参数及工作分配契约、哪些算法由本生成器展开、哪些留给后续工具。现存代码选择不是全部硬件必需，也不预设可以统一。

待核实：独立识别规则差异的依据；特殊输入及多次使用下的合法性；数值和形状语义；未接入设施的来源；发布与硬件运行缺口。若架构判断取决于其中某一点，需要针对性补查或原型，不能用现有文本测试补成证明。

不从本调查直接推出层数、IR 形式、按硬件家族组织方式、可删行数或新增目标工时。候选方案还需解释减少了哪些同步修改、保留多少特例，以及为此增加多少概念和依赖。
