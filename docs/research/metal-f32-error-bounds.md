# 首批 f32 Metal 验证用例的数值误差界：条件性研究记录

对应 [推导首批 f32 Metal 验证用例的数值误差界](https://github.com/Jopqior/tile-rs/issues/11)，承接
[确定 Metal 正确性判据与首批覆盖规则](https://github.com/Jopqior/tile-rs/issues/6#issuecomment-5663077264)。
本研究阻塞后续 [验证产物契约](https://github.com/Jopqior/tile-rs/issues/7) 和
[CI 分层门禁与设计验收](https://github.com/Jopqior/tile-rs/issues/8)，不是反向依赖。

**状态：部分完成，不能直接用本文公式判定 CI 通过。** 初稿及第一轮修订均有过强推论；本记录保留可解释的数学框架，撤回未证实的部署常数。
还须补齐参考实现、数学模式、适用规范与预算计算自身的误差依据。研究票保持打开，不解除后续阻塞。

本次未运行 Metal、PyTorch 或修改 emitter/CI。源码调查、规范声明、条件推导与手算分别注明；
不能把另一张票的最小执行探针当成本研究的数值模型实测。

## 1. 已确认边界，不重新选择

- kernel 在当前源码的 MLIR→MSL→GPU 回读链路执行；参考是 **PyTorch CPU**。
- 同一批实际 f32 输入先定稿；算术参考升 f64 求值，gather/transpose 保留 f32 做精确搬运比较。
- 范围：add/sub/mul、exp、`(a+b)*c`、行求和、softmax、f32 转置矩阵乘、gather、transpose。
- 有限正常数与零；用例避开预期中间量的溢出/下溢。特殊值算术、空维度、非法索引、别名及非连续布局不在首批。
- 关闭 fast-math，允许 FMA，不因此允许任意重结合；全量比较，合法用例出错保留失败。
- 按精度约束与规模推导误差，不照抄库默认容差，不根据当前错误输出调宽。

## 2. 审阅中撤回或修正的结论

1. `simd_sum` 的“跨 active threads 求和”不保证平衡树；不能把 32 项固定为深度 5。
   连“32 lanes”本身也须核对实际执行配置，不能从 Apple GPU 常见配置推广到虚拟设备。
   初稿的 `+10` 和修订中的无条件 `+62` 都不是可直接部署的通用常数。
2. `add(1, 2^-24)` 返回 `1+2^-23` 时误差也是 `2^-24`，不是 `1.5·2^-24`；它不能作为此误差界的界外反例。
3. f32 和/差不总在 f64 中精确；不能用指数差的简单条件给出必要且充分判定。
   f32→f64 加宽无损、两个有限 f32 之积在 f64 中精确，均不意味着后续求和精确。
4. 从真值 `M` 换成参考 `R` 必须计入参考误差；不能直接把 `|M|` 替换成 `|R|` 仍称严格上界。
5. arm64 参考不能引用 x86 AVX 的 Sleef 路径作为实际执行保证。具体 PyTorch 构建、dispatch、scalar tail、libm/BLAS 路径仍须核实。
6. 补偿求和不自动提供有证明的区间；给阈值随意乘 `1+2^-40` 也不是已推导的舍入补偿。本记录撤回这种部署建议。
7. 不比较 f32 和 f64 的编码位串；搬运比较应在相同 dtype 上进行。
8. 单个 add 探针不能证明所有运算和配置的舍入模式；一次符合预期的样本也不建立整个输入域的误差保证。
9. 次正规算术已经明确不在首批，不应重新作为默认待决政策。softmax 非幂次线程数的问题不能被自动改写为合法范围缩减。

## 3. 条件数学模型

以下是**实数表达的候选预算**，不是未经舍入分析即可照抄的 f64 实现。
记 `M` 为用例在定稿输入上的精确计算值，`g` 为 GPU f32 输出的数值，`R` 为 CPU f64 参考。

基本关系：

```text
|g - R| <= |g - M| + |M - R| <= E_gpu + E_ref
```

若相对界为 `|g-M| <= b_g |M|`、`|R-M| <= b_r |M|`，且 `b_r < 1`，则

```text
|g - R| <= (b_g + b_r)/(1 - b_r) * |R|
```

输入和运算不发生溢出、下溢/FTZ，并且每次基本算术符合相应舍入模型时：
`fl(z)=z(1+δ)`，`|δ|<=η`。binary32 RNE 可用 `η=2^-24`；RTZ 的保守模型可用 `η=2^-23`。
本文保留两条条件分支，**不替用户选择尚未确认舍入前提时的接受政策**。
CPU binary64 的候选模型取 `v=2^-53`，也须核对参考执行前提。

记 `γ_n(e)=ne/(1-ne)`，要求 `ne<1`。标准乘积误差引理给出含至多 n 个舍入因子的误差上界。
求和树未知时，只能从全部项/运算数建立保守路径界，不能默认对数深度。
该模型要求普通加法归约/乘加，不自动涵盖任意代数变换或未知库算法。

### 3.1 四则与两步组合

| 计算 | GPU 对真值的候选界 | CPU 参考候选界 |
| --- | --- | --- |
| add/sub | `η |M|` | `v |M|`，以一次正确舍入 binary64 运算为前提 |
| mul | `η |M|` | 0：两有限 f32 数的精确乘积可表示于 f64 |
| `(a+b)*c` | `γ_2(η) |M|` | `γ_2(v) |M|`，参考确实先加后乘 |

部署尺度可按上面的 `(b_g+b_r)/(1-b_r)` 转换。零结果依相同模型处理；带符号零的算术规则不在本次承诺内。
`(a+b)*c` 不是一条 FMA 的形式。该界依赖无重结合前提，但不能保证检测所有非法重结合：不同表达式可能仍落在同一误差界内。

### 3.2 行求和与转置矩阵乘

令 `S=Σ|x_i|`、`P=Σ_k|A[m,k] B[n,k]|`，此处均为精确实数和。

- **行求和候选**：若合法调度使每项恰参与一次普通加法归约，忽略精确的加零操作，W 项任意二叉结合树至多 W−1 次加法。
  可保守取 `E_gpu=γ_W(η) S`；CPU 普通 f64 加法归约同样可取 `E_ref=γ_W(v) S`。
  这不需要假设 SIMD 平衡树，但仍须核对指令的精度模型和合法调度，不能只靠“sum”名称证明全部前提。
- **矩阵乘候选**：语义为 `C[m,n]=Σ_k A[m,k]B[n,k]`。当前 emitter 是从零开始的串行 dot，四项展开后带标量尾。
  非融合时每个乘积一次舍入，其到结果的加法路径至多 K−1 次（首次加零精确）；融合或混合路径不增加这个上界。
  因而普通 dot 模型下 `E_gpu=γ_K(η) P`。`γ_K(v) P` 是 CPU 普通 f64 dot 的候选界，不能未经核实把任意 BLAS 算法都当成这种模型。

抵消用 S/P 尺度处理，不能仅以接近零的 `|R|` 作相对预算。较保守预算也可能容忍漏掉微小项，必须结合手算精确锚点和等量级/坐标编码输入检测结构错误。

### 3.3 exp

MSL 4.1 Table 8.1 为关闭 fast math 的 exp 声明至多 4 ulp；在正常结果域，候选相对常数可取
`q_g=4*2^-23=2^-21`。**须核对这一规范表确实适用于所选工具链、函数模式和设备**。

CPU 参考的对应常数记作 `q_r`，目前不填数值。若取得该实际参考路径在用例输入域上的保证，则

```text
E_gpu <= q_g |M|; E_ref <= q_r |M|
|g-R| <= (q_g+q_r)/(1-q_r) * |R|,  q_r < 1
```

其他架构的 Sleef 1 ulp 文档或某份 Apple 开源 libm 注释，不能独立证明 runner 中已安装二进制的 `q_r`。

### 3.4 softmax

定义为逐行稳定 softmax：`m=max x_j`，`t_j=x_j-m`，`M_i=exp(t_i)/Σ exp(t_j)`。
需要极差上界 `D >= max x-min x`，而不是可能向下舍入的未补偿 f64 极差。

在 GPU 正常数模型下，令

```text
H_minus = (1-q_g) exp(-η D)
H_plus  = (1+q_g) exp( η D)
beta_g  = γ_dg(η)
L_g = H_minus*(1-η) / (H_plus *(1+beta_g))
U_g = H_plus *(1+η) / (H_minus*(1-beta_g))
b_g = max(1-L_g, U_g-1)
```

这组合了参数减法舍入、exp 函数精度、分母求和及最终除法。要求 `q_g<1`、`beta_g<1`、全程正常数/零且分母正。
当前折半共享内存归约在 `T=2^m` 且分配/调度合法时，可用 `d_g=ceil(W/T)+log2(T)` 的保守路径计数。
**这只是该公式的前提，不是授权把非幂次 T 从合法覆盖中删除**；若其合法，丢项是失败。

CPU 参考可用相同结构推导：以 `v`、已证明的 `q_r` 和 CPU 归约路径界替换对应参数。
若实际 CPU softmax 是“取倒数后乘”，最终因子分别为 `(1-v)^2` 和 `(1+v)^2`，不能把二阶项丢掉后仍称严格上界。
由其得到 `b_r<1` 后，比较尺度为 `(b_g+b_r)/(1-b_r)*|R_i|`。
目前实际参考路径的 `q_r` 及预算计算误差尚未补齐，因此这不是可部署的 softmax 阈值。

### 3.5 gather / transpose

参考在 f32 上按声明索引/布局求值，逐元素比较 `bit32(g_f32)==bit32(ref_f32)`。
要求 gather 索引非负、在源行范围内、能被 emitter 的 f32 参数编码精确表示，且能正确转为 uint。
这不是声称所有超过 `2^24` 的整数都不可表示：须检查具体整数的精确可表示性。

## 4. 预算计算本身：仍需完成的证明

即使上节对实数成立，f64 比较器也可能把预算向下舍入，或把误差向上舍入。

- 若算得 `S_hat` 且已证明 `|S_hat-S|<=b S`、`b<1`，则实数上 `S<=S_hat/(1-b)`；P 同理。
  除法等后续操作本身仍须有向外的误差控制，不能仅称“在 f64 算所以精确”。
- `D`、γ、exp、差值、分母和最终阈值必须有可审计的求值误差控制；要检查有限性、正分母和模型适用域。
- 补偿求和、定向舍入/区间、显式分析后的余量都是候选实现手段，但本文未选择或完成其中一条。
- 一个固定的“很小”加成不是证明。公式在边界上的接受规则应与预算实现一起确定，不能用当前 GPU 输出反向标定。

在这些条件未补齐前，**不存在本文认可的通用比较器伪代码**。特别不能写成只检查 g 有限，再用可能为 NaN 的 τ 做比较。

## 5. 手算锚点及检测力边界

以下只使用精确可复核的值，不把本地数学复算称为 GPU 实测。

| 情形 | 精确结果/差值 | 可得结论 |
| --- | --- | --- |
| `add(1,2^-24)` | `M=1+2^-24`；RNE 结果为 1 | 候选 `η|M|` 界内 |
| 同输入，错误返回 `1+2^-23` | 误差也是 `2^-24` | 同样落在上述相对界内；需要精确锚点才能拒绝，不可伪称界外 |
| 同输入，返回 `1+2^-22` | 误差 `3*2^-24` | 超过 RNE/RTZ 两个单次相对候选界 |
| `add(1,1.5*2^-24)` | RNE 为 `1+2^-23`，RTZ 为 1 | 可区分该样本/运算的行为，不推广到全部运算 |
| `mul(1+2^-23,1-2^-23)` | `M=1-2^-46` | 验证乘法与参考输入量化，不能误用加法 |
| 求和 `[1,1,1,1]` | 4 | 手算精确锚点，可检测明显丢项 |
| softmax `[a,a]` | `[1/2,1/2]` | 数学锚点；所允许 exp 近似及合法调度仍按声明核对 |
| `A=[[1,2],[3,4]], B=[[5,6],[7,8]]` | `A B^T=[[17,23],[39,53]]` | 区别于 `A B=[[19,22],[43,50]]`，揭露布局/语义错误 |

## 6. 新环境事实与参考证据边界

[最小 Metal 编译与执行链路实测](https://github.com/Jopqior/tile-rs/issues/9#issuecomment-5663239242) 已完成：
macOS 15.7.9 arm64、Xcode 16.4、metalfe-32023.620、Apple Paravirtual device；16 元素自定义加法执行回读通过，离线编译也成功。
它不证明当前 emitter 全链路，也未验证本研究的舍入、exp/softmax 或 reference library 前提。

PyTorch v2.8.0 的源码线索：
- 通用 `Vectorized<T>::exp()` 可走 `map(std::exp)`；AVX2/AVX512 的 double 特化使用 Sleef；macOS 排除相关 MKL VML 特化。
- 这些条件说明 **不能把 x86 的误差证据直接套给 arm64**，不等于已识别所安装 wheel 的全部运行路径。
- softmax、scalar tail、CPU ISA dispatch 与 BLAS 选择须核对实际构建。Accelerate 是 Apple 构建的候选依赖，不在未安装核对前宣称它必然是本用例实际路径。
- Apple 的一份 ARM `exp_freeBSD.c` 声称误差小于 1 ulp；文件存在不证明 macOS 15.7.9 arm64 二进制采用该代码，也不证明其他 exp 入口。
- f64 参考是否满足已分析的累加模型必须有来源；线程数或版本固定只改善溯源，不自行构成数学证明。

## 7. 保持打开的具体问题

| 缺口 | 需要的后续结果 | 当前状态 |
| --- | --- | --- |
| 规范与配置适用性 | 对应实际工具链的精度规范、明确关闭 fast-math/函数 precise，记录允许的 contraction | 候选来源已找到，适用性与实际接线未确认 |
| 舍入模型 | 以规范/适用性说明建立 η；实验只能补证。若仍无法区分，是否接受覆盖两模式的保守预算需另作明确决定 | 参数化，不代选宽分支 |
| CPU reference 保证 | 选定实际 PyTorch 构建，核实逐入口/尾部的 libm/BLAS 路径及输入域误差依据 | exp/softmax 的 q_r 未确定；matmul 模型亦须核对 |
| 合法调度 | SIMD 宽度、T、共享内存容量及每入口参数前提；独立区分语义限制和现实现缺陷 | 不因 softmax 丢项而静默缩范围 |
| 预算计算误差 | S/P/D 和 τ 的可验证数值求值方案及边界比较行为 | 未完成，不用任意常数代填 |

这些缺口由当前研究票保留，不在本 session 再代替用户解决另一张 HITL 决策票。
若来源补不齐并需更改参考方案/覆盖政策，必须先提出明确决策，不能发布一个看似完整但靠未声明假设判绿的契约。
已决定的有限正常数范围、CPU 参考位置及 FMA 政策不重开。

## Sources

- MSL Specification 4.1（2026-06-04）：[Apple PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)，
  §1.6.3 数学选项、§6.10 SIMD-group reduction、§8.1–8.5 浮点规则及精度表。URL 可更新；本调查版本不自动匹配 Xcode 16.4。
- emitter：[`mlir_to_msl.rs` 固定 c3c8ec0](https://github.com/Jopqior/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/rustc_codegen_tile/src/mlir_to_msl.rs)，
  `emit_softmax_msl`（4145）、`emit_binop_msl`（4192）、`emit_sum_rows_msl`（4632）、`emit_matmul_transposed_msl`（13427）、gather（4513）、transpose（4360）。
- PyTorch v2.8.0 固定 ba56102：[`vec_base.h`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/cpu/vec/vec_base.h)、
  [`vec256_double.h`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/cpu/vec/vec256/vec256_double.h)、
  [`vml.h`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/cpu/vml.h)、
  [`SoftMaxKernel.cpp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cpu/SoftMaxKernel.cpp)、
  [`SumKernel.cpp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cpu/SumKernel.cpp)、
  [`CPUBlas.cpp`](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/CPUBlas.cpp)。均为候选实现线索而非运行溯源。
- Apple Libm：[`exp_freeBSD.c` 固定 17a5f9d](https://github.com/apple-oss-distributions/Libm/blob/17a5f9daa3f5679f7536b26f133b40cc078753c3/Source/ARM/exp_freeBSD.c)，其误差声明不能无溯源套用到当前系统。
- [Rust Reference：f32→f64 无损加宽](https://doc.rust-lang.org/reference/expressions/operator-expr.html)、
  [C++ 浮点转换规则](https://eel.is/c++draft/conv.double)。
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., ch.2–3：[SIAM DOI](https://doi.org/10.1137/1.9780898718027)，标准舍入模型及 γ 引理；本文给出适用前提，未把不明库算法当成该模型的自动实例。
- [CPU 参考适用性研究](https://github.com/Jopqior/tile-rs/blob/785c34ed31171038b9b227eedb4249c82ea7e25a/docs/research/cpu-reference-gpu-validation.md)。
