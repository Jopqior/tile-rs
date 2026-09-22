# 上游测试与 Metal 正确性 CI 的复用边界

研究票 [#43](https://github.com/Jopqior/tile-rs/issues/43)，所属地图 [#42](https://github.com/Jopqior/tile-rs/issues/42)。本文只提供事实和待决策输入，不实施 CI，不决定保留或删除独立验证入口，不替代 [spec #31](https://github.com/Jopqior/tile-rs/issues/31)。

## 结论

1. **公开构建障碍已有实质变化。** 新基线的 `tile_codegen --features emitters`、`tile_spec` 和 `kernel_correctness` 在本轮 Linux 本地检查中均成功。Metal emitter 不再经其他后端取得解析函数；公开 registry 已能调用当前 `convert_mlir_to_msl`。旧研究中“这几个入口目前不能构建/测试”的结论不能照搬。
2. **不能再把上游所有测试概括为文本测试。** 新增 `tile_codegen/tests` 的 GPU 分支有 MLIR → registry → 当前 emitter → Metal 编译、调度、回读比较的真实代码路径。它们主要覆盖 attention、量化、模型专用 kernel，不等于 #31 的基础 f32 首批已覆盖。此处“有执行路径”是源码事实，**本轮没有执行 GPU，也没有编译 macOS 专属 Rust 分支**。
3. **也不能把直接生成器测试算成完整 emitter 验证。** `emit_tail_tests` 直接调用 `emit_*`，但只检查字符串和重复生成一致性，没有 GPU。`msl_indexer_score_one_direct` 中的 direct 是 kernel 名的一部分，实际测试仍输入 MLIR。不能按文件名或“emitted”注释分类。
4. **没有一套新增上游测试可以原样充当 #31 的完整成功门禁。** macOS integration tests 缺设备时打印 skipping 后正常返回；默认 `cargo test` 未启用 `emitters` 时不会纳入这些测试；Linux 编译裁掉 GPU 分支。比较阈值、数学模式、保持区域、状态与漏跑检查亦不符合原 spec 的统一要求。它们是有价值的适配对象，不是不可用资产。
5. **小 add 入口确已交付并有历史执行证据。** `4e66bc8` 的 push run 日志记录当前提交、PyTorch CPU、fast-math=false、Apple Paravirtual device、`add_f32_small` 和 `RUN_RESULT=pass`。但 `4e66bc8` 不是 `e8d1acd` 的祖先；新上游树缺少此入口不意味着从未交付或在该历史线上被删除。

## 1. 固定基线、来源与历史关系

| 对象 | 固定版本 | 用途 |
| --- | --- | --- |
| 原研究源码 / 共同祖先 | [`c3c8ec0169bd02757b51370ba9c6ec3b116b833d`](https://github.com/Jopqior/tile-rs/commit/c3c8ec0169bd02757b51370ba9c6ec3b116b833d) | 区分上游新增资产，不拿交付树作线性历史 |
| 本轮源码 | [`e8d1acdf58f4e2239bf6be9d59333d7a337282b3`](https://github.com/Jopqior/tile-rs/commit/e8d1acdf58f4e2239bf6be9d59333d7a337282b3) | 下文 U 链接全部固定此版本 |
| 已交付入口 | [`4e66bc824872a95bfae51f6503bfbdb360199fe3`](https://github.com/Jopqior/tile-rs/commit/4e66bc824872a95bfae51f6503bfbdb360199fe3) | 下文 D 链接；小 add，不是首批全量 |
| 原规划分支 | [`b1772be1df05edeb475cb7ee16e644a368c2e04c`](https://github.com/Jopqior/tile-rs/commit/b1772be1df05edeb475cb7ee16e644a368c2e04c) | 本轮 `origin/wayfinder/2-metal-correctness-ci` 解析到此提交 |

本轮用 `git merge-base`、`git merge-base --is-ancestor`、`git show -s --format='%H %P %s'` 核对：

```text
c3c8ec0 ── 上游同步及 emitter 修改 ── e8d1acd
    ├──── 4e66bc8（父提交就是 c3c8ec0）
    └──── 规划历史 ── e848c7a ── b1772be
                         4e66bc8 ──┘（第二父提交）
```

- `merge-base(e8d1acd, 4e66bc8)` 和 `merge-base(e8d1acd, b1772be)` 都是 `c3c8ec0`；两次祖先检查退出 1。
- `4e66bc8` 是 `b1772be` 的祖先；入口目录与 workflow 在 D 和规划提交之间没有差异。
- D→U 的双端树 diff 显示 `metal_correctness` 删除，仅描述两棵树的差别，不是删除提交的证据。上游新增测试按 C→U diff 审计：19 个 `tile_codegen/tests/msl_*.rs` 全为新增，`tests/kernel_correctness` 全为新增，`tile_spec` 的接线和 feature scenarios 有修改，旧 `phase6_fusion_lowering.feature` 被删除。
- API 还列出 `9401ea0`、`90b0fbd` 的后续运行；它们不是题定交付基线。本文不把其元素边界扩展计入 D 的保证，也不凭其成功推广至 U。

规划文档通过 `git show origin/wayfinder/2-metal-correctness-ci:<path>` 读取，未 checkout 规划分支：[领域语言](https://github.com/Jopqior/tile-rs/blob/b1772be1df05edeb475cb7ee16e644a368c2e04c/CONTEXT.md)、[tracker](https://github.com/Jopqior/tile-rs/blob/b1772be1df05edeb475cb7ee16e644a368c2e04c/docs/agents/issue-tracker.md)、[领域文档规则](https://github.com/Jopqior/tile-rs/blob/b1772be1df05edeb475cb7ee16e644a368c2e04c/docs/agents/domain.md)。该树未列出相关 ADR。

原始决策逐票读取，以最终 resolution 为准：[边界 #5](https://github.com/Jopqior/tile-rs/issues/5#issuecomment-5662584699)、[首批范围 #6](https://github.com/Jopqior/tile-rs/issues/6#issuecomment-5663077264)、[最终比较 #11](https://github.com/Jopqior/tile-rs/issues/11#issuecomment-5692509661)、[仅日志 #7](https://github.com/Jopqior/tile-rs/issues/7#issuecomment-5693495760)、[CI 验收 #8](https://github.com/Jopqior/tile-rs/issues/8#issuecomment-5693966058)。#11 已取代 #6 的 f64 参考及严格误差推导前置要求；不能拿上游 f64 手写参考重新恢复旧政策。另读 [#4 的研究及限定补充](https://github.com/Jopqior/tile-rs/issues/4#issuecomment-5662347359)、[#10](https://github.com/Jopqior/tile-rs/issues/10#issuecomment-5662963030)、[#3](https://github.com/Jopqior/tile-rs/issues/3#issuecomment-5662283593)、[#9](https://github.com/Jopqior/tile-rs/issues/9#issuecomment-5663239242)。Issue/comment URL 是决策定位链接，正文可编辑；源码证据则固定完整 SHA，不使用分支 tip 链接充当不可变证据。

## 2. 测试到底走到哪里

### 2.1 可公开调用的完整 MLIR emitter 入口

[U manifest][u-manifest] 是独立 workspace，默认 feature 为空，`emitters` 无 LLVM、MLIR 库或 rustc 私有 API 依赖；macOS dev-dependency 为 `metal = "0.29"`。实际链路为：

```text
测试内 MLIR 字符串
→ TargetRegistry::with_builtin().select("msl").emit(...)
→ EmitterTarget / convert_mlir_to_msl
→ mlir_parse + mlir_to_msl + canned / msl_kernel_writer
→ 新生成的 MSL
→ macOS 分支：new_library_with_source → pipeline → buffers
→ dispatch_thread_groups → commit → wait_until_completed → contents 回读
→ 测试内 Rust CPU 参考，或同 kernel 的另一个配置输出
```

接线依据：[lib.rs][u-lib]、[emitters.rs][u-emitters]、[EmitterTarget][u-targets]。源码 `mlir_to_msl.rs` 现从 `mlir_parse` 导入解析函数，并 path-include `mlir_to_msl_canned.rs` 与 writer。它验证 MLIR 边界，不验证 Rust→MLIR；也不使用下载 dylib。用 registry 复用可避免各 harness 自己维持源文件模块列表，这是候选方案，不是本票作出的架构决定。

### 2.2 直接生成器、文本 golden 与 GPU 执行必须分开

[U `emit_tail_tests`][u-tail] 的 `check` 两次调用 `Fn(&mut String)`，检查非空、相等及 `contains(needle)`。**没有 MLIR 输入、Apple 编译或 GPU 调度。** 它们适合内部生成行为回归，不能提供数值证据。

新增 integration tests 通常在同一文件内混合三种检查：默认参数生成的 MSL 与 `benchmarks/ds4_msl/emitted/*.metal` 字节相等；参数传递及非法参数拒绝；macOS 数值执行。golden 是已提交生成源码，不是独立数学真值。只有最后一种执行了 GPU 才能称数值验证，且还须说明 oracle 独立性。

本轮对 U 跟踪的 Rust 文件检索 `new_library_with_source` / `Device::system_default` 并回溯输入。识别到的执行路径是上述 integration tests 和下面的内嵌 ondevice 模块；**没有在这些已审计路径中找到“只调用直接生成器、不经 MLIR”的 GPU 数值测试。** 不应虚构此类覆盖；若今后移植直接生成器 GPU 测试，它至多覆盖生成器→GPU，仍需另补 MLIR 识别、参数解析和 dispatch 选择验证。`msl_indexer_score_one_direct` 仍经 registry；`msl_ported_iquant_contract` 则读源文件和契约表作文本断言，连 Metal 编译也没有。

### 2.3 内嵌 ondevice 测试不是现成的公开 Cargo 目标

[U emitter 的 29129–32445 行][u-ondevice] 增加 `matvec_i8_v4_ondevice` 与 `laguna_ondevice`，条件是 `all(test, target_os="macos", feature="ondevice-metal")`，测试还标 `#[ignore]`。

- i8 matvec correctness、Q2/Q3/Q4/Q5/Q6/Q8、MXFP4、head RMSNorm/RoPE、QKVG、GQA 等数值函数通过 `convert_mlir_to_msl` 或最终调用它的 `emitted_one` 生成 MSL，用确定性权重及手写解量化/CPU 公式比较。不能因它们在 emitter 文件内部，就归类为直接生成器测试。
- `max_rel`、全局 `max_abs/scale`、f32 累积等政策与 #31 不同；某些最大值归并缺少逐元素 finite 拒绝。不能把这些阈值推广给首批 f32。
- `laguna_emitted_lib_concat_compiles` 读取已生成的 `.metal` 文件，属于库拼接编译检查，不是当前 emitter 数值执行。性能 A/B / decode throughput 也不提供 #31 的数值保证；大模型形状和 best-of-3 计时不应搬进正确性门禁。
- 公共 `tile_codegen` 没有声明 `ondevice-metal` feature；`rustc_codegen_tile` 公开目录无 Cargo manifest，注释中的 `-p rustc_codegen_tile` / `-p mlir_to_msl_tests` 不是本公共树可直接照跑的入口。本轮实际请求 `--features ondevice-metal --no-run`，退出 101：package 不包含该 feature。

因此这组资产是“适配或仅参考”，不是 `cargo test --features emitters` 已验证的 GPU 用例。若选择接入，需公开目标、依赖、显式非 ignored 清单、隔离性能测试，并逐项审查 oracle 与资源规模。

## 3. 复用矩阵

分类口径：**直接复用**指不改源可作为其原有证据层的辅助检查，不意味着直接满足 #31；**适配**指保留 fixture/执行链后补充门禁契约；**仅参考**指只借输入、数学定义或接线；**不可用**指当前公开树/免费 Metal 环境不能原样运行该路径。

### 3.1 家族级矩阵

| 测试家族 | 被测入口、参考来源及比较 | 平台/依赖与证据边界 | 对 Metal CI 的复用结论 |
| --- | --- | --- | --- |
| `tile_codegen` registry、parser/emitter 单元测试 | registry 或 `convert_mlir_to_msl`；字符串、拒绝、结构断言 | 公开 std-only emitter；本轮 Linux 成功 | **直接复用为源码层检查**，不能替 GPU |
| `emit_tail_tests` | 直接 `emit_*`；关键字符串、字节确定性 | 无 GPU，无 MLIR；同文件 test module | **直接复用为内部回归 / 仅参考**，不作为 MLIR→GPU 用例 |
| `tile_spec` Gherkin | 现在通过 `tile_codegen` registry；MSL 片段、场景结果 | 公开独立 workspace；本轮成功，无 Metal runtime | **直接复用辅助检查，fixture 可适配**；不要求整套成为数值 harness |
| 新增 18 个有 macOS GPU 分支的 integration 文件 | MLIR→当前 emitter→GPU；Rust 参考或跨配置回归，见下表 | `emitters` + macOS + metal 0.29；无设备 return；本轮仅非 GPU 部分通过 | **适配**；多为 #31 之外扩展族，先明确覆盖承诺 |
| `msl_ported_iquant_contract` | 读 canned 源码和 `ported_contracts.rs`，检查 simdgroup 义务及 `migrated:false` | 非 GPU 静态契约；不证明运行正确 | **直接复用辅助检查**，仅参考调度约束 |
| 内嵌 ondevice i8/Laguna | 多数 MLIR→GPU→手写 CPU；另含编译/性能测试 | feature 未公开接入、ignored；部分读文件 | **原样不可用，数值链可适配**，性能部分不纳入首批 |
| `tests/kernel_correctness` | 安全 Rust 标量循环/组合函数，对手算或 JSON golden；`assert_approx` | serde/serde_json，无 Metal；本轮 CPU 成功 | **仅参考或复用 CPU 自检**；不能当 emitter GPU 覆盖 |
| `test_998_equiv` / kernel sync 脚本 | 名称归类、参考函数不 panic 且 finite；名称同步 | 不是逐 kernel 编译/执行；归类允许最多 5% Unknown | **仅辅助清单检查**；“998”不是执行覆盖保证 |
| `tests/compiletest` | Rust UI→codegen backend；目标常量 `davinci-huawei-none` | 查找 `TILERS_CODEGEN_SO` / 已安装 backend；不提供当前公开源码 Metal 数值链 | **仅参考输入定义，原样不适用** |
| `tests/npu_correctness` | Ascend runtime / 动态加载；非 Metal | CANN/NPU 依赖；免费 macOS Metal 不是其环境 | **对本票执行路径不可用**，CPU 公式可另审 |
| D `metal_correctness` | MLIR→当前源码 emitter→Metal→同输入 PyTorch f32；全量输出、输入与 pad 比较 | metal 0.33.0，独立 workspace，PyTorch；历史小 add 真执行 | **优先评估迁移复用**，非 U 原样可用、非首批全量 |

源码：[U tests 目录][u-tests]、[tile_spec manifest][u-spec-manifest] / [steps][u-steps]、[CPU lib][u-cpu-lib] / [998][u-998]、[compiletest][u-compiletest]、[NPU manifest][u-npu]、[D README][d-readme]。

### 3.2 新增 macOS integration tests 逐族盘点

下表文件均位于 `crates/tile_codegen/tests/`，链接固定 U。除单列静态契约外，**均先经 registry 的完整 MLIR emitter 调用**；函数名中的 direct、staged、ported 不能改变这个事实。数据通常来自文件内固定数组、整数构造或固定种子 PRNG；以下列出 oracle 和政策差异，而非把文件数当覆盖率。`T×max(|ref|,F)` 表示测试原有逐元素阈值，不是 #31 的 `atol+rtol*|ref|`。

| 文件 / 家族 | 数值执行和参考 | 比较与特殊限制 | 复用边界 |
| --- | --- | --- | --- |
| [msl_bitonic_sorts][t-sort] | 整数排序、浮点 argsort，CPU 排序 / 索引 | `assert_eq!`；容量、top-k、行数变化 | 接线和合法/非法容量 fixture 可复用；不是 gather/transpose 首批替代 |
| [msl_router_finalize_one][t-router] | expert/top-k/bias/hash/token 配置；CPU sort replay | 整数 ID 精确；参考显式重放排序 | 可适配路由回归；重放同算法的独立性要另审 |
| [msl_mul_mm_id_map0][t-map] | generic intrinsic 与旧别名；CPU expert→ID 分桶 | counts、有效 ID 区域精确；检查部分 pipeline 能力 | 可适配多输出；不能视作全部缓冲区保护区已检查 |
| [msl_indexer_score_one_direct][t-score] | 多 head count；确定性输入、CPU 点积/组合 | `1e-4×max(|ref|,1)`，要求 SIMD width 32 | 是完整 MLIR 链，不是直接 emit；专用阈值不能继承给 add |
| [msl_indexer_scores_tiled][t-tiled] | f32/half、多形状；CPU score reference | f32 `1e-4`、half `2e-2` 系数；SIMD width=32 与线程上限检查 | 可适配 mixed precision；需核对参考与 half 舍入含义 |
| [msl_indexed_mixed_attention][t-mixed] | 多 head_dim/head/half staging/batched；CPU attention，含 top-k 跳项语义 | `1e-4×max(|ref|,1e-2)`，手写 half 转换、SIMD 检查 | 稀疏输入、阶段参考可复用；不等于基础 softmax 覆盖 |
| [msl_flash_attn_ref][t-fa] | plain/vector，dk/dv 含 512、64、128/256、32/16；5 query、7 key；量化到 1/256 的共同输入 | CPU f64 softmax 累积后转 f32；`1e-4×max(|ref|,1e-2)` | 完整 MLIR 路径；不是同 dtype PyTorch，精度政策待定 |
| [msl_flash_attn_staged][t-fas] | 分别运行 setup/score/out/out_ms；CPU query echo、softmax 分母/最大值及 attention 输出 | `2e-3×max(|ref|,1e-2)`；各阶段特定布局/调度 | 可复用分阶段绑定；不是上一阶段 GPU 输出接下一阶段的整链测试 |
| [msl_flash_attn_vec_staged][t-fav] | 分别运行 vector setup/score/out/out_ms；CPU 对应阶段参考 | 同族 `2e-3` 系数；不同 dk/dv | 同上；不能据此额外声称 vec_reduce 或阶段间交接已验证 |
| [msl_laguna_decode][t-decode] | 多 split 的 GQA decode；CPU attention，显式 half 输入处理 | `1e-4×max(|ref|,1e-2)`；检查 pipeline 线程能力 | 可适配 split 边界，不能推广至所有 attention |
| [msl_fp8_kv][t-fp8] | store 与 quantize、不同 block；CPU E4M3FN/half round-trip | `1e-5×max(|ref|,1e-3)`；检查 kv/raw 等不同输出 | 有较有价值的多区域检查，但 FP8 不在旧首批 |
| [msl_hc_expand][t-hc] | 不同 HC unroll；CPU 加权组合 | `1e-5×max(|ref|,1e-3)` | 可适配，不能替 `(a+b)*c` 的独立 MLIR 解析用例 |
| [msl_hc_expand_q8_0][t-hcq] | Q8 matvec + HC expansion，构造量化权重和 CPU 解码/组合 | `2e-3×max(|ref|,1e-2)`；多输出 | 可复用量化布局 fixture；扩展范围需决策 |
| [msl_mul_mv_q8_0][t-q8] | dense/routed Q8，多 split；CPU 解量化 matvec | `1e-3×max(|ref|,1e-2)` | 可适配，但不是 f32 `A @ B.T` |
| [msl_q8_0_fused_matvecs][t-q8f] | Q8 pair-SwiGLU/attention-low；CPU 融合参考 | `2e-3×max(|ref|,1e-2)` | 可适配融合族，保持量化和激活政策分离 |
| [msl_laguna_q8_0_matvec][t-lq8] | Laguna Q8，不同工作分配；CPU 解码点积 | `2e-3×max(|ref|,1e-2)` | 可复用调度/数据，不继承旧基础首批保证 |
| [msl_laguna_matvecs][t-lmv] | QKVG、attention-output residual；CPU half matvec/残差 | `2e-3×max(|ref|,1e-2)`，多输出 | 可适配，需各输出的统一状态报告 |
| [msl_wide_matvecs][t-wide] | paired half 与 routed IQ2_XXS；固定种子权重，48 行、K=512、2 experts | half 有独立 CPU（`2e-3`）；IQ2 仅与 shipped split GPU 输出比（`1e-4×max(|base|,1e-3)`） | **必须拆分结论**：half 是语义参考；IQ2 是跨配置行为回归，共同错误可漏检；48 行整除设计也不能证明尾部 |
| [msl_ported_iquant_contract][t-iq] | 读生成契约表与 canned 源码，推导固定 staging 的 simdgroup 要求 | 文本/算术断言，`migrated:false` 明示义务未齐 | 无 GPU；静态辅助，不承诺量化执行正确 |

共同缺口（由上述 GPU 函数源码可见）：使用 `CompileOptions::new()`，没有统一显式 `set_fast_math_enabled(false)`；没有统一来源/设备日志与预期清单；缺设备均正常 return；并非普遍读取并精确核对所有输入与输出 pad。多数数值断言会因 NaN 比较为 false 而失败，但这不等于已具备两侧 finite、自检、shape/dtype 及完整性政策，尤其不能用某个 `max` 聚合器推广 finite 保证。编译错误通常附 MSL panic，有价值，但没有统一失败阶段和后续未执行状态。

## 4. CPU 测试能给首批什么，不能给什么

[CPU strategy][u-strategy] 声称三层覆盖；实际 [`src/lib.rs`][u-cpu-lib] 明确是 NPU kernel 的安全 Rust 镜像。[golden generator][u-generate] 首选 PyTorch，但缺 PyTorch 时退回 NumPy；已提交 JSON 不记录每次运行的实际参考版本。Rust `assert_approx` 检查长度后使用 `err < tol || (|want|>1e-6 && err/|want| < tol)`，不是 #31 的容差公式。

| #31 路径 | 可借资产 | 仍须补足 |
| --- | --- | --- |
| add/sub/mul | emitter 的 MLIR/文本用例；CPU 基础运算、手算向量；D 的 add | 三条各自单元素、向量宽度、实际 threadgroup 前后、多组尾部 GPU；不是用 CPU pass 填空 |
| exp | CPU activation 的 exp、emitter MLIR fixture | 独立 exp MLIR→GPU；有限正常数域与全输出 |
| `(a+b)*c` | 组合相关文本/CPU 测试可供构造 | 精确声明两步顺序、MLIR 对应与跨组尾部，不能用 HC/SwiGLU 替换 |
| 行求和 | [`reduce_sum`][u-reduce] 的标量参考和 reduce 用例 | 多行轴、跨 SIMD/threadgroup、非整齐行宽、抵消；CPU flat sum 不是完整行布局测试 |
| softmax | CPU normalization/attention 参考及 MLIR fixture | `dim=-1` 基础入口独立执行；flash attention 通过不代表基础 softmax 通过 |
| 转置矩阵乘 | [`matmul_transposed_b`][u-matmul] 明确 `A(M,K) @ B(N,K).T`；手算/JSON | 不可误用 transposed_a/both；当前 emitter K 展开为 **16**，旧 K=3/4/5 仍有价值但应额外检查 15/16/17，不能继续称 3/4/5 覆盖当前展开边界 |
| gather | CPU index 输入/索引参考 | 核对合法整数行号与 emitter 参数编码；首尾、逆序、重复、stride/尾部、搬运精确 |
| transpose | CPU 布局例子、emitter fixture | 单行/单列/非方阵与跨步进 GPU，坐标编码和保持区精确检查 |

K 展开事实见 [U 常量及说明][u-msl] 与 `emit_matmul_transposed_msl`；说明中的既往性能数据是源码作者的历史声明，本轮未重测。这里建议增加当前实现风险边界，不改变 #31 原先规定的合法输入或擅自删除旧边界。

## 5. 公开构建与本轮观察

所有本轮命令只在 `/tmp/tile-rs-research-metal-test-reuse` 执行，分支 `research/metal-ci-upstream-test-reuse` 基于 U。父工作区未切换、未写入源码；父目录未跟踪 `Cargo.lock` 未触碰。以下是 **Linux 本地非 GPU 实测**，不是 GitHub runner 结果：

```sh
rustc --version  # 1.91.0-nightly (f34ba774c 2025-08-03)
cargo --version  # 1.91.0-nightly (840b83a10 2025-07-30)
cargo test --manifest-path crates/tile_codegen/Cargo.toml --features emitters
cargo test -p kernel_correctness
cargo test --manifest-path crates/tile_spec/Cargo.toml
cargo test --manifest-path crates/tile_codegen/Cargo.toml --features ondevice-metal --no-run
```

| 命令 | 观察 | 不应推出 |
| --- | --- | --- |
| codegen + emitters | exit 0；汇总 366 passed、5 ignored（含 doc-test）；有 warnings | macOS GPU 函数未编译/执行；ignored 不算通过 |
| CPU correctness | exit 0；382 passed；拉取 serde 等公共依赖 | 没有生成 MSL，没有 GPU，没有本轮 PyTorch golden 再生成 |
| tile_spec | exit 0；14 个 lib tests + 1 个场景总入口 | 一个场景总入口不是一个 GPU 用例；不存在数值执行证据 |
| ondevice feature | exit 101，feature 不存在 | 不是 Metal 设备失败，是公开 Cargo 接线缺失 |

这些计数仅用于复现识别，不是正确性覆盖度。没有添加 `-D warnings`，没有运行 coverage instrumentation、全 root workspace、compiletest、NPU、macOS cross-build 或历史 harness 移植。root 仅定向 `-p kernel_correctness` 成功不能证明所有 CANN 相关 member 在普通 Linux 可构建。

新 root manifest 已显式列举公开成员，`tile_codegen`/`tile_spec` 作为独立 workspace 排除，旧“根 manifest 拉入大量缺失路径”不能继续当该定向入口的阻塞。[U root manifest][u-root]。两个旧缺失模板现在位于 `crates/deepseek_metal/templates/`，相关文本测试本轮不再复现旧失败。`tile_spec` 不再自行 path-include 整套 emitter，而是依赖 registry。公开构建改善不代表整个 rustc 私有驱动已公开，也不保证发布 dylib 与 U 同源。

[U coverage workflow][u-coverage] 在 Ubuntu 跑覆盖率，显式清空默认 `-D warnings`，canned 生成函数不计入 gate；因此作者所说的约 64% 和新增 CPU 测试不能用来证明 Metal 数值覆盖。[上游同步说明](https://github.com/yijunyu/tile-rs/issues/3#issuecomment-5705453181) 只是同步动机，不是 GPU 执行日志。

## 6. 已交付入口的真实范围与迁移障碍

[D README][d-readme]、[main][d-main]、[gpu][d-gpu]、[reference][d-ref]、[compare][d-compare]、[CLI tests][d-cli] 和 [workflow][d-workflow] 的实际契约：

- 只交付 `add_f32_small`：8 个固定 f32 元素、两个输出 pad、输入不变检查；同输入 PyTorch CPU f32；Rust 比较器实施 `rtol=1.3e-6, atol=1e-5`，并检查类型、形状、有限值与保持区。并非实际在 Python 调用 `torch.testing.assert_close`，而是实现其所选数值政策。
- path-include 当前 emitter/parser；binary `test=false`，CLI 集成测试驱动自检，不顺带执行 emitter 的整套 fixture 测试。GPU 通过 metal 0.33.0，fast-math 显式 false，线程组规模从 pipeline 能力与输入长度取得。
- 缺设备/PyTorch 是 unverified、退出 2，不是假绿；默认模式执行 delivered list，自检和漏跑检查；`--case`、`--self-check` 均不等于完整 CI。
- workflow 对 PR、main push、手动触发，无路径过滤，无数值自动重试；仅 PyTorch 下载有限重试；仅日志，没有 artifact 上传。job 名明确 initial add / not issue 31 complete。

**历史执行观察**：本轮只读取已有 [push run 35091228128](https://github.com/Jopqior/tile-rs/actions/runs/35091228128) 的 API 和日志，未发起运行。API `head_sha=4e66bc824872a95bfae51f6503bfbdb360199fe3`、conclusion=success；job 标签 `macos-15`，开始 `2026-09-16T11:36:39Z`，结束 `11:37:39Z`。日志包含：

```text
SOURCE_COMMIT=4e66bc824872a95bfae51f6503bfbdb360199fe3
ImageOS=macos15
CASE id=add_f32_small
PYTORCH_VERSION=2.14.0
METAL_DEVICE_COUNT=1
METAL_DEVICE_SELECTED name="Apple Paravirtual device" ...
COMPILE_FAST_MATH=false
METAL_DEVICE=Apple Paravirtual device ... threadgroup=8 groups=1
RUN_RESULT=pass
```

这是 D 的当前 emitter 小 add 全链路历史证据，比手写 MSL 探针更强；不是 U 的通过证据，不是物理 Apple GPU 覆盖，更不是首批全量。另一个同 SHA [manual run 35091344925](https://github.com/Jopqior/tile-rs/actions/runs/35091344925) API 显示成功，本轮不以未逐项审阅的日志增加保证。

**交付自检也不等于 #31 全部验收已完成**：D 的设备不可用自检只是检查 `Status::Unverified.is_success()` 为 false，不是实际注入无设备路径并检查整个 CLI 退出状态；`gpu.rs` 在 wait 后直接回读，未单独核对 command-buffer status/error。迁移时应保留这些已实现的责任，同时把端到端故障注入和执行状态诊断列为审查项，不把现有自检名称当完整证明。

**移植不能原样复制即宣称完成**：D main 仍声明 path `mlir_to_pto.rs`，U 已不含该文件。U 的解析接口已集中到 `mlir_parse`，emitter 还新增 sibling canned/writer。选择继续独立 harness 时必须更新接线或改用 registry，并重新验证新 emitter；选择复用上游 cargo tests 时则应迁移 D 的状态、自检、日志、保护区和比较政策，而非仅删除这些责任。本轮没有尝试移植、没有声称移植后可编译。D 的通过记录不能作为移植验收。

## 7. 免费 runner 的可行性与成本证据

GitHub 官方文档固定到 [`github/docs@73f38c5edb3f47a01eb3db96fa723775e2bfeaf9`](https://github.com/github/docs/commit/73f38c5edb3f47a01eb3db96fa723775e2bfeaf9)：

- [标准 runner 表][gh-runners] 把 `macos-15` 列为 arm64、3 CPU（M1）、7 GB RAM、14 GB SSD；公开仓库标准 runner 免费且不限量。仓库 API 本轮返回 `private=false, visibility=public`。
- [计费规则][gh-billing]：larger runner 即使公开仓库也收费；日志和 job summary 不计入 artifact 存储额度。不能把免费计算误写成任意 cache/artifact 存储免费。本研究不要求附件或额外服务。
- #9 的 [run 34838340638](https://github.com/Jopqior/tile-rs/actions/runs/34838340638) 是更早的 16 元素手写 MSL 探针；D 的上述 60 秒 job 提供了真实 emitter 小 add 的独立历史样本。60 秒是整 job 耗时，不是 GPU 性能，不含排队，也不是所有新测试未来耗时的估计。

因此免费标准 arm64 runner 上验证链并非未经任何实测的设想。但 integration tests 的 SIMD=32、特定 threadgroup memory、Metal 功能和大形状是否全部适用 Paravirtual device，仍未证明；Linux build 不能消除这项未知。新增数值族的 MSL 编译耗时、CPU 参考成本、GPU 时间和内存峰值均未测。金钱成本依据官方标准 runner 政策，执行预算应在后续授权验收中记录每族耗时，不能用“文件不多”推算。

可行候选测试命令是 `cargo test --manifest-path crates/tile_codegen/Cargo.toml --features emitters --test msl_flash_attn_ref -- --nocapture` 等显式 target，但 **在修复设备缺失返回成功、执行清单和数学配置以前，退出 0 不能作为可信 GPU gate**。本票不运行这些 macOS 命令、不修改其实现。

## 8. 对旧 spec 的精确差异

| #31 的前提/要求 | 新证据 | 处理含义（非决议） |
| --- | --- | --- |
| 旧入口存在构建/缺模板障碍；tile_codegen 接线失败 | U 三个定向公开入口本轮通过；parser 中立化，模板公开 | **历史阻塞描述过时**；不要继续把独立入口说成唯一可构建路线 |
| 用 `tile_spec` 的 path-include 接线作先例 | U tile_spec 已通过 registry 依赖 tile_codegen | 新候选 seam 是公开 registry；D 旧 path 引用不能照抄 |
| 当时既有测试主要文本，GPU 证据只有探针 | U 新增完整 MLIR→GPU 测试代码；D 已有真实 emitter 小 add 运行 | **“无执行测试/入口从未实现”失效**；仍须区分源码存在与实际运行 |
| 独立验证入口是选定实现方式 | 上游已有可用公共库及执行用例，D 已实现部分职责 | 入口组织应重审；复用、独立或混合都可能，证据不替人工选型 |
| MLIR→当前源码 emitter→GPU 验证边界 | 新 integration tests 能满足这个路径边界，直接生成器文本测试不能 | **边界仍成立**，不能降成仅 generator 或旧 MSL golden |
| PyTorch CPU 同 dtype、默认容差、finite/shape/dtype 检查 | 新上游主要手写 Rust f32/f64/量化参考与不同阈值，IQ2 部分以同 kernel 为 oracle | 参考选择可重审，但不能默认为符合 #11；若沿用则适配，否则明确决策替换 |
| 基础 f32 首批，各路径及边界独立 | 新 GPU 家族多为量化、attention、专用融合；CPU 同名测试不提供 GPU 保证 | **首批全量仍未证明**；扩展或换范围必须显式决定 |
| K=3/4/5 触及旧展开边界 | U f32 transposed matmul 展开因子 16 | 保留小 K 风险，补当前 15/16/17；spec 的“展开边界”具体映射需更新 |
| fast-math off、保持区域、手算锚点及反例 | D 小 add 部分实现；新上游 GPU 无统一该契约 | 不因上游测试多而删除验证链自检；逐族补证 |
| 环境不可用、漏跑不得通过 | macOS 无设备正常 return，Linux cfg 排除，ondevice feature 未接线且 ignored | **直接接 cargo test 有假绿风险**，门禁适配是前置 |
| 免费标准 macos-15、设备真实记录 | 官方政策和 D 历史执行支持小链可行 | **资源约束仍成立**；不能推导新族全通过或物理 GPU 保证 |
| 每 PR/main/manual 完整清单，独立报告，仅日志 | U coverage 是 Ubuntu；D workflow 只跑 delivered 小 add | 调度/日志规范不能由 coverage 替代；承接 D workflow 时重新验证合并树 |
| 接入完成与全量通过分开 | D 已交付初始 add，U 新族仍未实测 | **仍成立**；不能因单 run 绿或 CPU suite 绿关闭首批实施验收 |

## 9. 可操作缺口与交回决策

1. **先固定承接树**：未来实施要说明 U 与 D/规划分支如何整合，记录真正受测 commit。合并历史、移植代码与借用历史成功是三件事。不能在新树缺文件时重做全部交付，也不能把旧成功“继承”过来。
2. **确定入口组织**：比较 registry + 上游 tests、继续独立 harness、混合三条路线的维护面。共同必须承担当前源码溯源、非跳过执行、参考比较、自检、完整性和日志；本文不替用户保留或删掉 harness。
3. **明确首批范围与 oracle 政策**：旧基础 f32 清单若保留，需要补 GPU fixtures/边界；若选已有 attention/quant 族先行，则是覆盖承诺变更，不能记作旧票实现完成。IQ2 同 kernel 配置比较只能标行为回归，另需语义 oracle。
4. **把执行完整性变成可验收对象**：显式平台/feature、预期用例清单、拒绝无设备成功、ignored/过滤/空测试检查；对无法汇总、依赖/fixture 缺失、编译与执行失败给出非零状态和未执行原因。
5. **逐族审查比较与数学模式**：记录 CPU/GPU 共同定稿输入、dtype/量化编码、全部输出、保护区、finite、有效容差、fast-math；用错误注入证明比较器和漏跑检查会失败。不能直接复制 `1e-3`/`2e-3` 或全局最大值比值。
6. **公开接线验证与实际 runner 验收分两步**：本轮公开 CPU 构建已观察；下一阶段仍需在免费 macos-15 上编译 macOS Rust 分支及生成 MSL、运行全部选定用例并记录设备、配置和耗时。没有这些证据，不宣称 U 的 GPU 基线通过。
7. **旧 spec/实施票如何承接仍交地图决定**：可局部修订，也可关闭被替代票后重新规划；关闭替代不等于实现完成。本研究不修改共享地图，不关闭票，不发 resolution comment。

## 证据索引

下面 U / D 的所有源码链接均固定完整 SHA。它们是本报告表格的依据，不是最新分支内容。本文 Linux 观察为本轮执行记录；历史 GPU 观察明确链接已有 run；其余静态可达性及未测限制已逐节注明。

[u-root]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/Cargo.toml
[u-manifest]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/Cargo.toml
[u-lib]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/src/lib.rs
[u-emitters]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/src/emitters.rs
[u-targets]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/src/targets.rs
[u-msl]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L43-L89
[u-tail]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L26596-L26618
[u-ondevice]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L29129-L32445
[u-tests]: https://github.com/Jopqior/tile-rs/tree/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests
[u-spec-manifest]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_spec/Cargo.toml
[u-steps]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_spec/src/steps.rs
[u-coverage]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/.github/workflows/coverage.yml
[u-strategy]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/kernel_correctness/TESTING_STRATEGY.md
[u-cpu-lib]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/kernel_correctness/src/lib.rs
[u-generate]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/kernel_correctness/golden/generate.py
[u-998]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/kernel_correctness/tests/test_998_equiv.rs
[u-reduce]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/kernel_correctness/src/reduce.rs
[u-matmul]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/kernel_correctness/src/matmul.rs
[u-compiletest]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/compiletest/src/main.rs
[u-npu]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/npu_correctness/Cargo.toml
[t-sort]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_bitonic_sorts.rs
[t-router]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_router_finalize_one.rs
[t-map]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_mul_mm_id_map0.rs
[t-score]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_indexer_score_one_direct.rs
[t-tiled]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_indexer_scores_tiled.rs
[t-mixed]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_indexed_mixed_attention.rs
[t-fa]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_flash_attn_ref.rs
[t-fas]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_flash_attn_staged.rs
[t-fav]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_flash_attn_vec_staged.rs
[t-decode]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_laguna_decode.rs
[t-fp8]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_fp8_kv.rs
[t-hc]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_hc_expand.rs
[t-hcq]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_hc_expand_q8_0.rs
[t-q8]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_mul_mv_q8_0.rs
[t-q8f]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_q8_0_fused_matvecs.rs
[t-lq8]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_laguna_q8_0_matvec.rs
[t-lmv]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_laguna_matvecs.rs
[t-wide]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_wide_matvecs.rs
[t-iq]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/tests/msl_ported_iquant_contract.rs
[d-readme]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/README.md
[d-main]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/src/main.rs
[d-gpu]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/src/gpu.rs
[d-ref]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/src/reference.rs
[d-compare]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/src/compare.rs
[d-cli]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/crates/metal_correctness/tests/cli.rs
[d-workflow]: https://github.com/Jopqior/tile-rs/blob/4e66bc824872a95bfae51f6503bfbdb360199fe3/.github/workflows/metal-correctness.yml
[gh-runners]: https://github.com/github/docs/blob/73f38c5edb3f47a01eb3db96fa723775e2bfeaf9/data/reusables/actions/supported-github-runners.md
[gh-billing]: https://github.com/github/docs/blob/73f38c5edb3f47a01eb3db96fa723775e2bfeaf9/content/billing/concepts/product-billing/github-actions.md
