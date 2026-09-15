# tile-rs 现状：crate 地图与目标转换阶段

状态：用户已确认此领域模型、crate 关系及目标转换链路足以支撑后续讨论，保留明确列出的证据缺口。本文记录源码证据，不推荐分层方案；决策详情以对应票的 resolution comment 为准。

对应决策票：[建立 tile-rs 领域模型、全仓 crate 地图与 Metal 完整转换链路](https://github.com/Jopqior/tile-rs/issues/13)。领域词汇见根 `CONTEXT.md`。

## 证据范围

- 主仓库：`Jopqior/tile-rs`，源码版本 `1ee797f1f5834ac9ce581ce69e8b8f1775ece076`。
- 与地图最初参照 `c3c8ec0169bd02757b51370ba9c6ec3b116b833d` 相比，只有 `AGENTS.md` 和 `docs/agents/` 的文档变化，生产源码未变。
- 本文路径与行号均相对此固定源码版本。可用 `https://github.com/Jopqior/tile-rs/blob/1ee797f1f5834ac9ce581ce69e8b8f1775ece076/<path>#L<line>` 查看。
- **源码事实**：读到的声明、控制流、文件或调用关系，不等于实际构建、执行成功。
- **文档声明**：README、注释、构建配方描述的行为，尚未运行重现。
- **发布流程**：工作流描述从另一仓库构建、打包二进制的过程，不证明某个已下载二进制的具体内部实现。
- **未验证**：未构建 kernel，未加载发布编译组件，未运行 GPU，也未验证所有目标代码的语法或数值正确性。
- 关联仓库 `yijunyu/tile-rs-metal` 由用户补充。其独立核对见 [Metal 关联仓库调查](https://github.com/Jopqior/tile-rs/blob/711e9b3f661cfcf991d632d76a33041a653525a0/docs/research/tile-rs-metal-source-audit.md)（独立分支 `research/metal-source-audit`；提交 `711e9b3`）。不能把下面“主仓库缺席”扩展为“整个项目不存在”。

## 1. 全仓 crate 与生命周期

`crates/` 实际有六个目录。其中五个有 Cargo manifest，`rustc_codegen_tile` 是公开源码目录而不是本树可独立构建的完整 crate。根 workspace 的 exclude 列表引用了许多不在 checkout 的目录，不能据此增加现存 crate 数量。

| 目录 | 实际角色 | 生命周期与依赖 | 证据 |
|---|---|---|---|
| `tile_std` | kernel 侧 Rust DSL；Tile、设备内存视图、UB 缓冲区、同步包装、复合操作及最小核心支持 | kernel 编译期；依赖 `tile_std_macros`；不是 host 设备管理库 | `crates/tile_std/Cargo.toml`；`src/lib.rs:1-75`；`src/tile.rs:35-135`；`src/buf.rs:1-46`；`src/pipeline.rs:1-20` |
| `tile_std_macros` | `#[tile_kernel]` 属性宏；保留符号和 kernel 标记，必要时改写视图参数并插入边界 prelude | 在主机上的 Rust 编译期执行；依赖 `syn`、`quote`、`proc-macro2`；产出仍是 Rust | `crates/tile_std_macros/src/lib.rs:1-126` |
| `tile_codegen` | `CodegenTarget`、注册表、普通转换函数适配器及 feature 控制的登记 | 独立 workspace；默认 std-only；目标代码生成阶段，不承载设备执行或目标工具链调用 | `crates/tile_codegen/Cargo.toml`；`src/target.rs:1-69`；`src/targets.rs:16-114` |
| `tile_spec` | Gherkin 风格可执行规格及目标代码生成检查 | 独立 workspace；dev-dependency 指向 `tile_codegen`；测试以 `#[path]` 引入生成器源码而非链接完整 rustc 编译器后端 | `crates/tile_spec/Cargo.toml`；`tests/cucumber.rs:1-80` |
| `tile_hal` | host 侧 Device、Stream、DeviceBuffer、KernelLauncher 等接口；目标选择与可选 CUDA 实现 | 运行期；不依赖本仓其他 crate；`cuda` feature 可选依赖 `libloading` | `crates/tile_hal/Cargo.toml`；`src/backend.rs:1-144`；`src/kernel.rs:22-42`；`src/cuda.rs` |
| `rustc_codegen_tile` | 本树只提供 `mlir_parse.rs` 与十四份 `mlir_to_*.rs` 源码 | 由其他 crate 的 `#[path]` 引入参与编译；没有本目录的 Cargo.toml、lib.rs 或 MIR 转换实现 | `git ls-files crates/rustc_codegen_tile`；`crates/tile_codegen/src/emitters.rs`；`crates/tile_spec/tests/cucumber.rs` |

默认 workspace 成员是 `tile_std`、`tile_std_macros`、`tile_hal`。`tile_codegen`、`tile_spec` 被根 manifest 排除并各自定义 workspace。`Cargo.lock` 的包名、exclude 路径及 README 表格不等于现存依赖调用图。

### 依赖与调用图

```text
kernel Rust 源
  └─使用→ tile_std ─Cargo 依赖→ tile_std_macros
                               └─编译期展开→ Rust 函数与 kernel 标记

[发布的 rustc 编译器后端：内部调用接线在本树不可见]

公开源码的接线：
  tile_codegen
    ├─默认→ DebugTarget：MLIR 文本回显
    └─emitters feature→ EmitterTarget → convert_mlir_to_* 源码

  tile_spec
    ├─dev-dependency→ tile_codegen（注册表/接口规格）
    └─#[path]→ mlir_parse + mlir_to_*（直接调用转换函数）

  host 程序 → tile_hal 接口 → cuda feature → CUDA Driver API
  Ascend 示例工具 → ACL + CCEC kernel launch（不经 tile_hal）
```

不能把源码目录位置解读为完整 crate 依赖，也不能把 `tile_spec` 的直接转换测试解读为 DSL 经 rustc 编译的端到端测试。

### crate 之外的工具

- `scripts/install.sh`：安装 release 中的预编译组件及配套文件，不从本树构建完整 rustc 编译器后端。
- `.github/workflows/codegen-release.yml:36-38,133-140`：从 `yijunyu/ascend-rs-priv` 获取源码构建编译组件；工作流还打包 LLVM/MLIR 依赖与使用配置。
- `.github/workflows/toolchain-drift.yml:19-20`：说明完整 MIR→MLIR 编译组件不在此公开树范围内。
- `scripts/coverage.sh`：运行公开 crate 的测试/覆盖率；存在对不在本树的历史目录的提及，不能当作实际调用已成功。
- `tools/ascend/README.md`：PTO→AscendC→host/device 可执行程序的命令配方。
- `tools/ascend/pto_run.cpp`、`attn_run.cpp`：ACL host 与目标 kernel 包装代码；不是通用 HAL 实现。配方的硬件运行数据属于文档声明，本次没有重现。
- 本 checkout 没有 `examples/`、`benchmarks/`，虽有根 manifest 中的排除项。关联仓库中的例子另行核对。

## 2. DSL 内部对象及边界

以下是源码中的对象关系，术语是否收入词汇表以用户确认为准。

| 对象 | 实际表示与关系 | 注意事项 |
|---|---|---|
| `Tile<R,C,T>` | 二维形状、元素类型及内部缓冲区 ID；不是 host 上装有 R×C 元素的数组 | 不由该类型推定实际物理存储；`tile.rs:35-49` |
| `GmView` / `GmViewMut` | 带形状、类型与生命周期的设备全局内存指针视图 | 不等于搬运后的 Tile，也不等于 host HAL 分配所有权；`tile.rs:85-106` |
| `UbBuf` / `L1Buf` / `L0*Buf` | 分别带有 UB、L1、L0 等内存层级语义的原始句柄 | Ascend 风格概念，不能直接改称所有平台共有的物理存储；`lib.rs:45-73` |
| `UbView<CAP,T>` | 容量型 UB 句柄；操作另外传有效长度 | 与固定二维形状 Tile 不同；不能只凭文档就宣称所有运行期长度已检查；`buf.rs:58-69,128-160` |
| `DmaPending` / `VecBuf` | 数据搬运后待同步及可计算的状态包装；`sync()` 调 barrier | `buf` 和 `pipeline` 中有不同具体类型；不等于通用异步执行器或 host command queue；`pipeline.rs:24-45` |
| `tile_*` / `__tile_*` | Rust 包装与操作入口；既有 extern 声明，也有复合 Rust 实现 | 名称前缀不能代替对具体函数体的核对；`tile.rs:164+`、`2235-2242`；`lib.rs:75+`；`kernel_ops.rs:1-120` |

安全属性不是本票的审计结论。例如 `Tile` 非 Copy 不会自动证明“恰好 store 一次”；`UbView` 的类型容量也不能自动证明每个运行期 `len` 均经过上界校验。现状模型记录接口含义，不能把注释中的安全宣传扩展成全项目已证性质。

## 3. 表示、方言与转换动作

- Rust DSL 是输入语言接口，宏展开后仍是 Rust，不直接产出 MSL。
- rustc MIR 是 Rust 编译过程的中间表示。本树没有 MIR→MLIR 实现；该箭头只能标为发布组件相关的声明与缺口。
- 公开生成器接收 MLIR **文本**。测试样例含 `llvm.func`、`llvm.call`、`!llvm.ptr<1>` 和入口标记 `hacc.entry`。
- `llvm.*` 是 MLIR 的 LLVM 方言写法，不等于 LLVM IR 文本；`__tile_*` 是调用符号，不足以证明存在 `tile.*` 方言。
- `mlir_parse::parse_module` 产出 `MlirModule`，其中函数体主要是 `Vec<String>`。这是 MLIR 文本解析结果，不是完整 MLIR API 对象或独立的 Tile IR。
- `tile_spec/features/tile_dsl_shapes.feature:2` 用 “tile IR” 指 `Tile<R,C,T>` 与 tile intrinsics 集合。根 README 架构图只写 MLIR。本文不把该规格中的宽泛称呼套到 parser 输出上。
- PTO 与 linalg 是生成器可以输出的另外两类 MLIR 方言文本。它们不是已证实位于所有目标路径中间的必经阶段。
- parse 提取文本结构；lowering 落实更具体的实现；emit 生成目标代码。这些是动作，不预设独立 pass、函数或额外 IR。

证据：`crates/rustc_codegen_tile/src/mlir_parse.rs:85-213`；`mlir_to_pto.rs:101-105`；`mlir_to_linalg.rs`；`crates/tile_spec/tests/cucumber.rs:214-226`。

## 4. 各目标转换阶段总览

公共输入边界是 MLIR 文本。主仓库公开了十四份目标代码生成器源码；`tile_codegen` 的 `emitters` 分支登记其中十三个，`tile_spec` 直接调用表包含十四个（额外含 PTO）。默认注册表仅有 `debug`。README 的“十五目标”包含未在本树公开的 AscendC 生成器，不能与这三种计数混用。

下面工具名若标“声明”，表示只在注释或配方中出现，没有本次成功调用证据。

| 源码登记/调用名 | 生成器文件 `crates/rustc_codegen_tile/src/` | 实际输出 | 后续目标编译与执行的主仓证据 |
|---|---|---|---|
| `gpu` | `mlir_to_gpu.rs` | CUDA C | nvcc/clang CUDA 工具声明；HAL 有可选 CUDA Driver 实现，但接收已编译模块，不接收 `.cu` |
| `musa` | `mlir_to_musa.rs` | MUSA 源码 | 复用 CUDA 生成文本再改写；mcc 声明；HAL 有枚举无设备实现 |
| `spirv` | `mlir_to_spirv.rs` | GLSL compute 文本 | glslang/glslc→SPIR-V 的声明；无 Vulkan host 实现。输出本身不是 SPIR-V 二进制或 WGSL |
| `msl` | `mlir_to_msl.rs` | MSL 文本 | Metal 工具及运行时调用只有注释；主仓 HAL 无 Metal 实现；关联仓另查 |
| `nki` | `mlir_to_nki.rs` | NKI Python | Neuron 工具声明；HAL 有枚举无设备实现 |
| `aie` | `mlir_to_aie.rs` | IRON Python | AIE 工具声明；HAL 有枚举无设备实现 |
| `bang` | `mlir_to_bang.rs` | BANG-C | cncc 声明；HAL 有枚举无设备实现 |
| `gaudi` | `mlir_to_gaudi.rs` | TPC-C | tpc-clang 声明；HAL 有枚举无设备实现 |
| `tpu` | `mlir_to_tpu.rs` | JAX/Pallas Python | JAX JIT 使用声明；主仓 HAL 没有对应枚举或实现 |
| `csl` | `mlir_to_csl.rs` | CSL | cslc 声明；无 host 实现 |
| `hexagon` | `mlir_to_hexagon.rs` | HVX/QNN C | hexagon-clang 声明；无 host 实现 |
| `ttmetal` | `mlir_to_ttmetal.rs` | Tensix C++ | TT-Metal 工具/API 声明；不是 Apple Metal；无 host 实现 |
| `linalg` | `mlir_to_linalg.rs` | linalg MLIR 文本 | MLIR 校验/CPU bridge 声明；本树没有 CPU 执行链路 |
| `pto` | `mlir_to_pto.rs` | PTO-MLIR 文本 | `tools/ascend` 有 ptoas→AscendC→ccec 配方及 ACL host 源码；不通过主仓 HAL |
| `cpp` | 本树缺席 | AscendC（文档声明） | HAL 可解析该名字且缺省使用它，但本树无 Ascend HAL 实现 |
| `debug` | `tile_codegen/src/targets.rs` | 带注释的 MLIR 回显 | 接口测试用途，不是硬件执行目标 |

表中各源文件的开头注释和 `convert_mlir_to_*` 函数为输出证据；登记集中在 `crates/tile_codegen/src/targets.rs:59-80`，测试调用表在 `crates/tile_spec/tests/cucumber.rs:50-80`。HAL 支持范围见 `crates/tile_hal/src/backend.rs:5-130`、`Cargo.toml:8-12`。

不存在可由本树证明的 ROCm/HIP 或 WGSL 生成器，亦无名为 CPU 的独立生成器。NPU 是硬件类别，不是唯一平台或转换路径；目标名存在不保证所有 DSL 操作均可用。

## 5. Metal 在主仓的逐阶段追踪

```text
Rust kernel + tile_std
  → #[tile_kernel] 展开：Rust 函数、裸指针边界和 kernel 标记
  ⇢ rustc MIR ⇢ MLIR 文本                [内部实现不可见]
  → parse_module
      MlirModule / MlirFunc / 参数 / body_lines
  → convert_mlir_to_msl / generate_func_msl
      入口筛选、操作分类、操作链分析、具体实现选择与 emit
  → MSL 字符串
  ⇢ 目标编译与执行                      [主仓只有注释；关联仓另核]
```

### 宏与编译组件边界

`tile_std_macros/src/lib.rs:52-126` 的 `expand` 将视图参数改写为裸指针并插入 prelude，添加 `#[unsafe(no_mangle)]` 与 `#[tile::kernel]`。它不生成 MLIR。`tile_std` 中的 intrinsic 声明也不等于可在 host 上直接调用的普通库实现。

发布工作流表明可加载的 `librustc_codegen_tile` 从另一仓库构建。该二进制内部怎样消费 MIR、产生 MLIR、选择目标及驱动工具链，不能由公开转换函数反推为完整事实。

### 文本输入与 parse

`mlir_parse.rs:85-213`：扫描 LLVM 方言函数/全局声明，支持特定 pretty/generic 文本形状，提取参数并保留正文文本行。没有完整 MLIR 方言注册、验证器或 pass manager。

MSL 生成器通过 `mlir_to_pto` 的再导出取得部分 parser API，见 `mlir_to_msl.rs:45-51`、`mlir_to_pto.rs:101-105`。因此文件间存在真实源码耦合，但它不表示 Metal 必须先 emit PTO。

### 分析与 emit

`mlir_to_msl.rs:431-511` 的 `convert_mlir_to_msl`：

1. 调 `parse_module`。
2. 写 MSL 头及按需辅助定义。
3. 只处理带入口标记、非空且非 builtin helper 的函数。
4. 调 `generate_func_msl`，最终返回字符串；没有入口时报错。

`generate_func_msl`（`519+`）登记指针参数、调用 `classify_body`，再尝试 matvec、elementwise、gemm、partition 等操作链分析。部分路径根据所识别的 kernel 模式写预定实现，其他路径根据组合操作生成表达式。并非统一遍历完整 MLIR 语义图，也并非所有 kernel 都只按单个名字套模板。

这里的 lowering/实现选择与 emit 交织，没有由本树可见的“Metal 专用 IR”强制隔开。

### 输出及主仓终点

`convert_mlir_to_msl` 返回 `Result<String,String>`。`EmitterTarget` 包装后给出 `EmitOut { source, ext, meta }`，其中 Metal 对应扩展名 `metal`。`CodegenTarget` 没有目标工具链调用或 GPU launch 方法（`target.rs:1-69`）。

MSL 文件头写出的 `xcrun metal`、`xcrun metallib` 和 `new_library_with_source` 是注释，不是被执行的调用。主仓没有实现 Metal 的 `KernelLauncher`，`BackendSelector::device` 对 Msl 返回不可用。

### 关联仓补齐的可见环节

固定关联仓版本 `yijunyu/tile-rs-metal@23130eeba6df91c555dbc71135f836eddb92568d`，详见 [独立源码核对及固定版本链接](https://github.com/Jopqior/tile-rs/blob/711e9b3f661cfcf991d632d76a33041a653525a0/docs/research/tile-rs-metal-source-audit.md)。

```text
关联仓 example 的 kernel Rust 源
  → 使用本地 tile_std + tile_std_macros 副本

关联仓 example/build.rs
  → KernelBuilder::new("kernels").codegen_path("metal").build()
  → 子 cargo + -Zcodegen-backend=<外部动态库>
      [MIR→MLIR→选路/生成/目标编译的二进制内部接线不可见]
  ⇢ builder 预期得到 tile.so / tile.gen.cpp / tile.o 之一
  → example 在返回路径旁搜索 .metal / .metallib
  → 将路径交给 host Rust 编译

关联仓 example/src/main.rs
  ├─有可用 .metallib→ new_library_with_file
  └─否则读 .metal→ new_library_with_source（运行时目标编译）
  → get_function("rsqrt") → compute pipeline
  → 创建输入、输出及长度参数的 Metal buffer
  → 绑定 buffer(0/1/2)
  → dispatch_thread_groups（1 组 × 256 线程）
  → commit → wait_until_completed
  → 读取输出并计算与 CPU rsqrt 的误差
```

源码明确写出了构建器调用和 host 执行两端，但中间仍有外部二进制与产物契约缺口；不能画成已经验证的一条连续实线。

关键边界：

- 实际 Rust Metal host 实现在 `examples/tile_rsqrt_metal/src/main.rs:21-123`，直接使用 `metal` crate，不通过 `tile_hal`。关联仓 HAL 也没有 Metal 设备实现。
- kernel 源参数是输入、输出两个指针；host 额外绑定长度 buffer。目标入口参数与原始 Rust 签名不能假定机械一一对应，生成签名与 host 参数绑定共同构成调用契约。
- 例子使用 raw-pointer tile API，不使用 `GmView`；领域对象关系图不是说每条实际链路都必须经过 GM view 包装。
- `build.rs:23-60` 设置 `ACLRS_METAL_RUN` 并查找产物。注释声称动态库会调 xcrun，但两仓公开 Rust 代码中未见这段实际工具调用。
- 运行时 `new_library_with_source` 调用有源码。该调用委托外部 Metal 框架编译并得到 library，不要求先持久化 `.metallib` 文件。
- 关联仓另有 Python 脚本对检入 `.metal` 调 xcrun，以及 Swift Metal benchmark harness。这是另外的工具/执行路径，不是上述 kernel-builder 接线，也不证明这些检入源码来自当前主仓生成器。
- 关联仓的 DSL、宏和 HAL 是本地副本；核对时与主仓对应源码内容相同，不是 Cargo 依赖主仓，也不保证未来同步。

因此该关联仓准确的现状描述是：包含 DSL/HAL 副本、kernel 构建接线、Metal 示例 host 程序及相关 benchmark 资产的目标集成仓库。它没有本地 MSL 代码生成器，也不是已实现 Metal HAL 的专门 crate。

## 6. 名称、登记和验证边界

- README/发布使用说明的 `metal`、`cuda`、`vulkan`，分别与公开登记的 `msl`、`gpu`、`spirv` 不同。不能假定所有组件共享别名解析。
- HAL 的 `from_codegen_path` 接受 `msl`，不接受 `metal`；因此环境变量字符串也不是抽象层次一致性的证明。
- HAL `ascend` 分支仅有 cfg 引用，本树无该实现文件，manifest 也未定义可启用的 ascend feature。可选的实际设备实现是 CUDA。
- `tile_codegen` 的 `emitters` 编译接线引用生成器源码；其中部分生成器引用 `crate::mlir_to_pto`，而其 crate root 未声明该模块。本文记录为源码接线疑点，未编译证实，不在本票修复。
- MSL 单测中有对缺席的 `deepseek_metal/templates` 的读取；测试覆盖不能直接解释为本 checkout 全部测试成功。
- `tile_spec` 的字符串标记断言、目标语言编译成功、硬件 launch 成功、数值等价及性能是不同强度的证据。README 的上机声明未在本次重现。

## 7. 尚未闭合的事实边界

1. 发布 rustc 编译器后端内部的 MIR→MLIR、选路与目标构建调用不可见。
2. 当前公开生成器源码与某个具体发布二进制是否同版本、是否使用相同接线，尚未证实。
3. 关联仓补齐了构建器调用和 Metal 示例 host 源码；但 builder 的 Ascend 形状目标参数/产物搜索与例子预期的 Metal 旁路产物是否能接通，未运行验证。目标 spec、外部动态库与具体发布版本的关系仍有缺口。
4. 公开规格对生成代码的检查不构成全部 DSL 操作跨目标数值等价证明。
5. 用户确认的是具有上述证据边界的现状模型，不是所有目标均端到端可用，也不是任何分层方案。后续职责调查、外部机制研究和架构选择仍按地图各自的前置关系进行。
