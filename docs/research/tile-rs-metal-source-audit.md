# tile-rs-metal 源码核对（固定 SHA）

服务于决策票：[建立 tile-rs 领域模型、全仓 crate 地图与 Metal 完整转换链路](https://github.com/Jopqior/tile-rs/issues/13)，所属地图：[多后端 emit 分层探索：先厘清领域模型，再形成整体判断](https://github.com/Jopqior/tile-rs/issues/12)。

研究资产分支：`research/metal-source-audit`；路径：`docs/research/tile-rs-metal-source-audit.md`。领域模型与已确认词汇另在 `wayfinder/12-emit-layering`，本分支不复制它们。

核对对象：公开仓库 [`yijunyu/tile-rs-metal`](https://github.com/yijunyu/tile-rs-metal) @ `23130eeba6df91c555dbc71135f836eddb92568d`（`23130ee`，提交说明：`attn_gqa: attend to the whole sequence, one simdgroup per query row`）。

对照主仓库：本工作树 `wayfinder/12-emit-layering` @ `1ee797f1f5834ac9ce581ce69e8b8f1775ece076`。只记接线事实，不做架构建议。

方法：只读 clone 到 `/tmp/tile-rs-metal`，按该 SHA checkout；未运行外部源码，未安装工具，未执行 `cargo`/`xcrun`/`gh release download`。预编译 `librustc_codegen_tile` 产物未打开。

源码 URL 一律用固定 SHA + 行号。术语按本次会话在 `wayfinder/12-emit-layering` 的 `CONTEXT.md` 中确认的用语：`rustc` 编译器后端、目标代码生成器、目标编译工具链、主机运行时、Metal 链路、Rust DSL。源码标识符（如 `BackendKind`、`TILERS_CODEGEN_PATH`）保留原名。

## 证据分级

| 标记 | 含义 |
|------|------|
| **源码事实** | 该 SHA 树内可读文本（Rust/TOML/脚本/工作流） |
| **README 声明** | README / USAGE / CI 注释 / 清单注释自称，未在同树被实现或与实现不一致 |
| **外部预编译不可见** | 依赖 `yijunyu/tile-rs` release 里的 `librustc_codegen_tile.{so,dylib}` 或该产物内部行为 |
| **未运行验证** | 逻辑上可从源码读出，但本核对未执行 |

---

## 1. README 自称 vs 源码角色

README 自称（**README 声明**）：本仓库是 Apple GPU (Metal) 目标的「host runtime + examples + builder」；`rustc` 编译器后端是共享库 `librustc_codegen_tile.so`，用 `TILERS_CODEGEN_PATH=metal` 选择；本仓库提供「Apple GPU (Metal) runtime glue + examples」。

来源：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/README.md#L1-L31>

同树源码能核对到的角色：

| README 表格路径 | 源码里实际是什么 |
|-----------------|------------------|
| `crates/tile_std` | 存在；Rust DSL 副本，与主仓库逐文件 SHA-256 相同（**源码事实**） |
| `crates/tile_kernel_builder` | 存在；构建期调用 `cargo` + `-Zcodegen-backend=`，产物搜索与目标三元组仍是 Ascend 形状（**源码事实**） |
| `crates/tile_hal` | 存在；多目标 HAL 特质 + `BackendKind::Msl` 枚举项；无 Metal 设备实现，无 `metal` crate 依赖；本仓库没有任何 crate 引用它（**源码事实**） |
| `examples/` | 仅 `examples/tile_rsqrt_metal`；主机侧用 `metal` crate 加载/提交，不经过 `tile_hal`（**源码事实**） |
| `rustc_codegen_tile` / `mlir_to_msl` | 本仓库无此源码树（**源码事实**）；README 写明不随源码发布（**README 声明** + **外部预编译不可见**） |

USAGE 另称「runs it via wgpu/Metal」：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/docs/USAGE.md#L52-L54>。例子 `Cargo.toml` 依赖是 `metal = "0.29"`，全树 Rust 源无 `wgpu`（**源码事实**）。wgpu 出现仅在 USAGE 与 `benchmarks/kernels/manifest.toml` 注释（**README 声明**）。

---

## 2. Crate 实际存在性

磁盘上 `crates/` 五个包（均有 `Cargo.toml`）：

| 包名 | 路径 | 本仓库内被谁依赖 |
|------|------|------------------|
| `tile_std` | `crates/tile_std` | `examples/tile_rsqrt_metal/kernels` |
| `tile_std_macros` | `crates/tile_std_macros` | `tile_std` |
| `tile_hal` | `crates/tile_hal` | **无人引用**（仅 README 点名） |
| `tile_kernel_builder` | `crates/tile_kernel_builder` | `examples/tile_rsqrt_metal` 的 **build-dependencies** |
| `tile_kernel_builder_config` | `crates/tile_kernel_builder_config` | `tile_kernel_builder` |

不存在的包（相对 README / 主仓库常见名字）：`rustc_codegen_tile`、`tile_codegen`、`tile_spec`、`tile_metal_py`。无 `davinci-huawei-none.json`。

根 `Cargo.toml`（**源码事实**）：

```toml
members = [
    "crates/*",
    "examples/*",
]

exclude = [
    "crates/*",
    "examples/tile_rsqrt_metal",
    "examples/tile_rsqrt_metal/kernels",
]
```

来源：[根 Cargo.toml](https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/Cargo.toml#L3-L13)。声明上 `members` 的 `crates/*` 又被 `exclude` 的 `crates/*` 排除，唯一 example 也被 exclude；实际 Cargo 成员解析及该配置是否可构建尚未运行验证。`Cargo.lock` 仍锁了上述五 crate 的传递依赖（`anyhow`/`cmake`/`libloading`/`serde` 等），**没有** `rustc_codegen_tile`、**没有** `metal`/`wgpu`（`metal` 只在被 exclude 的 example 里）。

`install.rs` 与 CI 仍执行 `cargo build --workspace --release`：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/install.rs#L82-L86>、<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/.github/workflows/ci.yml#L41-L42>。空 workspace 下该命令实际编了什么：**未运行验证**。

独立包（有自己的 `[workspace]` 或被 exclude）：

- `examples/tile_rsqrt_metal` + `examples/tile_rsqrt_metal/kernels`（kernels 自带 `[workspace]`）
- `benchmarks/model_qwen3_1p5b`（独立 Cargo 工程）

`tests/compiletest/ui/` 有 109 个 `*.rs`（git 跟踪），无 compiletest 驱动 crate / `Cargo.toml` / CI 步骤。`vec_add.rs` 头是 `// build-pass` + `tile_std::get_block_idx`，不是 Metal 主机代码。

---

## 3. DSL 副本

`tile_std` / `tile_std_macros` / `tile_hal` 的源文件与当前主仓库对应路径 **逐文件 SHA-256 相同**；三份 `Cargo.toml` 也 `diff` 相同（**源码事实**，本工作树对照）。

`tile_std` 仍是 `#![no_core]` 的 Rust DSL（`tile` / `buf` / `kernel_ops` / `pipeline`），不是独立 Tile IR 方言。Metal 例子 kernel 通过 `tile_std = { path = "../../../crates/tile_std" }` 引用这份本地副本。

`tile_kernel_builder` 与 `tile_kernel_builder_config` **不在**当前主仓库 git 树中。主仓库 `Cargo.lock` 仍有 `tile_kernel_builder` 条目（依赖还写了 `rustc_codegen_tile`），与当前源码树不一致；本核对只记该锁文件残留，不解释历史。

---

## 4. `tile_kernel_builder`：输入、输出、如何接到 `rustc` 编译器后端

公开 API（**源码事实**）：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/crates/tile_kernel_builder/src/kernel_builder/mod.rs#L33-L93>

- **输入**：`KernelBuilder::new(path)`。`path` 是目录 → 当作 Rust kernel crate；是文件 → 当作 C++，走 `cpp_builder`。
- **可选**：`codegen_path(...)` 写入子进程 `TILERS_CODEGEN_PATH`；`copy_to(...)` 复制最终文件。
- **输出**：`build() -> PathBuf`。Rust 路径由 `rs_builder::build_with_env` 决定。

`rs_builder` 实际动作（**源码事实**）：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/crates/tile_kernel_builder/src/kernel_builder/rs_builder.rs>

1. `find_rustc_codegen_tile()` 解析 `rustc` 编译器后端动态库路径：先 `TILERS_CODEGEN_SO`，否则 `ASCEND_RUSTC_CODEGEN_TILE_SO`，否则在 `DYLD_FALLBACK_LIBRARY_PATH`（macOS）或 `LD_LIBRARY_PATH` 里找 `{DLL_PREFIX}rustc_codegen_tile{DLL_SUFFIX}`；找不到则 `panic`（L157–177）。macOS 上后缀是 `.dylib`，不是 README 写的 `.so`。
2. 组 `CARGO_ENCODED_RUSTFLAGS`：`-Zcodegen-backend=<该动态库>`，外加 `register_tool(tile)`、`panic=abort`、`lto=off`（L25–31, L56–68）。
3. 在 kernel crate 目录执行 `cargo build --release --message-format=json-render-diagnostics --target=...`。
4. 目标三元组常量是 `davinci-huawei-none`（L10）。若向上找到 `davinci-huawei-none.json` 则 `--target=<json 绝对路径>`。本仓库 **没有** 该 JSON。
5. 若 `codegen_path` 有值，只给 **子 cargo** 设 `TILERS_CODEGEN_PATH`（L70–77）。不在此文件里调用 `xcrun`。
6. 从 cargo JSON 的 `compiler-artifact` 文件名同目录搜索后缀（L89–107, L183–223）：
   - 优先 `tile.so`
   - 否则 `tile.gen.cpp` → 交给 `cpp_builder`（CANN cmake → `libkernels.so`）
   - 否则 `tile.o`
   - 都没有 → `BuilderError::BuildFailed`

`cpp_builder` / `ops_builder` / `tile_kernel_builder_config` 全部是 Ascend/CANN（`bisheng`、`atc`、`ACLRS_CANN_PATH`、`DEFAULT_SOC_VERSION = "Ascend310P1"`）。`lib.rs` 再导出 `add_ascend_link_args`。

因此：builder **会**把 `rustc` 编译器后端动态库接到 `cargo`/`rustc` 的 `-Zcodegen-backend=`；**不会**在本仓库源码里编译 MSL、不会搜 `.metal`/`.metallib`。Metal 例子把 `codegen_path("metal")` 传进去，只改变环境变量；产物搜索仍是 `tile.so` / `tile.gen.cpp` / `tile.o`。该路径在 Metal 链路上能否成功：**未运行验证**，且依赖 **外部预编译不可见** 的动态库是否写出这些后缀或旁路文件。

---

## 5. MSL 编译、metallib 加载、Metal 提交：HAL 还是 example

### 5.1 `tile_hal`：没有 Metal 主机运行时实现

<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/crates/tile_hal/src/backend.rs#L4-L111>

- `BackendKind::Msl` 注释为 Apple GPU via Metal Shading Language。
- `from_codegen_path` 只认 `"msl"`，**不认** `"metal"`。`"metal"` 会走 `None` → `from_env()` 报 `BackendNotAvailable`。
- `codegen_path()` 对 `Msl` 返回 `"msl"`。
- `BackendSelector::device` 仅在 feature `ascend` / `cuda` 下构造设备；`Msl` 落入 `other => BackendNotAvailable`。
- 无 `metal.rs` 模块。`Cargo.toml` 只有可选 `cuda = ["libloading"]`，默认无 feature；`lib.rs` 有 `#[cfg(feature = "ascend")]` 但 **没有** `ascend` feature 定义。
- `cuda.rs` 加载 `libcuda.so`（Linux CUDA），与 Metal 无关。

全仓库 `tile_hal` 引用：README 一行 + 自身源码。例子不依赖它。

### 5.2 本仓库 Rust 源里的 `xcrun`

`rg xcrun --glob '*.rs'`：只出现在 **注释**（`examples/tile_rsqrt_metal/{build.rs,kernels/src/lib.rs,src/main.rs}`）。没有任何 `Command::new("xcrun")`。

注释声称：`ACLRS_METAL_RUN=1` 时 **`rustc` 编译器后端**（本仓库未包含的 `rustc_codegen_tile`）会跑 `xcrun metal` + `xcrun metallib`。这是 **README/注释声明** + **外部预编译不可见**。当前主仓库 `crates/rustc_codegen_tile/` 只有 `mlir_to_*.rs` / `mlir_parse.rs`，无 `compile_via_metal` 实现；`tile_codegen` 把 `compile_via_metal` 写在「今日 host 的 if/else」的 **ignore 示例注释**里（`crates/tile_codegen/src/registry.rs` L1–14），并写明目标编译工具链留在 host、不在 `CodegenTarget::emit`。

### 5.3 真正出现 Metal 加载/提交的源码位置

**A. 例子主机程序（Rust，`metal` crate）**：本次定位到的 Rust Metal dispatch 实现：

<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/examples/tile_rsqrt_metal/src/main.rs#L21-L96>

- `Device::system_default()`
- 优先 `new_library_with_file(.metallib)`，否则读 `.metal` 再 `new_library_with_source`（运行时走 Metal 框架编译，不是 `xcrun`）
- `get_function("rsqrt")` → compute pipeline → `dispatch_thread_groups` → `wait_until_completed`

**B. Python 脚本** `benchmarks/metal/bench_ascendrs_msl.py`：对已检入的 `kernels_ascendrs/*.metal` 调 `xcrun metal` / `xcrun metallib`（目标编译工具链调用在脚本里，不在 HAL）。同文件写明设备跑是「Tier 2, future work」，并提到主仓库 `crates/tile_metal_py/`（**该路径不在 tile-rs-metal 树，也不在当前主仓库 git 跟踪文件中**）。

**C. Swift harness** `benchmarks/apple-mojo-vs-msl/harness/bench_metal_gpu.swift`：读已检入的 `kernels/msl/*.metal`，用 Metal.framework 提交。与 `tile_hal` / `tile_kernel_builder` 无编译依赖。

结论（接线）：metallib 加载与 Metal 提交的 **可执行 Rust 实现在 example，不在 HAL**。MSL→AIR→metallib 的 `xcrun` 调用 **不在** 本仓库任何 crate 的可执行 Rust 里；例子只设 `ACLRS_METAL_RUN=1` 并假定外部 `rustc` 编译器后端会做。Python/Swift 是另一条对着检入 `.metal` 文本的旁路。

---

## 6. 一条完整例子链路：`tile_rsqrt_metal`

以下是源码写出的调用关系，不是一次实测。

```
examples/tile_rsqrt_metal/kernels/src/lib.rs
  Rust DSL: #[tile_std::tile_kernel] fn rsqrt(*const f32, *mut f32)
  tile_load_f32 / tile_rsqrt_f32 / tile_store_f32  (ROWS=1, COLS=256)
        │
        │ kernels/Cargo.toml: crate-type = ["cdylib", "lib"]
        │ 注释：rustc_codegen_tile 只在 LINK（Cdylib/Dylib/Executable）跑
        │       MLIR→MSL emit + compile_via_metal
        ▼
examples/tile_rsqrt_metal/build.rs
  若未设则 set ACLRS_METAL_RUN=1
  KernelBuilder::new("kernels").codegen_path("metal").build()
        │
        ▼
crates/tile_kernel_builder ... rs_builder
  找 librustc_codegen_tile.{dylib,so}
  cargo + -Zcodegen-backend=... --target davinci-huawei-none
  TILERS_CODEGEN_PATH=metal（仅子进程）
  在 artifact 目录找 tile.so / tile.gen.cpp / tile.o
        │
        │  （注释期望旁路文件：<stem>.metal / .metallib / .metal_runner.sh）
        │  例子再在 built_path.parent() 搜 .metal / .metallib
        │  cargo:rustc-env TILE_RSQRT_METAL / TILE_RSQRT_METALLIB
        ▼
examples/tile_rsqrt_metal/src/main.rs
  metal crate：加载 metallib 或编译 .metal 源
  绑定 buffer(0/1/2)，dispatch 1×256，与 CPU rsqrt 比，max abs err < 1e-4
```

Kernel 源：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/examples/tile_rsqrt_metal/kernels/src/lib.rs#L35-L41>

build.rs：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/examples/tile_rsqrt_metal/build.rs#L23-L60>

与文档样本的差异（**README 声明** vs **源码事实**）：

- USAGE 样本是 `#[tile_kernel]` + `GmViewMut<32,64,f32>`；实际 kernel 是原始指针 + `#[tile_std::tile_kernel]`，形状 `1×256`。
- build.rs 注释写「`.acl.o` placeholder」；builder 搜索的是 `tile.o` / `tile.so`，不是 `.acl.o`。
- `find_with_suffix` 注释写会跳过 `.metal_runner.sh`；实现是 `ends_with(".metal")`，`.metal_runner.sh` 恰好不匹配，但没有单独过滤。
- 主机 crate **不**依赖 `tile_hal` / `tile_std`；只 build-dep `tile_kernel_builder`，runtime-dep `metal`。

CI `metal-smoke`：`cargo run --release -p tile_rsqrt_metal || true`，`continue-on-error: true`，且 **不下载** codegen 动态库：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/.github/workflows/ci.yml#L44-L59>。因根 workspace exclude 了该包，根目录 `-p tile_rsqrt_metal` 能否解析：**未运行验证**。CI 注释仍写「wgpu/Metal」。

---

## 7. `metal` / `msl` 命名，以及版本 / ABI 选择

### 7.1 选择键（同一 SHA 内不一致）

| 位置 | 用的字符串 | 分级 |
|------|------------|------|
| README / USAGE / `install.rs` / CI `env.TILERS_CODEGEN_PATH` / example `codegen_path("metal")` / kernel 注释 | `metal` | 声明或源码 |
| `tile_hal::BackendKind::from_codegen_path` | 只认 `msl` | 源码事实 |
| 当前主仓库 `tile_codegen` 注册 | `EmitterTarget::new("msl", "metal", convert_mlir_to_msl)`：`name()`=`msl`，文件扩展名=`metal` | 源码事实 |
| 当前主仓库 `tile_spec` cucumber | `"msl" => convert_mlir_to_msl` | 源码事实 |
| 当前主仓库 README 目标表 | `TILERS_CODEGEN_PATH` 列写 `metal` | README 声明 |
| 当前主仓库 `mlir_to_msl.rs` 模块注释 | `ACLRS_CODEGEN_PATH=metal`；`xcrun metal` via build.rs | 注释声明；该文件本身只 emit MSL 文本 |

`rs_builder` 把例子给的 `"metal"` **原样**写入子进程 `TILERS_CODEGEN_PATH`。预编译 `rustc` 编译器后端认 `metal` 还是 `msl`：**外部预编译不可见**。若只认 `msl`（与当前主仓库 `tile_codegen`/`tile_spec`/`tile_hal` 一致），则例子传入的 `metal` 对不上。若认 `metal`（与 README/CI/例子一致），则 HAL/`tile_codegen` 的 `msl` 对不上。本核对未检查该动态库，不能判定这些名字或别名在具体发布产物中如何生效。

文件扩展名 `.metal` 与目标语言 MSL：例子与检入产物（`benchmarks/metal/kernels_ascendrs/*.metal`、`benchmarks/apple-mojo-vs-msl/kernels/msl/*.metal`）用 `.metal` 文本。检入文件头写 `Generated by ... mlir_to_msl` 和 `xcrun metal`/`metallib` 编译行——这是检入文本，不是本仓库生成器。

### 7.2 版本与 ABI

本仓库 `rust-toolchain.toml`：`channel = "nightly-2025-08-04"`，组件含 `rustc-dev` / `llvm-tools` / `rust-src`。与当前主仓库 `rust-toolchain.toml` 相同。`install.rs` 常量 `NIGHTLY` 同值，注释写「codegen backend is ABI-pinned to it」。

预编译获取（`install.rs --codegen`，**源码事实**）：<https://github.com/yijunyu/tile-rs-metal/blob/23130eeba6df91c555dbc71135f836eddb92568d/install.rs#L90-L108>

- 默认 tag：`TILERS_CODEGEN_TAG` 或 `v0.0.1-pre+nightly-2025-08-04`
- 资产名：`tile-rs-codegen-{triple}.tar.gz`，`triple` 来自 `uname -m` → `aarch64-apple-darwin` 或 `x86_64-apple-darwin`
- `gh release download --repo yijunyu/tile-rs` 到目录 `tile-rs-codegen`
- **没有** `tar` 解压；成功日志却写 “Extracted”
- **没有** 设置 `TILERS_CODEGEN_SO`
- 无 `gh` 则只 warn
- 日志要用户把 `TILERS_CODEGEN_SO` 指到 `librustc_codegen_tile.dylib`

当前主仓库 `scripts/install.sh`：默认 tag `v0.0.1+nightly-2025-08-04`（**无** `-pre`）；会下载并 `tar xzf`；注释区分 macOS `.dylib` / Linux `.so`。两边默认 tag **不同**。产物是否存在、ABI 是否匹配该 nightly：**外部预编译不可见**，**未运行验证**。

README 全程写 `librustc_codegen_tile.so`；macOS 安装路径写 `.dylib`。`rs_builder` 用 `env::consts::DLL_SUFFIX`，在 Darwin 上是 `.dylib`。

`install.rs` 拒绝非 Darwin；非 arm64 只 warn。CI 固定 `macos-14` + `TILERS_CODEGEN_PATH: metal`。

---

## 8. 与当前主仓库边界（只记有/无）

对照主仓库 `1ee797f1`（`wayfinder/12-emit-layering`），同一概念在两边的落点：

| 环节 | tile-rs-metal @ 23130ee | 当前主仓库 @ 1ee797f1 |
|------|-------------------------|------------------------|
| Rust DSL | 有 `tile_std` + `tile_std_macros`（字节相同副本） | 有（同内容） |
| `tile_hal` 特质 + `BackendKind::Msl` 认 `msl` | 有（字节相同）；无 Metal 设备 | 有（同内容）；同样无 Metal 设备 |
| kernel 构建器（`-Zcodegen-backend=`、搜 `tile.so`） | 有 `tile_kernel_builder` | **无** 该 crate 源码（lock 仍残留名字） |
| 目标代码生成器 `convert_mlir_to_msl` | **无** | 有 `crates/rustc_codegen_tile/src/mlir_to_msl.rs`（无该 crate 的 `Cargo.toml`/`lib.rs`） |
| `CodegenTarget` 注册名 | 无 `tile_codegen` | `"msl"` + 扩展名 `"metal"` |
| 可链接的 `rustc` 编译器后端（`.so`/`.dylib` 工程） | 本树无源；README 指向 tile-rs release | 本树无完整 crate；`scripts/install.sh` 拉 release |
| 目标编译工具链（`xcrun metal`/`metallib`） | 可执行 Rust 无调用；例子注释推给外部动态库；Python 脚本有 | 当前树无 `compile_via_metal` 实现（仅注释） |
| Metal 主机提交 | Rust 例子 `examples/tile_rsqrt_metal/src/main.rs`（`metal` crate）；另有 Swift benchmark harness | **无** `examples/` 目录 |
| 端到端例子 | 有 `tile_rsqrt_metal`（exclude 出 workspace） | 根 `Cargo.toml` 注释仍提到 `examples/tile_rsqrt_metal`，工作树无该目录 |
| 检入的 `.metal` 文本 | `benchmarks/metal/kernels_ascendrs/`、`benchmarks/apple-mojo-vs-msl/kernels/msl/` | 无对应路径 |
| ABI nightly | `nightly-2025-08-04` | 同 |
| 默认 codegen tag | `v0.0.1-pre+nightly-2025-08-04` | `v0.0.1+nightly-2025-08-04` |

主仓库 README 仍把 Apple GPU 的 `TILERS_CODEGEN_PATH` 写成 `metal`，与同树 `tile_codegen`/`tile_hal`/`tile_spec` 的 `msl` 并列存在——这是主仓库自己的文档/源码分裂，不是 tile-rs-metal 独有。

主仓库 `Cargo.toml` 对已删除 example 的注释仍描述链路：`tile_std` → `rustc_codegen_tile`（`TILERS_CODEGEN_PATH=metal`）→ `.metal`/`.metallib` → Mac Metal GPU。该注释是主仓库文档，不是 metal 仓库源码。

---

## 9. 缺口（本核对停在这里）

1. **`rustc` 编译器后端产物内部**：是否实现 `compile_via_metal`、是否读 `ACLRS_METAL_RUN`、是否写 `.metal`/`.metallib`、选择键是 `metal` 还是 `msl`、链接阶段是否仅对 cdylib 触发——**外部预编译不可见**。当前主仓库发射器源码不能代替该动态库。
2. **`KernelBuilder` 在 Metal 路径上的返回值**：源码只认 `tile.so`/`tile.gen.cpp`/`tile.o`，例子再在父目录找 `.metal`。缺目标 JSON、缺动态库时 `build()` 会 `panic`/`BuildFailed`。未跑，不能证实例子 build.rs 的 `expect("Metal codegen did not produce a .metal source file")` 是否能过。
3. **空 workspace 与 CI**：`cargo build --workspace`、`cargo run -p tile_rsqrt_metal`、`cargo fmt --all` 的实际成员集未验证。CI smoke 带 `|| true`。
4. **`install.rs --codegen`**：默认 tag 与主仓库不同；下载后不解压、不设 `TILERS_CODEGEN_SO`。未调用 `gh`，release 是否仍有 `v0.0.1-pre+...`：**外部预编译不可见**。
5. **`tile_hal` 作为「this target's backend」**：源码是通用 HAL 副本，Metal 臂不可用，且无消费者。README 角色与源码职责不一致；不一致已记录，不在此改词。
6. **wgpu**：文档/清单注释有，Cargo/Rust 无。
7. **compiletest/ui 与 model/micro bench 脚本**：文件在树里；无接到 `tile_kernel_builder` 或 Metal 提交的驱动被本核对执行。
8. **检入 `.metal` 与现发射器是否同源**：文件头自称 `mlir_to_msl`，未做文本 diff 对照主仓库 `mlir_to_msl.rs` 的当前 emit 格式。
9. **未运行验证**覆盖：任何 cargo/xcrun/GPU 提交、数值误差、动态库能否加载进 `nightly-2025-08-04`。

以上缺口都不改变第 5–6 节的接线事实：本 SHA 里，Metal 链路的主机加载与提交写在 example；HAL 不承担；MSL 的 `xcrun` 编译不在本仓库 crate 的可执行 Rust 中。
