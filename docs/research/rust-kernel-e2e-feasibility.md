# Rust kernel → MLIR → 当前源码 MSL → Metal GPU → 参考比较：可行性核查

- 调研日期：**2026-09-29 UTC**。范围：[本地地图 #42](https://github.com/Jopqior/tile-rs/issues/42) 的**新要求**（取代该票正文中“Rust → MLIR 超出范围”的旧边界）；只研究，不更改票据或实现。
- 固定源码：规划 checkout [`2924af350f8843738f276b9a8695b0416a2d3d49`](https://github.com/Jopqior/tile-rs/commit/2924af350f8843738f276b9a8695b0416a2d3d49)（含固定上游 `e8d1acd`）；本地 `main` / `upstream/main` [`6d59de3018ff85dacd8a8b0a98215f7063eff211`](https://github.com/yijunyu/tile-rs/commit/6d59de3018ff85dacd8a8b0a98215f7063eff211)。引用后者是分析已有接口，**不等于**让规划分支跟进它。
- 初稿方法（下文 §1–5）：`gh` 查公开仓库、Release API、工作流及运行记录；对两个固定提交只读源码。初稿未下载/运行 Release 或 macOS CI。**后来已获授权实施隔离 macOS 实验，实测结论见 §6–7；§1–5 的“尚未验证”及“不得启动 workflow”仅描述当时状态，已由 §6–7 取代。**始终没有访问私有源、运行 GPU 或合并正式 CI。

> **最新补充（§7；§6 仍是固定 DSL 对照）：** 已实际测试公开 Release tag `c3c8ec0` 的 `tile_std` + 同 tag 源码宏，以及该 `tile_std` + Release 包内预编译宏。两种组合均使 Release 前端完成 Cargo release 编译、真实导出相同的 kernel MLIR；固定 `2924af3` 的 registry emitter 对两份原始 MLIR 均拒绝，因而均无当前源码 MSL。公开 tag 的源码是有出处的候选，**不能证明**与二进制私有构建同版。详情、逐阶段日志及产物见 §7；§6 中 `2924af3` DSL 的内置 MSL 失败不能再概括成“Release 对所有 DSL 均无法编译”。

## 结论（初稿静态判断，最终以 §6 为准）

**选用固定预构建前端作为候选，而不是“从当前公开源码构建 Rust→MLIR 前端”**：两份公开 checkout 都有 `tile_std`、宏、MLIR parser/Metal emitter、`tile_codegen`，但没有可构建的 `rustc_codegen_tile` 前端 crate（例如 `src/lib.rs` / 其 manifest）；上游公开工作流明确从私有代码源取前端。公开 Release 确实有可下载的 **Apple Silicon** 预构建前端，建议以 **0.0.2** 为兼容性候选（0.0.1 对标准 rustc rlib/core 有已记载缺陷），但尚未验证它与固定 `2924af3` 或 `6d59de3` 的 DSL/MLIR 契约。[公开 README 架构](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/README.md)、[私有 checkout 的公开工作流](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml)、[公开树的文件列表](https://github.com/yijunyu/tile-rs/tree/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/rustc_codegen_tile)、[0.0.2 Release](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04)。

**二进制不是“只含前端”的可替换插件**：官方说明它直接选择 `TILERS_CODEGEN_PATH=metal` 并产出 Metal；但 `main` 的公开 CLI 源码又说明其编译过程**同时写出 `.mlir` 和 `.tile.*`**，CLI 的 `lower_rs::lower()` 会分别读回两个文件。故存在具体的可研究接缝：**仅消费预构建前端产出的 MLIR，明确丢弃其内置 Metal，再以被测试提交的公开 `tile_codegen`（`emitters` 特性）或其 `convert_mlir_to_msl` 处理 MLIR**。不证明这个跨版本接缝可运行；也不能把“使用二进制自带 Metal”冒充“当前源码 emitter 受测”。[CLI 源码](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/lower_rs.rs)、[CLI route 行为](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/bin/tile.rs)、[registry/emitters](https://github.com/Jopqior/tile-rs/tree/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src)。

## 1. 公开性、真实 Release 与 ABI：把“计划”与“已发布”分开

- `gh api repos/yijunyu/tile-rs`：公开仓库 (`private=false`)；两个 checkout 的 `crates/rustc_codegen_tile` 为 MLIR→目标源码文件，缺少完整可构建前端。公开 [codegen-release.yml @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml) 注释称代码源私有，以 `actions/checkout@v5` + 只读 `PRIVATE_SOURCE_TOKEN` checkout；具体 repo 名用 `PRIVATE_SOURCE_REPO` secret，**不得从公开文件推定私有仓库此刻可被我们读取**。`2924af3` 的同一工作流仍硬编码 `yijunyu/ascend-rs-priv`，这只是**当时公开配置**，不是可访问性测试。[旧工作流 @ 2924af3](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/.github/workflows/codegen-release.yml)。未访问私有 repo/API，也未探测 secret。
- `gh release list` 和 `gh api repos/yijunyu/tile-rs/releases`：截至调研日**恰有两个公开 prerelease**，均各仅一份 `tile-rs-codegen-aarch64-apple-darwin.tar.gz`：[`v0.0.1+nightly-2025-08-04`](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.1%2Bnightly-2025-08-04)，2026-06-21，101,724,767 bytes，sha256 `231f85f11622ae38eeb724d4804432a7a394f3c2f371076de8bd57292c3d63b4`；[`v0.0.2+nightly-2025-08-04`](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04)，2026-09-06，103,449,511 bytes，sha256 `70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891`。sha256 为**Release API 所列 digest**，不是本次本地下载验证。无已发布 Linux/x86 macOS 资产；别将 [工作流计划的 `macos-14` arm64 与 `ubuntu-latest` x86_64 双平台矩阵](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml) 当成真实资产。
- 两份 [Release notes](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04) 声明 LLVM/MLIR 20 通过 `@loader_path` 随包，后端 dylib 依赖确切 **`nightly-2025-08-04`**（rustc `1.91.0-nightly`, `f34ba774c`）的 `librustc_driver` ABI，需 `rustc-dev`、`llvm-tools`、`rust-src`；0.0.2 不改变 ABI，只修复 0.0.1 把标准 `ar` 格式 rustc `rlib` 当作 `tar` 解读、无法依赖 `core` 的问题。Release notes 说其发布前曾用真实 CLI 降低 `#![no_std]` softmax 并用 `xcrun metal -c` 编译；这是**上游的声明，不是本次实测**。本地 `main` 的 [provision.toml @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/assets/provision.toml) 已固定 **0.0.2 + 同一 checksum**，且不为没有资产的平台制造条目。
- [安装脚本 @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/scripts/install.sh) **默认仍指向 0.0.1**，可用 `TILERS_CODEGEN_TAG` 改写；因此不能把 README 的一行安装命令视为自动安装修复版。`tile` CLI provision 是不同的、固定 0.0.2 的安装通道。上游 [install-smoke.yml @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/install-smoke.yml) 在 `macos-14` 核验脚本安装、配置、动态库与依赖，**并不执行 Rust kernel→MLIR→GPU**；[`gh run` 的成功记录](https://github.com/yijunyu/tile-rs/actions/workflows/install-smoke.yml) 不能当全链验收。公开 [Codegen Release workflow 页面](https://github.com/yijunyu/tile-rs/actions/workflows/codegen-release.yml) 所见最近两条 push 运行状态为 failure；既有真实 Release 资产与工作流最新运行失败并不矛盾，不能据工作流矩阵推断后续发布成功。

## 2. 确切输出与接缝：已有证据 / 尚未证实

- [Release 0.0.2 文本](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04) 与 [工作流生成的 USAGE.md 模板 @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml)：模板声称包内有 `lib/librustc_codegen_tile.dylib`、`lib/libtile_std_macros.dylib`、`.cargo/config.toml`、`USAGE.md`，**但 §6 实际验包发现 `.cargo/config.toml` 缺失**；工作流所写模板里的配置使用 `-Zcodegen-backend=lib/librustc_codegen_tile.dylib`、`-Zcrate-attr=feature(register_tool)`、`-Zcrate-attr=register_tool(tile)`、`-Cpanic=abort`、`-Clto=off`；`TILERS_CODEGEN_SO` 指定 dylib，`TILERS_CODEGEN_PATH=metal` 选择内置 Metal（**不是**本地 registry 的 `msl` 名）。文档例子 `cargo +nightly-2025-08-04 build --release` 要求 kernel crate `crate-type=["cdylib","lib"]`。
- **导出 MLIR 的确切公开接口**：[`lower_rs.rs @ 6d59de3`](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/lower_rs.rs) 建临时 kernel crate，依赖 `tile_std` 路径；写入上述 rustflags；`cargo +nightly-2025-08-04 build --target <host-triple>` 并传 `TILERS_CODEGEN_SO=<已安装 dylib>`、`TILERS_CODEGEN_PATH=metal`；在 `target/<triple>/debug/deps/` 读取 `*.mlir` 与 `*.tile.*`，返回 `Lowered { mlir: String, target_source: Option<String> }`；没有 `*.mlir` **报错**，不是悄悄合格。`tile_cli` 的 CLI 处理器在有内置目标源码时优先复用它、而不是用本地源码重发；`--keep` 会留中间结果。官方 [CLI tutorial](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/docs/cli/tutorial.md) 给出 `tile k.rs -t msl -o k.metal -k`。这证明**公开调用方期望** MLIR 文件存在，不证明此次未运行的 0.0.2 与两个目标提交的所有 kernel 都确实产出相容的 MLIR；直接用 `-t mlir` 不能从本调查推定可用，`lower()` 需要一个实际 codegen target。[CLI 映射：`msl`→`metal`](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/forms.rs)。
- **当前源码消费接口**：`2924af3` 的 [`CodegenTarget::emit(&str, &EmitOpts)`](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src/target.rs)、[`TargetRegistry::with_builtin().select("msl")`](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src/registry.rs)、[`emitters` feature 所挂的 `convert_mlir_to_msl`](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src/emitters.rs)；`main` 同一路径 [@ 6d59de3](https://github.com/yijunyu/tile-rs/tree/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_codegen/src)。[当前测试入口 @ 2924af3](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/metal_correctness/src/main.rs) **直接调 emitter**，不经过 registry，且引用已删的 `mlir_to_pto.rs`，需后续迁移；它的 [case MLIR](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/metal_correctness/src/cases.rs) 是手写，尚无 Rust kernel 对应。`main` 的 [`tile_cli::emit` @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/emit.rs) 可以读 MLIR 直接调用公开 MSL emitter，但它不替代“受测提交的源码”归属检查。
- **未证实的关键点**：预构建前端吐出的真实 MLIR 是否被 `2924af3` / `6d59de3` 的 parser/emitter 接受且语义相同；`0.0.2` 是否能编译这两个提交里各自的 `tile_std`、`tile_std_macros`（源码存在版本差异）；公开发布包确切内置 emitter 来自哪一个**私有构建提交**以及是否匹配任何公开快照；没有独立的只执行 MIR→MLIR、不执行内置 emitter 的发布后端开关/命令证据。建议未来合法实验保存原始 `.mlir`、生成 `.metal`、完整工具链/源提交标识并拒绝无输出，但本次没有做。

## 3. Rust 源码输入依赖与免费 macos-15

- 固定 `2924af3` 的 [kernel 实例](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/examples/tile_softmax/kernels/src/lib.rs) + [manifest](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/examples/tile_softmax/kernels/Cargo.toml)：`#![feature(no_core)]`, `#![no_std]`, `#![no_core]`, `tile_std` 的路径依赖、`#[tile_std::tile_kernel]`、`crate-type=["cdylib","lib"]`。[`tile_std` manifest](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_std/Cargo.toml) 再依赖 [`tile_std_macros` proc-macro（syn/quote/proc-macro2）](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_std_macros/Cargo.toml)；[宏实现](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_std_macros/src/lib.rs) 把 GmView 参数改写为原始指针，注入 `#[unsafe(no_mangle)]` 与 `#[tile::kernel]`；宏和 backend 必须认同这些标记/布局。[root nightly pin](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/rust-toolchain.toml) 与 Release ABI 日期一致，但**日期相同不保证跨私有版本 DSL/MLIR 相容**。
- `main` 的 [`lower_rs.rs` @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/lower_rs.rs) 特别要求 `--target <triple>` 以免 backend 污染宿主 build-script/proc-macro，并拒绝外部 `RUSTFLAGS` 覆盖 `build.rustflags`；需要本地 `tile_std`（可用 `TILE_STD_PATH` 指向 checkout）及安装的 backend。单独的 [`main` CLI workflow](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/tile-cli.yml) 用 **stable** 构建 `tile_cli` / 跑 fixture MLIR→MSL，不能推定 Rust→MLIR 路由已在 CI 通过。`main` CLI 使用二进制 + `main` 的 DSL 不意味着自动测试 `2924af3` 的 emitter。
- 托管 runner：GitHub [标准 runner 文档](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) 将 `macos-15` 列为 Apple Silicon arm64 标准 runner；[GitHub 计费文档](https://docs.github.com/en/billing/concepts/product-billing/github-actions) 称**公开仓库标准 hosted runner 免费**，大 runner 收费；`gh api repos/Jopqior/tile-rs` 与上游均返回 public。固定分支已有 [metal-correctness.yml @ 2924af3](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/.github/workflows/metal-correctness.yml) 用 `macos-15`、真实 Metal/PyTorch 比较；[此前成功运行](https://github.com/Jopqior/tile-rs/actions/workflows/metal-correctness.yml) 只支持**既有 MLIR 起点、相应早期提交**，不支持新链的结论。加上 rustc-dev / rust-src、发布包获取/校验、编译前端导出与当前源码 emitter 后，资源、Metal toolchain 可用性、PyTorch 安装、运行时设备及 45 分钟超时都仍需在真实 runner 测量；不能由标签、上游发布说明或前述 CI 成功自动断言可行。缺设备/跳过/只语法编译不能算 GPU 数值正确性。[当前测试 README 的状态/边界](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/metal_correctness/README.md)。

## 4. 路径和验证归属

| 候选路径（均未在本研究运行） | 真正验证的组件 | 尚有阻塞/不能宣称 |
| --- | --- | --- |
| **优先研究：固定 0.0.2 的 macOS arm64 前端 + 目标提交的 `tile_std`/宏 → 保存 `.mlir` → 目标提交的 `tile_codegen` `msl`/MSL emitter → Metal GPU → CPU 参考** | Rust kernel 源码、**固定二进制版前端**、所选 `2924af3` **或** `6d59de3` 的公开 emitter、宿主执行及比较；须在记录中分开标注版本 | 二进制仍可能同时生成并丢弃其自己的 `.metal`；真实 MLIR 契约与 DSL 兼容性未知；现有入口要脱离手写 MLIR 和失效 import；runner 实测未做 |
| `main` 的 `tile k.rs -t msl -o k.metal -k` / 原生 release workflow 原样使用内置源码 | Rust + 发布后端自身的 MLIR→**发布后端自带 emitter**，如后续执行 GPU 才覆盖该二进制整链 | **不**测试规划 checkout/current-source emitter；`--keep` 暴露 MLIR 不会自动替换 emitter；不能当 #42 的当前源码验收 |
| 从已授权的私有源码构建同一前端并将它与当前 emitter 集成 | 可能可验证同源码编译的前端和当前 emitter | 当前公开仓库无此前端，公开免费 workflow 依赖私有源和秘密；需权利方授权、固定私有提交和可重现构建；本调查没有权限/可访问性证据，不建议靠猜测取得源码 |
| 现有 `metal_correctness` 手写 MLIR 起点 | 公开 MLIR→MSL emitter、Metal/参考比较（修正其固定分支兼容性后） | **不覆盖 Rust→MLIR**，不能满足用户新要求 |

**建议的下一道研究/实施前门槛**：在授权、可信来源和人工审查下载前提下，固定 0.0.2 digest 与公开代码提交；以小 Rust add kernel 测出 MLIR 文件与内置 Metal 源、单独将同一 MLIR 送进该提交的 `msl` registry，再按现有 GPU/CPU 参考路径执行、报告精确每一步结果。如果无法拿到兼容的 MLIR 或无法证明使用的是当前源码 emitter，应如实写明阻塞，不能降级为使用内置 emitter 或手写 MLIR 却称全链通过。

## 5. 授权后的兼容性实验：环境门槛阻塞（2026-09-29 UTC）

本节是**实验执行记录，不是兼容性通过记录**。先检查本机与已有、明确供本项目使用的 macOS 执行环境；本机是 Linux x86_64，公开 Release 唯一资产是 macOS arm64。项目已有 `macos-15` / `macos-14` GitHub-hosted 工作流配置，但调用远程 workflow 已被本次授权明确排除；未找到可直接使用的、明确为本项目配置的 macOS 主机。故在环境门槛停止，**没有下载/解包/运行 Release，没有运行 Rust 前端或 emitter**。没有把静态示例、发布方自测或包内 Metal 当成本次结果；也未改 tracker、CI、主工作目录，未发起远程执行。

### 执行环境、命令及原样关键输出

检查时间 `2026-09-29T16:38:40Z`；工作区为既有独立 worktree `/tmp/tile-rs-rust-kernel-e2e-feasibility`，分支 `research/rust-kernel-e2e-feasibility`（检查时 HEAD `5c57e005462727d7a7aea334a342e56b479bbe97`）。固定受测**公开源码**为 `2924af350f8843738f276b9a8695b0416a2d3d49`；检查 `git diff --stat 2924af3 HEAD -- crates/tile_std crates/tile_std_macros crates/tile_codegen` 输出为空，即这三部分在报告 worktree 与固定提交间无差异。这不改变固定基线。主工作目录只读检查结果 `?? Cargo.lock`，保留未跟踪文件。

```text
$ uname -s -m
Linux x86_64
$ for t in git gh rustc cargo curl shasum sha256sum xcrun xcodebuild ssh; do command -v "$t" || true; done
/usr/bin/git
/usr/bin/gh
/home/whh/.cargo/bin/rustc
/home/whh/.cargo/bin/cargo
/usr/bin/curl
/usr/bin/shasum
/usr/bin/sha256sum
/usr/bin/ssh
$ rustc +nightly-2025-08-04 -vV
rustc 1.91.0-nightly (f34ba774c 2025-08-03)
binary: rustc
commit-hash: f34ba774c78ea32b7c40598b8ad23e75cdac42a6
commit-date: 2025-08-03
host: x86_64-unknown-linux-gnu
release: 1.91.0-nightly
LLVM version: 20.1.8
$ cargo +nightly-2025-08-04 -V
cargo 1.91.0-nightly (840b83a10 2025-07-30)
$ command -v xcrun || true; command -v xcodebuild || true
# stdout/stderr: empty
```

现有配置仅作本地文件核查：固定提交的 [Metal correctness workflow](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/.github/workflows/metal-correctness.yml) 采用 GitHub-hosted `macos-15`，[install-smoke workflow](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/.github/workflows/install-smoke.yml) 采用 `macos-14`；项目工作流中无 `self-hosted` 配置。检查本机 SSH **别名行**只有 `aisoft`、`github.com`；前者没有 macOS 或本项目用途的明确配置，未连接、未探测网络或私有访问。项目标识的 `TILE*`/`METAL*`/`MACOS*`/`RUNNER*`/`GH*` 环境变量名列表为空。已有 GitHub 托管 runner 并不等于当前获准启动它。

Release API 只读核查（**API digest，不是本机下载后 SHA256**）：

```text
$ gh api 'repos/yijunyu/tile-rs/releases/tags/v0.0.2%2Bnightly-2025-08-04' --jq '{tag_name,created_at,published_at,assets:[.assets[]|{name,size,digest,browser_download_url}]}'
{"assets":[{"browser_download_url":"https://github.com/yijunyu/tile-rs/releases/download/v0.0.2%2Bnightly-2025-08-04/tile-rs-codegen-aarch64-apple-darwin.tar.gz","digest":"sha256:70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891","name":"tile-rs-codegen-aarch64-apple-darwin.tar.gz","size":103449511}],"created_at":"2026-09-01T14:21:01Z","published_at":"2026-09-06T01:26:13Z","tag_name":"v0.0.2+nightly-2025-08-04"}
$ gh api 'repos/yijunyu/tile-rs/git/ref/tags/v0.0.2+nightly-2025-08-04' --jq '{ref,object:{sha:.object.sha,type:.object.type}}'
{"object":{"sha":"c3c8ec0169bd02757b51370ba9c6ec3b116b833d","type":"commit"},"ref":"refs/tags/v0.0.2+nightly-2025-08-04"}
# 两条命令 stderr: empty
```

[发布说明](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04) 只明确 0.0.2 的 rlib 修复所涉私有提交 `4c9e43f68`、`a1d1014ba`；[公开发布工作流](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml) 则从**私有源** checkout 编译后端。公开 Release tag 指向公开仓库的 `c3c8ec...`，**不能据此证明**该 tag 的 `tile_std`/宏来自实际二进制构建所用的私有源码，也不能把随包 `libtile_std_macros.dylib` 当作匹配的可替换**源码**。目前没有具出处的 Release 匹配 `tile_std`/宏源码版本可据以做对照；用户关于“必须同 Release”的假设仍待实验，不据此替换 `2924af3` 基线。

### 两个目标的实测状态与资产

| 目标 / 阶段 | 本次状态 | 原因 / 产物 |
| --- | --- | --- |
| (1) Release 前端 + `2924af3` `tile_std`/宏编译最小 Rust add kernel | **未运行：环境阻塞，非编译失败、非成功** | 本机 Linux x86_64；Release 只有 Apple Silicon macOS dylib；没有可在授权范围内启动的 macOS 运行环境。未生成或运行输入 crate。 |
| 前端实际 MLIR 导出 | **未运行，MLIR 缺失（未生成，不是已运行后未产出）** | 原始 `.mlir`：无；stdout/stderr：无（命令未执行）。 |
| (2) `2924af3` 公开 `tile_codegen` registry + `emitters` 的 `msl` 接受同一份真实 MLIR | **未运行，非 emitter 拒绝、非成功** | 上游真实 MLIR 不存在；当前源码生成 MSL：无；stdout/stderr：无。未使用随包/内置 MSL、手写 MLIR 或别的提交 emitter 代替。 |
| GPU 执行 / 数值正确性 | **不在本次要求内；未运行** | 无。 |

**没有可记录的实际 kernel 输入、MLIR、MSL、前端或 emitter stderr**；上面的环境/API 检查原样输出是本次全部执行记录。仓库的 `examples/bench_vec_add_rs/kernels/src/lib.rs` 与 `crates/tile_std/src/lib.rs`、`crates/tile_std_macros/src/lib.rs` 仅供后续设计输入使用，不是本次运行的 kernel。目标接口来自固定提交 [`tile_codegen/src/registry.rs`](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src/registry.rs) 与 [`emitters.rs`](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src/emitters.rs)，均只是静态核对。

### 若另行授权远程执行：最小实验方案（尚未执行）

1. 选择**已有且明确可由本项目使用**的 macOS arm64 环境，并单独授权其具体远程执行方式（例如在公开项目已有 `macos-15` runner 上由授权者运行一次性实验；本次**不得**自行触发/创建 workflow 或付费资源）。记录 `sw_vers`、`uname -m`、`rustc +nightly-2025-08-04 -vV`、`cargo -V`、`xcodebuild -version`，预检 `rustc-dev`、`llvm-tools`、`rust-src`，缺失时由授权者决定安装范围，不全局安装。
2. 在隔离的 `/tmp` 空间 checkout 固定 `2924af3`，建立最小 f32 add kernel crate，**只**以此提交的 `crates/tile_std` 路径依赖（由它依赖同提交宏），保存确切 `Cargo.toml`、`src/lib.rs`、sha256；不要直接接包内 macro dylib。通过 Release API 取固定 tag/URL/digest，下载到隔离空间后 `shasum -a 256` 与已知值比对，**校验通过才**列举归档成员、审阅 `USAGE.md` / `.cargo/config.toml`，拒绝危险路径再解包。按文档以确切 nightly + `--target aarch64-apple-darwin` 与 `TILERS_CODEGEN_PATH=metal` 调用后端（显式目标避免宿主 proc-macro 被后端接管），捕获命令、stdout/stderr、返回码、版本和输出目录。内置 Metal 可以同时出现，但**一律忽略**。
3. 编译失败时保存第一条诊断和完整 log，区分 `tile_std`/宏源码编译、标记识别、ABI/链接、后端本身等失败，**不要在没有报错证据时断言“版本不匹配”**；编译成功但无 `*.mlir` 另记为“MLIR 缺失”。仅在拿到来源可追溯的 Release 匹配 `tile_std` **及**宏源码时，用**同一 kernel、命令与工具链**做单独对照并保留两组报告；不能据 tag 日期、公开 repo tag 或 dylib 名猜私有构建提交。
4. 有真实 `.mlir` 后，从同一固定公开提交在隔离临时 crate 中依赖 `tile_codegen = { path = ".../crates/tile_codegen", features = ["emitters"] }`；读取该 `.mlir`，用 `TargetRegistry::with_builtin().select("msl")` + `CodegenTarget::emit(&mlir, &EmitOpts::default())`，独立保存 stdout/stderr/返回码与生成 `.metal`。报告 emitter **拒绝**（含原始错误）或**成功生成 MSL**，绝不以 Release 内置 MSL 替代。无需 GPU 执行或数值比较。

**所需下一步决策/授权（历史记录）**：提供明确供本项目使用的 macOS arm64 执行环境及获准的调用方式，或明确授权在现有 GitHub-hosted macOS runner 上开展独立实验；在此之前两项目标均维持“未实测/环境阻塞”。该授权后来已给出；以下是最终实测。

## 6. 授权后的真实 macOS arm64 实验：两个兼容目标均未通过

日期 2026-09-29 UTC。实验分支 [`experiment/rust-kernel-e2e-2924af3-v002-20260929`](https://github.com/Jopqior/tile-rs/tree/experiment/rust-kernel-e2e-2924af3-v002-20260929) 从固定公开源码 `2924af350f8843738f276b9a8695b0416a2d3d49` 建独立 `/tmp` worktree；只添加[分支限定 push workflow](https://github.com/Jopqior/tile-rs/blob/97ab34131175cf62cc31fb44df14a30fc868d4dd/.github/workflows/rust-kernel-e2e-experiment.yml) 和[实验输入/脚本](https://github.com/Jopqior/tile-rs/tree/97ab34131175cf62cc31fb44df14a30fc868d4dd/experiments/rust-kernel-e2e)。workflow 只匹配**该唯一分支的 push**，`contents: read`，标准 `macos-15`，30 分钟 timeout，无 secrets / 私有源 / tag / PR / 默认分支改动 / 正式 CI 接入。该分支的 `gh run list --branch ...` 仅列出六条同一实验 workflow，未看到别的任务被实验 push 触发。远端记录保留，不删除、不合并。主 worktree 始终只有原有的 `?? Cargo.lock`，未切分支或修改。

### 可复核的运行与证据

| run / 提交 | profile | 结果与分类 | 上传的 evidence |
| --- | --- | --- | --- |
| [36600093199](https://github.com/Jopqior/tile-rs/actions/runs/36600093199) / `5336937` | 未到编译 | **实验接线错误**：验证包成员时预设根目录为 `tile-rs-codegen-aarch64-apple-darwin`，实际是 `tile-rs-codegen/`；下载和 SHA 已成功。 | `rust-kernel-e2e-36600093199-1` |
| [36600234818](https://github.com/Jopqior/tile-rs/actions/runs/36600234818) / `aebcc73` | 未到编译 | **发布包与说明不一致/接线门槛**：`USAGE.md` 称附 `.cargo/config.toml`，经实际完整 SHA 校验的 tarball 无此文件；此 run 的包成员检查据此拒绝。 | `rust-kernel-e2e-36600234818-1` |
| [36600485282](https://github.com/Jopqior/tile-rs/actions/runs/36600485282) / `9655295` | 未到编译 | **外部 API 门槛**：匿名 Release API 暂时 403；已知 API digest/固定 URL 后续仍要求下载后 SHA 实测。 | `rust-kernel-e2e-36600485282-1` |
| [36600679857](https://github.com/Jopqior/tile-rs/actions/runs/36600679857) / `e311fcd` | debug | **实验接线错误**：Cargo 从仓库根执行时不按 `--manifest-path` 查找 kernel crate 的 `.cargo/config.toml`，后端 flags 缺席；`tile_std` 在默认 LLVM debug codegen ICE。不是发布后端兼容性结果。 | `rust-kernel-e2e-36600679857-1` |
| [36600896625](https://github.com/Jopqior/tile-rs/actions/runs/36600896625) / `02fbea9` | debug | **有效兼容性测试**：host macro 及目标 `tile_std` 使用正确目标后端 flags 编译；kernel 编译失败（内置 MSL 拒绝），实际 MLIR 存在；固定公开 emitter 也拒绝。 | `rust-kernel-e2e-36600896625-1`，含 `mlir/actual-kernel.mlir`、分开的 stdout/stderr、输入及命令 |
| [36601256344](https://github.com/Jopqior/tile-rs/actions/runs/36601256344) / [`97ab34131175cf62cc31fb44df14a30fc868d4dd`](https://github.com/Jopqior/tile-rs/commit/97ab34131175cf62cc31fb44df14a30fc868d4dd) | release | **固定输入的文档所示 release profile 对照**：仍是同一内置 MSL 错误、同一目标源码 emitter 拒绝；不是靠改 baseline 重新宣称通过。 | [`rust-kernel-e2e-36601256344-1`](https://github.com/Jopqior/tile-rs/actions/runs/36601256344)，含全部诊断、原始 MLIR 和使用方法 |

最后两次实际运行的 runner：macOS **15.7.9 arm64**（Xcode 16.4）；`rustc +nightly-2025-08-04 -vV` = **1.91.0-nightly f34ba774c**, host `aarch64-apple-darwin`, LLVM 20.1.8；`rustc-dev`、`llvm-tools`、`rust-src` 已安装。runner 读取官方 [0.0.2 Release API](https://api.github.com/repos/yijunyu/tile-rs/releases/tags/v0.0.2%2Bnightly-2025-08-04)（部分 run API 403，但之前及 debug 有确认）并下载唯一公开的 `tile-rs-codegen-aarch64-apple-darwin.tar.gz`；**实际文件** `shasum -a 256` = `70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891`，与已知 API digest 和大小 103,449,511 bytes 一致。先验证所有 tar 成员相对路径/类型，再隔离解包、审阅 `USAGE.md`；Release tar 仅有 `USAGE.md` 与 `lib/` 中五份 dylib，**确实没有** `USAGE.md` 所声称的 `.cargo/config.toml`。实验依据[公开 release workflow 的 flags 模板](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml) 自行生成仅限本次 kernel crate 的 `[target.aarch64-apple-darwin]` 配置；Cargo 从该 crate 目录执行并传 `--target aarch64-apple-darwin`，因此 `tile_std_macros`、syn、quote 等宿主 proc-macro 构建记录中**没有** `-Zcodegen-backend`，而目标 `tile_std`、`kernel_add_e2e` 的 rustc 命令**确有**该 flag（见 artifact `logs/frontend.stderr`）。`TILERS_CODEGEN_PATH=metal` 只选发布后端内置目标；其 MSL **从未送入固定公开 emitter、从未当作产物或通过证据**。

实际输入 `experiments/rust-kernel-e2e/kernel/src/lib.rs` 为 `#![no_core]`/`#![no_std]`、`#[tile_std::tile_kernel]` 的 `add_f32_small(a,b,c)`，以该提交 `tile_std::tile::{tile_load_f32, tile_add_f32, tile_store_f32}` 对一个 `1×8` f32 tile 执行 load/add/store；`Cargo.toml` 只依赖**固定 checkout** 的 `crates/tile_std`，其 manifest 再依赖同提交宏源码，crate-type `cdylib` + `lib`。输入与生成配置、完整 `cwd`+argv、各阶段 exit code、独立 stdout/stderr、发布 `USAGE.md`、成员列表、环境及 SHA 都在相应 run artifact。没有使用包内 `libtile_std_macros.dylib` 替代源码。

### 目标 (1)：Release 前端能否编译固定 DSL add kernel？**不能完整编译，但导出真实 MLIR**

在修正 Cargo 配置发现路径之后，debug run 前端 `exit 101`，release-profile run 也 `exit 101`；两者都是 kernel 阶段的**发布后端内置 MSL emitter**给出同一首个错误，而非宏展开失败：

```text
error: mlir_to_msl has no arm for `__tile_window_mask_f32`. Emitting it would drop the operation silently: the dispatch skips calls it does not match, so the kernel would compile, run, and quietly omit this step.
error: could not compile `kernel_add_e2e` (lib) due to 1 previous error
```

真实导出的原始文件（`target/aarch64-apple-darwin/<profile>/deps/libkernel_add_e2e.dylib.mlir`，artifact `mlir/actual-kernel.mlir`）分别为 debug **3,285,258 bytes** / sha256 `3985e1546cf2cc257208cad7a69cbc82fedaedece7b1d55fd87efec492017b07`、release **416,865 bytes** / sha256 `027a6aa0431904af30c7322515eb4f1ffef006e107aebca6b714c5e6ee157d80`。两者含 `sym_name = "add_f32_small"` 和 `hacc.entry`。release MLIR 的该入口仅有 `__tile_load_f32`×2、`__tile_add_f32`、`__tile_store_f32` 调用；`__tile_window_mask_f32` 出现在同一合并 module 的另一个 `_ZN8tile_std4tile20tile_window_mask_f32...` helper，而非 add 入口。故观察到的**确切报错位置**是发布后端内部对包含无关 helper 的 module 的 MSL 发射拒绝；无法读取私有后端源码来证明其完整扫描逻辑，不能仅由报错推出宏版本必须匹配。真正的 Rust→MLIR 阶段不是“没有输出”，但 Cargo 编译整体仍失败；不能把它称为目标 (1) 成功。

### 目标 (2)：同一提交公开 registry emitter 接受该份真实 MLIR？**拒绝，无 MSL**

从**固定 `2924af3` 的源码**构建临时 harness，依赖 `tile_codegen` `features=["emitters"]`，调用 `TargetRegistry::with_builtin().select("msl").emit(原始 MLIR, &EmitOpts::default())`，没有导入手写 MLIR 或发布后端内置 MSL。debug 原始 MLIR 首个拒绝：

```text
mlir_to_msl has no arm for `__tile_matmul_f32`; implemented are __tile_matmul_f32, ...
```

该 MLIR 在无关的 `tile_std::tile_matmul_f32` helper 内确有 `callee = @__tile_matmul_f32, ...`；[固定 emitter guard](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L1135-L1180) 却按文本 `@__tile_matmul_f32(` 识别已实现 arm，面对通用 MLIR 的 `@__tile_matmul_f32,` 会假报“不支持”。文档所示 release profile 的原始 MLIR 更短，首个拒绝变成：

```text
fixed-source registry msl rejected actual frontend MLIR: kernel @add_f32_small chains 2 compute intrinsics (__tile_add_f32}> : , __tile_add_f32) that this emitter cannot fold into one body; all but one would be silently dropped.
```

[固定 emitter 的 `reject_unfusable_compute_chains`](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L625-L653) 在函数体每行遇到 `@__tile_` 后以首个 `(` 截断；真实通用 MLIR 对**同一次** add 调用有 `llvm.mlir.addressof` 的 `global_name = @__tile_add_f32}> : ...` 和 `llvm.call` 的 `callee = @__tile_add_f32, ...`，于是误记成两种计算 intrinsic。此为根据**原始 MLIR + 固定公开源码**定位的拒绝机制，不是在实验中修改 emitter 或洗白错误。两个 run 的 `emitter.exit = 1`、`msl/fixed-source.metal` **均不存在**：**没有当前源码生成的 MSL**，不能宣称成功生成；后续可能还有其他未触发的兼容性问题。

### 版本假设、限制与最终状态

- 固定 `2924af3` `tile_std`/宏源码在正确配置下已编译，最小 add 的真实 MLIR 已导出；此对照的内置 MSL 报错发生在未用 helper 而非宏入口识别阶段；不能据此反推所有 DSL API 都兼容。**此处是 §6 当时的推断；§7 已实际尝试公开 Release tag 的 DSL/源码宏和包内宏，得到不同的 Cargo 编译结果。**
- 发布说明仅把 0.0.2 的修复关联到私有源 `4c9e43f68`、`a1d1014ba`；公开 Release tag `c3c8ec...` 和包内 macro dylib **不能证明**某份公开 `tile_std`/宏源码就是二进制构建版本。未发现可追溯且确定匹配的公开源码对照；没有私自猜提交、替换基线或探测私有仓库。
- **目标 (1)：在 §6 的固定 DSL 组合中实测失败（发布后端内置 MSL 错误）；MLIR 已存在。目标 (2)：固定公开 registry emitter 拒绝；当前源码 MSL 不存在。**两个 run 的 workflow 失败由上述产品/接口行为导致，前四次才是实验接线/网络门槛；无 GPU 执行、MSL 编译或数值正确性声明。不能将原始失败换基线后宣称 `2924af3` DSL 已通过。**公开 Release tag 对照的进一步实测见 §7。**

## 7. 公开 Release tag 的 DSL/宏及包内预编译宏：实际对照（2026-09-29 UTC）

### 来源、接线和边界

[官方 0.0.2 Release 的公开 tag](https://github.com/yijunyu/tile-rs/tree/v0.0.2%2Bnightly-2025-08-04) 指向公开提交 [`c3c8ec0169bd02757b51370ba9c6ec3b116b833d`](https://github.com/yijunyu/tile-rs/commit/c3c8ec0169bd02757b51370ba9c6ec3b116b833d)（提交标题 *Update from private development repository*）；该树**确有** [`crates/tile_std`](https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/tile_std) 和 [`crates/tile_std_macros`](https://github.com/yijunyu/tile-rs/tree/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/tile_std_macros)。相对于固定 `2924af3`，`tile_std`/宏树明显不同（`tile.rs` 约 4,313 行差异，宏实现 71 行差异）；并非悄悄重测原版。公开 tag 的 [`tile_std/src/tile.rs`](https://github.com/yijunyu/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/tile_std/src/tile.rs) 仍提供 `tile_load_f32::<1,8>`、`tile_add_f32`、`tile_store_f32`；[`tile_std_macros/src/lib.rs`](https://github.com/yijunyu/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/crates/tile_std_macros/src/lib.rs) 的 `tile_kernel` 仍注入 `#[tile::kernel]` 和 `#[unsafe(no_mangle)]`。故**不必改变 kernel 的 1×8 f32 逐元素相加计算含义，也不必改输入源码**；两 arm 的输入 `src/lib.rs` SHA256 均为 `71293e5b0335ea35fa59a6a464367f885e22a2cc5e2a65fe36b48d45f73ed4d3`。

[公开 tag 的官方发布工作流](https://github.com/yijunyu/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/.github/workflows/codegen-release.yml) 在发布时**从私有仓库 `main` checkout**，构建 `-p rustc_codegen_tile -p tile_std_macros --release` 并拷贝两个 dylib 打包；**未固定/公开实际私有构建 SHA、未证明** tag 的 `tile_std` 或宏源码与包内库的私有构建同版。[Release notes](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04) 只把 0.0.2 的 rlib 修复追溯到私有提交 `4c9e43f68` + `a1d1014ba`；不能从这些提交反推完整来源。此处的“Release 对应版”严格指**公开 Release tag 版本**，而不是已证明的**私有构建同版本**。尽管无法证明后者，仍实际测试了前者和包内库，而没有以不确定性为由停止。

安装路径也要区分：发布包 `USAGE.md` 和上述 workflow 给的是 `TILERS_CODEGEN_SO` 指向 backend、`TILERS_CODEGEN_PATH=metal`、nightly 及 rustflags；实际 SHA 校验后的资产**没有**说明声称的 `.cargo/config.toml`，脚本按发布 workflow 模板为实验 crate 生成等效、限目标架构的 flags。[官方 `install.sh` @ tag](https://github.com/yijunyu/tile-rs/blob/c3c8ec0169bd02757b51370ba9c6ec3b116b833d/scripts/install.sh) 默认还是 0.0.1，需 `TILERS_CODEGEN_TAG=v0.0.2+nightly-2025-08-04` 才装本资产。官方 USAGE/安装脚本**没有写出用包内 `libtile_std_macros.dylib` 代替 Cargo 路径依赖宏的步骤**；这个 dylib 确实在资产内，但包内宏 arm 是**明确记录的实验性显式接线**，不是声称官方推荐的自动安装路径。

新独立 `/tmp` worktree 的[实验分支](https://github.com/Jopqior/tile-rs/tree/experiment/rust-kernel-e2e-release-dsl-v002-20260929) 始于 §6 的 `97ab341`，只改该分支的[临时 workflow](https://github.com/Jopqior/tile-rs/blob/5df0cfe87ae50e74c4454f23c359864ee99d0d06/.github/workflows/rust-kernel-e2e-experiment.yml)触发为**新分支唯一 push**并添加[两臂实验脚本](https://github.com/Jopqior/tile-rs/blob/5df0cfe87ae50e74c4454f23c359864ee99d0d06/experiments/rust-kernel-e2e/run.sh)；不修改生产 emitter、默认分支或原研究主工作区，原有 `?? Cargo.lock` 保留。标准 `macos-15`，每 job 30 分钟，`contents: read`，无 secrets / 私有访问 / GPU 验证。实际 runner：macOS 15.7.9 arm64，Xcode 16.4，nightly-2025-08-04 `rustc 1.91.0-nightly f34ba774c`，LLVM 20.1.8。前端二进制**仍是** §6 的官方 0.0.2、实测 SHA256 `70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891`；runner 的匿名 Release API 返回 HTTP 403，使用此前从官方 API 固定的 URL/大小/digest，并在每次下载后实际校验大小与 SHA 才解包，不能把此次 403 当成新 API 在线确认。包内 `libtile_std_macros.dylib` 是 arm64 Mach-O dylib，`otool -L` 仅显示自身 `@loader_path` 与 `/usr/lib/libSystem.B.dylib`，实测 SHA256 `02093811a50b0fe6c2518d1a8b48a5ee001c1713299dbbcfd436656ccccb4157`。它不仅“存在”，下述第二臂也证明在该 nightly 上**能供 rustc 实际加载/展开 kernel**。

每臂用同一 kernel、公开 tag `tile_std`，`cargo +nightly-2025-08-04 build --target aarch64-apple-darwin --release -v`、`TILERS_CODEGEN_PATH=metal` 和发布后端；单独目标目录保留每臂真实 MLIR。第一臂从 tag 的 manifest 正常编译、加载 tag 源码 `tile_std_macros`；第二臂也依赖 tag 的 `tile_std`，由[实验 rustc wrapper](https://github.com/Jopqior/tile-rs/blob/5df0cfe87ae50e74c4454f23c359864ee99d0d06/experiments/rust-kernel-e2e/bundled-macro-wrapper.py) **仅对 `tile_std` 的 `--extern tile_std_macros=…` 换成包内 dylib**，并为 re-export 在目标 rustflags 增加 `-Ldependency=<包内 lib 目录>`。该臂的 Cargo **仍会构建 tag 的源码宏依赖**（工作流日志可见），但日志记录了 `tile_std` 实际消费的是包内 dylib，不虚称源码宏完全没构建；其余 host 依赖不走发布 backend。两臂 MLIR 分别送到**固定 `2924af3`** `tile_codegen` 的 `TargetRegistry::with_builtin().select("msl").emit(&mlir, &EmitOpts::default())`，只输出当前源码 emitter 的 MSL，绝不复用包内内置 Metal。

### 运行和实测矩阵

| 来源及 run | Release 前端 Cargo 结果（与 MLIR 分开） | kernel 的真实 MLIR | 固定 `2924af3` registry MSL |
| --- | --- | --- | --- |
| `2924af3` `tile_std` + 其源码宏，[§6 release run 36601256344](https://github.com/Jopqior/tile-rs/actions/runs/36601256344) | **exit 101**；发布后端*内置* MSL 不接受未用 helper `__tile_window_mask_f32`（不是宏编译失败） | **有**，416,865 bytes，sha256 `027a6aa0431904af30c7322515eb4f1ffef006e107aebca6b714c5e6ee157d80` | **拒绝**原始 MLIR，误计两次 `__tile_add_f32`；无当前源码 MSL |
| 公开 `c3c8ec0` tag `tile_std` + tag 源码宏，[有效 run 36603219250](https://github.com/Jopqior/tile-rs/actions/runs/36603219250) | **exit 0、`Finished release`**；不是仅“有 MLIR” | **有**，373,002 bytes，sha256 `26639ca32e8b6ff7f7fe3ec7fe02e71a59fa0ee1f13b81f8a41318bb22e77ff3` | **exit 1、拒绝**真实 MLIR，误计两次 `__tile_add_f32`；当前源码 MSL **不存在** |
| 公开 `c3c8ec0` tag `tile_std` + 包内预编译宏，**同一有效 run** | **exit 0、`Finished release`**；`--extern` 实际被替换、包目录进入依赖搜索路径 | **有**，373,002 bytes，sha256 **同上一行**（逐字节相同） | **exit 1、拒绝**真实 MLIR，同一错误；当前源码 MSL **不存在** |

[首个 run 36602935190](https://github.com/Jopqior/tile-rs/actions/runs/36602935190)已让 tag 源码宏臂 **Cargo exit 0、真实 MLIR、emitter 拒绝**；包内宏臂虽 `tile_std` 编译、wrapper 确实换掉 `--extern`，但 kernel 在加载其 re-export 时首先 `error[E0463]: can't find crate for tile_std`，同时缺 `panic_handler`/`sized`（连锁诊断），当时无该臂 kernel MLIR。原因是实验没有向依赖方 rustc 暴露包内 dylib 的目录；这属于**显式接线错误**，不能算包内宏 ABI 不兼容。加入 `-Ldependency=<bundle>/lib` 后在有效 run 36603219250 两臂均完整编译；该修复及两次独立运行/日志全留在分支历史与 artifacts，未无限重试。

有效 run 的 `tag_source_macro.frontend.exit` 和 `bundled_macro.frontend.exit` 均为 `0`，独立的 `*.emitter.exit` 均为 `1`；因此 Actions workflow 整体是 **failure**，含义是当前 emitter 不通过，**不是**“cargo 编译失败”。原始拒绝日志（两臂相同，完整 stderr 在 artifact）：

```text
fixed-source registry msl rejected actual frontend MLIR: kernel @add_f32_small chains 2 compute intrinsics (__tile_add_f32}> : , __tile_add_f32) that this emitter cannot fold into one body; all but one would be silently dropped. Use a fused tile_std op, split the kernel, or add the pair to FUSED_COMPUTE_PAIRS with a fused KernelType.
```

MLIR 中 `add_f32_small` 入口真实存在，有 `hacc.entry`；针对单一 add 的 `llvm.mlir.addressof` 写 `global_name = @__tile_add_f32}> : …`，`llvm.call` 写 `callee = @__tile_add_f32, …`，正好触发 §6 所述[固定 emitter 的文本扫描](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/crates/rustc_codegen_tile/src/mlir_to_msl.rs#L625-L653)误计。同一份 tag 组合的 MLIR 内已无 `__tile_window_mask_f32`；这解释为何 §6 的内置阶段报错不再出现，但不能证明删除该 helper 是唯一根因，也不能证明任意复杂 kernel、同一私有源码版本或 Metal 数值正确性。实验从未修改 emitter 来绕过拒绝。

**复核路径：** [有效 run 36603219250 的 artifact `rust-kernel-e2e-36603219250-1`](https://github.com/Jopqior/tile-rs/actions/runs/36603219250) 含 `identity.txt`、`environment.txt`、`source-provenance.txt`、`asset-url.txt`、`archive-members.txt`、`inputs/`（两臂 Cargo/config、同一 kernel、wrapper、发布 USAGE、emitter harness）、`commands.txt`（cwd 与实际 argv）、`logs/`（每阶段独立 exit/stdout/stderr、宏 dylib 的 file/otool/sha 及 `bundled-macro-swaps.txt`）、`tag_source_macro/mlir/actual-kernel.mlir` 和 `bundled_macro/mlir/actual-kernel.mlir`（各自 SHA）。`msl/` 只用于**成功时**保存 `fixed-source.metal`；本次两个 emitter 均拒绝，**没有可上传的当前源码 MSL**，证据是各臂 `*.emitter.stderr` / `*.emitter.exit`，不能拿内置输出充当缺失产物。[首 run 36602935190 的 artifact](https://github.com/Jopqior/tile-rs/actions/runs/36602935190) 保留接线错误的原始 stderr。两 run 所在分支仅见该临时 workflow；未运行 GPU、`xcrun metal -c` 或数值参考比较。

**结论更新：** Release tag 源码 DSL/源码宏的实际尝试使目标 (1) **通过 Cargo 编译**且**有 MLIR**；包内预编译宏也能在明确接线后供这一 DSL/kernel 使用，并导出字节相同的 MLIR，纠正了把“无法证明私有版本同源”当成停止条件的误区。目标 (2) 对两臂仍**失败**：固定当前源码 registry 不接受**真实原始** MLIR，当前源码 MSL 没生成。可说 tag 候选在这一 kernel 上消除了原 `2924af3` DSL 组合的内置 MSL 失败；**不能**说二进制的私有构建就是 tag 代码、不能说跨版本约定全兼容或 Rust→MLIR→当前源码 MSL→GPU 已通过。
