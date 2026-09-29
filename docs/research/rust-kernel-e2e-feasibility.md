# Rust kernel → MLIR → 当前源码 MSL → Metal GPU → 参考比较：可行性核查

- 调研日期：**2026-09-29 UTC**。范围：[本地地图 #42](https://github.com/Jopqior/tile-rs/issues/42) 的**新要求**（取代该票正文中“Rust → MLIR 超出范围”的旧边界）；只研究，不更改票据或实现。
- 固定源码：规划 checkout [`2924af350f8843738f276b9a8695b0416a2d3d49`](https://github.com/Jopqior/tile-rs/commit/2924af350f8843738f276b9a8695b0416a2d3d49)（含固定上游 `e8d1acd`）；本地 `main` / `upstream/main` [`6d59de3018ff85dacd8a8b0a98215f7063eff211`](https://github.com/yijunyu/tile-rs/commit/6d59de3018ff85dacd8a8b0a98215f7063eff211)。引用后者是分析已有接口，**不等于**让规划分支跟进它。
- 方法：`gh` 查公开仓库、Release API、工作流及运行记录；对两个固定提交只读源码。没有下载/执行发布二进制，没有运行 macOS/GPU/CI，也没有读取或测试任何凭据/私有仓库。因此以下“可接”是**静态可行路径**，不是兼容性或正确性验收。

## 结论（供下一步决策）

**选用固定预构建前端作为候选，而不是“从当前公开源码构建 Rust→MLIR 前端”**：两份公开 checkout 都有 `tile_std`、宏、MLIR parser/Metal emitter、`tile_codegen`，但没有可构建的 `rustc_codegen_tile` 前端 crate（例如 `src/lib.rs` / 其 manifest）；上游公开工作流明确从私有代码源取前端。公开 Release 确实有可下载的 **Apple Silicon** 预构建前端，建议以 **0.0.2** 为兼容性候选（0.0.1 对标准 rustc rlib/core 有已记载缺陷），但尚未验证它与固定 `2924af3` 或 `6d59de3` 的 DSL/MLIR 契约。[公开 README 架构](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/README.md)、[私有 checkout 的公开工作流](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml)、[公开树的文件列表](https://github.com/yijunyu/tile-rs/tree/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/rustc_codegen_tile)、[0.0.2 Release](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04)。

**二进制不是“只含前端”的可替换插件**：官方说明它直接选择 `TILERS_CODEGEN_PATH=metal` 并产出 Metal；但 `main` 的公开 CLI 源码又说明其编译过程**同时写出 `.mlir` 和 `.tile.*`**，CLI 的 `lower_rs::lower()` 会分别读回两个文件。故存在具体的可研究接缝：**仅消费预构建前端产出的 MLIR，明确丢弃其内置 Metal，再以被测试提交的公开 `tile_codegen`（`emitters` 特性）或其 `convert_mlir_to_msl` 处理 MLIR**。不证明这个跨版本接缝可运行；也不能把“使用二进制自带 Metal”冒充“当前源码 emitter 受测”。[CLI 源码](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/lower_rs.rs)、[CLI route 行为](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/src/bin/tile.rs)、[registry/emitters](https://github.com/Jopqior/tile-rs/tree/2924af350f8843738f276b9a8695b0416a2d3d49/crates/tile_codegen/src)。

## 1. 公开性、真实 Release 与 ABI：把“计划”与“已发布”分开

- `gh api repos/yijunyu/tile-rs`：公开仓库 (`private=false`)；两个 checkout 的 `crates/rustc_codegen_tile` 为 MLIR→目标源码文件，缺少完整可构建前端。公开 [codegen-release.yml @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml) 注释称代码源私有，以 `actions/checkout@v5` + 只读 `PRIVATE_SOURCE_TOKEN` checkout；具体 repo 名用 `PRIVATE_SOURCE_REPO` secret，**不得从公开文件推定私有仓库此刻可被我们读取**。`2924af3` 的同一工作流仍硬编码 `yijunyu/ascend-rs-priv`，这只是**当时公开配置**，不是可访问性测试。[旧工作流 @ 2924af3](https://github.com/Jopqior/tile-rs/blob/2924af350f8843738f276b9a8695b0416a2d3d49/.github/workflows/codegen-release.yml)。未访问私有 repo/API，也未探测 secret。
- `gh release list` 和 `gh api repos/yijunyu/tile-rs/releases`：截至调研日**恰有两个公开 prerelease**，均各仅一份 `tile-rs-codegen-aarch64-apple-darwin.tar.gz`：[`v0.0.1+nightly-2025-08-04`](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.1%2Bnightly-2025-08-04)，2026-06-21，101,724,767 bytes，sha256 `231f85f11622ae38eeb724d4804432a7a394f3c2f371076de8bd57292c3d63b4`；[`v0.0.2+nightly-2025-08-04`](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04)，2026-09-06，103,449,511 bytes，sha256 `70fffed473aa2d5c3f51bcc04caf9d18e0379188e764689ed52be0c373926891`。sha256 为**Release API 所列 digest**，不是本次本地下载验证。无已发布 Linux/x86 macOS 资产；别将 [工作流计划的 `macos-14` arm64 与 `ubuntu-latest` x86_64 双平台矩阵](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml) 当成真实资产。
- 两份 [Release notes](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04) 声明 LLVM/MLIR 20 通过 `@loader_path` 随包，后端 dylib 依赖确切 **`nightly-2025-08-04`**（rustc `1.91.0-nightly`, `f34ba774c`）的 `librustc_driver` ABI，需 `rustc-dev`、`llvm-tools`、`rust-src`；0.0.2 不改变 ABI，只修复 0.0.1 把标准 `ar` 格式 rustc `rlib` 当作 `tar` 解读、无法依赖 `core` 的问题。Release notes 说其发布前曾用真实 CLI 降低 `#![no_std]` softmax 并用 `xcrun metal -c` 编译；这是**上游的声明，不是本次实测**。本地 `main` 的 [provision.toml @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/crates/tile_cli/assets/provision.toml) 已固定 **0.0.2 + 同一 checksum**，且不为没有资产的平台制造条目。
- [安装脚本 @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/scripts/install.sh) **默认仍指向 0.0.1**，可用 `TILERS_CODEGEN_TAG` 改写；因此不能把 README 的一行安装命令视为自动安装修复版。`tile` CLI provision 是不同的、固定 0.0.2 的安装通道。上游 [install-smoke.yml @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/install-smoke.yml) 在 `macos-14` 核验脚本安装、配置、动态库与依赖，**并不执行 Rust kernel→MLIR→GPU**；[`gh run` 的成功记录](https://github.com/yijunyu/tile-rs/actions/workflows/install-smoke.yml) 不能当全链验收。公开 [Codegen Release workflow 页面](https://github.com/yijunyu/tile-rs/actions/workflows/codegen-release.yml) 所见最近两条 push 运行状态为 failure；既有真实 Release 资产与工作流最新运行失败并不矛盾，不能据工作流矩阵推断后续发布成功。

## 2. 确切输出与接缝：已有证据 / 尚未证实

- [Release 0.0.2 文本](https://github.com/yijunyu/tile-rs/releases/tag/v0.0.2%2Bnightly-2025-08-04) 与 [工作流生成的 USAGE.md 模板 @ 6d59de3](https://github.com/yijunyu/tile-rs/blob/6d59de3018ff85dacd8a8b0a98215f7063eff211/.github/workflows/codegen-release.yml)：包内 `lib/librustc_codegen_tile.dylib`、`lib/libtile_std_macros.dylib`、`.cargo/config.toml`、`USAGE.md`；其中 `.cargo/config.toml` 使用 `-Zcodegen-backend=lib/librustc_codegen_tile.dylib`、`-Zcrate-attr=feature(register_tool)`、`-Zcrate-attr=register_tool(tile)`、`-Cpanic=abort`、`-Clto=off`；`TILERS_CODEGEN_SO` 指定 dylib，`TILERS_CODEGEN_PATH=metal` 选择内置 Metal（**不是**本地 registry 的 `msl` 名）。文档例子 `cargo +nightly-2025-08-04 build --release` 要求 kernel crate `crate-type=["cdylib","lib"]`。
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

**所需下一步决策/授权**：提供明确供本项目使用的 macOS arm64 执行环境及获准的调用方式，或明确授权在现有 GitHub-hosted macOS runner 上开展独立实验；在此之前两项目标均维持“未实测/环境阻塞”。
