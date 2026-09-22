# TileLang 及 Ascend 实现的 CI 与资源边界

调查日期：2026-09-22（UTC）。票据：[核实 TileLang 及 Ascend 实现的 CI 与资源边界](https://github.com/Jopqior/tile-rs/issues/51)。仅研究，不运行上游构建或设备测试，不实施 tile-rs CI。

## 摘要

- 主项目是 `tile-ai/tilelang`，Ascend 是主项目 README 明确链接的外部生态实现，不是其 CUDA 测试矩阵的一项。AscendC/PTO 主要调查对象是 `tile-ai/tilelang-ascend:ascendc_pto`；MLIR 路径另有 `tile-ai/tilelang-mlir-ascend:main`，旧 `tilelang-ascend:npuir` 分支仍存在，不能当成同一个当前版本。[S1][S2][S3]
- AscendC/PTO **已有公开 NPU workflow 和实际 PR、定时运行**。不能依据 README 的“手动 bench_test.sh”说明，或被禁用的旧 `ci.yml`，得出只有本地脚本的结论。[S4][S5][R1][R2]
- 格式检查不需要 NPU；host 构建、编译参数/IR 测试、设备执行需要分层。MLIR 路径公开演示了 x64/arm64 托管 host 构建与 lit 测试后，再将 arm64 wheel 交给 NPU runner 测试。[S10][S11][S12][R3]
- AscendC/PTO 样本 runner 名是 `ascend_cicd_a3`，工作流限制逻辑设备可见性为 `0`；这不是“一张物理卡”的证明。MLIR 的 `linux-aarch64-a3-800i-2` 也只应先视为标签，末尾 `2` 不是经过核实的卡数。[S4][S12][R1][R3]
- 有 camodel 仿真实现，但只证明存在另一种运行入口，不证明当前 NPU CI 使用仿真，也不证明所有测试无真机可跑。[S9]
- 尚缺设备池清单、物理卡数、驱动/固件组合、AscendC/PTO 容器镜像及 CANN 实际版本、外部贡献者准入和强制合并门禁。不可据此估算采购规模。

## 来源与版本、项目归属

本次使用 GitHub 第一方仓库、REST API、固定提交源码及公开 Actions 日志。环境没有 web_search 工具，采用 `gh search repos tilelang ascend`、仓库 README 交叉链接、workflow 列表及运行 API 多角度核实，没有依赖搜索摘要或第三方镜像。

| 对象 | 分支及固定提交 | 归属与范围 |
| --- | --- | --- |
| TileLang 主项目 | `main`，`8cf0e7eff4593d8c0236bd45e74a5cacb797072f` | README 将 Ascend 标为 Ecosystem，链接下面两个实现；其当前 CI 是 NVIDIA CUDA、AMD ROCm、Apple Metal，不是 Ascend CI。[S1][S6] |
| TileLang-Ascend | 默认 `ascendc_pto`，`585ad8c3b5fae175f1d1661059d347645f1f9536` | README 自称 specialized variant，AscendC/PTO 路径；GitHub API `fork:false`，无 parent，故“fork”只可作代码衍生关系描述，不能断言它是 GitHub fork network 中的 fork。[S2] |
| TileLang-MLIR-Ascend | 默认 `main`，`54a19e28dfd8756daa253b387e00d10ba7f952e7` | 主项目明确链接的 MLIR 生态仓库；README 仍有指回旧 npuir 分支的链接，不能仅凭旧链接忽略新仓库。[S1][S3] |
| 旧 NPUIR 分支 | `tilelang-ascend:npuir`，`bd7c60c312b0721885c78d1ce03c62be9225314c` | 抽查其测试 workflow：标签是 `linux-aarch64-a3-2`，例子显式枚举；新仓库已是 `linux-aarch64-a3-800i-2` 和 `examples/run_all.py`，两者不完全相同。[S13] |

成熟度只能按证据描述：AscendC/PTO README 记载 2025-09-29 开源、2026-01-23 PTO 后端落地；已有代码、测试、PR/nightly 运行。MLIR 有构建、lit、设备测试及 release 资产依赖链。它们不是“只有计划”，但这些也不构成生产 SLA、所有型号覆盖或与主项目功能齐平的保证。[S2][S10]

## CI 入口、触发与门禁

### 主项目：只用于澄清归属

`ci.yml` 在 PR 和手动事件触发。Quick Lint 用 `ubuntu-latest` 执行 Python AST、pre-commit 等；测试依赖 lint，且限制 `tile-ai` owner、非 draft PR。矩阵是 `[self-hosted,nvidia]` / CUDA-auto、`[self-hosted,amd,gfx942]` / ROCm-7.2、`macos-latest` / Metal，另有 CuTeDSL CUDA examples。不能把此处 GPU runner、CUDA 工具链或 pytest 并行数移植为 Ascend 资源事实。[S6]

### AscendC/PTO

1. `ci_ascend.yml` 虽名为 AscendC-PTO CI，实际只有 `ubuntu-latest` 上针对变更文件的 Ruff 格式/lint、clang-format；没有设备步骤。[S5]
2. 真正设备链在 `ci_cd.yml`：面向 `ascendc_pto*` 基础分支的 PR（忽略文档/图片等）、手动 `workflow_dispatch`、每日 UTC 18:00 cron，以及评论事件。`/re-test` 请求重跑先前 PR workflow 的失败 jobs；评论事件自身的 test job 被条件排除，不能将一个成功的评论运行当成设备执行。[S4]
3. `check_changes` 在 `ubuntu-latest` 选测试范围；核心代码变更走 examples + pytest，测试/例子变更可增量，存在 migrated operator 路由及排除项。PR 排除 `low_priority` 和 `ci_skip`，定时/手动仅排除 `ci_skip`。所谓“全量”仍受脚本排除规则约束，不是仓库每个测试。[S4][S7]
4. `Benchmark Tests (Ascend NPU)` 用 `[self-hosted, Linux, tilelang-ascend-ci]`，PR 超时 60 分钟，定时/手动 360 分钟。`docker exec tilelang_x1` 内先 `install_ascend.sh`，再 `set_env.sh`、`bench_test.sh` 和/或 pytest。nightly 强制清理构建，PR 可恢复 runner/branch/关键文件哈希关联的构建缓存。[S4]
5. 进程退出码及 pass-rate 检查使测试失败成为 job failure；日志/artifact `test-results` 保留配置为 30 天。nightly 另创建结果 issue。旧 `ci.yml` 的 format job `if:false`，build-test 依赖它，不能拿旧 self-hosted 名称证明活跃 NPU 门禁。[S4][S5]
6. workflow 内部依赖不等于强制合并门禁。本次读取 `branches/ascendc_pto/protection` 返回 404，无法区分权限/配置等原因；未据此断言无 branch protection。required checks、管理员 bypass、外部 PR 审批政策待确认。

### MLIR / AscendNPU IR

`base_ci.yml` 对 `main` PR 和 release `published/prereleased` 触发。format 与预构建并非串行强依赖；NPUIR、TVM 的 x64/arm64 host 预构建后，wheel job 构建 TileLang、运行 `lit -vv build/testing/mlir`、打包，再调用 NPU wheel 测试。release/prerelease wheel 上传依赖 test 成功；正式 release 的 Docker build 也依赖 test。这里证明的是发布步骤的 workflow 门禁，不是已核实的 branch required checks。[S10][S11]

设备测试使用 arm64 wheel、CANN 8.5.0 A3 Ubuntu 22.04 Python 3.11 容器、torch/torch-npu 2.7.1；执行 `examples/run_all.py --sequential` 与 `pytest testing/npuir/*_ops -n 2`，输出 HTML/JUnit。NPUIR 预构建另支持 main push/手动触发。本次未核实此路径 nightly 或真实 release 运行，不能以 PR 样本替代。[S12][S14]

## 实际运行证据，不只看绿色 workflow

| 运行 | 观察到的内容 | 边界 |
| --- | --- | --- |
| [R1：AscendC/PTO PR](https://github.com/tile-ai/tilelang-ascend/actions/runs/35721352283)，2026-09-22，head `bf20e2b78b2e3c606ba3347342e9e8ca9ef84e58` | Job `106726807744`，`ascend_cicd_a3`，Build and Test、Check pass rate 均成功；日志 11:37:45 UTC 报 `[PASSED] ./gemm/example_gemm_pto_developer.py`，11:46:56 报 PTO min E2E PASSED。 | PR head 不是本报告默认分支提交；用于证明真实执行，不能将此 PR 所有测试细节归到默认分支。 |
| [R2：AscendC/PTO schedule](https://github.com/tile-ai/tilelang-ascend/actions/runs/35658878616)，2026-09-21 | head 恰为本报告固定提交 `585ad8c...`；Job `106529130733`，runner `ascend_cicd_a3` 成功。日志 21:57:53 报 PTO GEMM PASSED，22:10:21 报 PTO min E2E PASSED，最终 bench/pytest 摘要 100%。 | 实际启动 21:44 UTC，与 cron 名义 18:00 不同，不承诺准时调度；摘要包含过滤/预期失败语义，不代表所有可能配置。 |
| [R3：MLIR PR](https://github.com/tile-ai/tilelang-mlir-ascend/actions/runs/35698983671)，2026-09-22，head `4fe74a6c15e1b1fdc0f4d6e60acd96458d965e99` | x64/arm64 build-wheel jobs 成功；设备 job `106700797733`，runner `linux-aarch64-a3-800i-2-cn12-001-hb7qm-runner-bl4gm`。日志 10:44:21：70 examples passed；10:53:13：501 pytest passed，含 GEMM。 | PR 样本，不证明 release 已执行，也不证明 x64 wheel 已在 NPU 真机运行。 |

通过 `gh api repos/.../actions/runs/<id>/jobs` 及 `gh run view <id> --log` 核对上述 jobs/日志，避免仅靠 workflow success。日志里的 `.npu()` 算子与公开测试源码形成设备执行证据；未拿到设备物料清单或 `npu-smi` 盘点。[S8][R1][R2][R3]

## 按阶段的 NPU 必要性与证据矩阵

“无 NPU”不等于“无 SDK/库/host 软件依赖”。下表把源码逻辑判断和运行事实分开。

| 阶段 | NPU 必要性 | 证据及限制 |
| --- | --- | --- |
| 文档、Python/C++ 格式与静态检查 | 不需要 | Ascend 格式 job 在 Ubuntu hosted runner 上执行；不安装 NPU 栈。[S5] |
| 编译器 host 构建/打 wheel | 原理上与设备执行分开 | AscendC/PTO 安装脚本做 pip/submodules/CMake/make、`USE_ASCEND ON`，没有主动启动设备 kernel；但 CI 把它放在 NPU 容器内，本次没有独立无卡构建实测，不保证默认安装在任意 CPU 主机成功。[S15] |
| MLIR host 构建和 IR lit 检查 | 公开明确不依赖 NPU runner | x64/arm64 GitHub hosted jobs 构建后运行 lit，再传递 wheel；有真实 build-wheel 成功样本。[S11][S14][R3] |
| Ascend 编译参数/缓存等纯逻辑测试 | 选定子集不需要 | `test_ascend_compile_flags.py` 明说参数测试 hermetic（no NPU / bisheng），mock 外部命令；文件末尾仍有 NPU-gated regression，不能把整个 testing/python 标作 CPU-only。导入 TileLang 的动态库依赖仍须满足。[S16] |
| 生成 AscendC/PTO C++、用 Bisheng 编译共享库 | 编译与设备运行可分，但需要工具链 | `libgen.py` 调用 Bisheng，PTO 走 `-xcce`、PTO ISA headers、CANN/runtime 链接，含 driver header 搜索路径；这是编译依赖证据，不是必须插卡的直接证据。默认 JIT 还涉及 adapter/load，纯 host 可用边界需独立验证。[S17] |
| camodel 仿真 | 特定入口意在无真机模拟 | `TL_RUN_MODE=sim` 改连 `libruntime_camodel`；A5 模板设置 CPU 库路径、关闭 torch backend autoload，使用 camodel runtime。仅源码入口，未运行、本次未发现已检查的 CI 显式启用它。[S9][S17] |
| 单设备 correctness / E2E | 默认 NPU 路径需要可执行设备 | PTO GEMM `.npu()` 分配张量、调用 kernel、与矩阵乘参考结果 assert_close；R1/R2 有成功日志。不能用 host 编译成功代替数值、同步、运行时集成正确性。[S8] |
| 性能/稳定性验证 | 真机结论须真机测试 | 仿真、格式/IR 测试不验证真实带宽、延迟、驱动或长期稳定性；已阅 workflow 没有证实跨版本性能回归阈值是强制门禁。 |
| 多设备/分布式、shmem | 是不同资源与测试层 | CI 变更筛选、bench 默认排除 `dispatch_combine` / `shmem`；不能从单逻辑设备 CI 推导分布式覆盖，更不能从 pytest `-n 8` 推导 8 卡。[S4][S7] |

## 公开可证实的资源条件

| 项 | 观察事实 / 配置事实 | 官方项目要求 / 推断 / 未知 |
| --- | --- | --- |
| Ascend 型号 | AscendC/PTO 样本 runner 名含 A3；MLIR 标签含 A3/800i。[R1-R3] | README 声称 A2/A3 tested devices，这是项目验证声明，不是实际 CI 设备池列表。[S2][S3] 精确 SKU、内存、设备池规模未知；libgen 的 A5 分支也不是 A5 真机池证据。 |
| 数量与隔离 | AscendC/PTO 对 runner `ascend_cicd_a3` 设置 `ASCEND_RT_VISIBLE_DEVICES=0`，R1 日志确认该分支执行。[S4][R1] | 仅逻辑设备可见性；不证明物理卡数、独占、切分方式。MLIR 标签后缀 `2` 和 pytest `-n 2` 都不能当设备数。 |
| Host 架构 | MLIR host build：x64 和 arm64；NPU 测试标签 aarch64、安装 arm64 wheel。[S11][S12] AscendC/PTO 日志有 `linux-aarch64.egg` 路径、Python 3.12.13。[R1] | AscendC/PTO ARM64 是安装产物的线索，不是 lscpu 盘点；CPU 型号、核数、RAM 未核实。make 使用 `nproc` 的 50%，不是固定资源最低值。[S15] |
| CANN/torch | AscendC/PTO R2：torch `2.7.1+cpu`、torch-npu `2.7.1.post2`；CPU 标记不表示 NPU 测试没用 NPU。[R2] MLIR 配置 CANN `8.5.0` 镜像、torch/torch-npu `2.7.1`。[S12] | AscendC/PTO README 要求 CANN 至少 `8.3.RC1`、torch-npu 至少 `2.6.0.RC1`。[S2] 不是 PTO 所有算子已验证的最低兼容组合；AscendC/PTO CI 实际 CANN/bisheng 版本未知。 |
| 驱动/固件 | PTO compile command 搜索 `/usr/local/Ascend/driver/kernel/inc`。[S17] | 编译搜索路径不证明驱动已加载；两个实现的 CI 驱动、固件版本及 CANN 配套矩阵未核实。 |
| 容器 | AscendC/PTO 复用既存容器 `tilelang_x1`；MLIR 使用 `swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-a3-ubuntu22.04-py3.11`。[S4][S12] | 前者 image/tag/digest、创建命令、device mounts 未公开于该 workflow；后者 tag 不是不可变 digest，标签也不等于镜像内软件实测清单。未据此要求 privileged。 |
| 网络与权限 | AscendC/PTO checkout、pip、submodule 需要对应服务可达；子模块含 GitHub 与 GitCode（catlass/PTO ISA/shmem）。Docker exec 需要访问容器及共享工作目录。[S4][S15][S18] | 可用预缓存减少联网，但不能假定完全离线可复现；实际代理、网络白名单、Docker 授权策略未知。 |
| MLIR 内部访问 | 测试配置依赖 `cache-service.nginx-pypi-cache.svc.cluster.local`，改写 apt/pip 源，并访问华为镜像/软件源。[S12] | 该集群域名不是公众互联网可达性的保证；外部 runner 直接复用需要替换源或获得网络接入。没有追逐内部系统。 |
| 用户可申请资源 | MLIR README 指向 [HiDevLab Online Development](https://hidevlab.huawei.com/online-develop-intro)。[S3] | 只证明申请入口，不证明有可供 CI 自动调度的配额、API、期限或 SLA，更不等于可以接入维护者 runner。 |

PTO ISA 子模块在 AscendC/PTO 固定提交中指向 `682771db273272a5c34332c790c445106d34dbb6`，origin 是 `https://gitcode.com/cann/pto-isa.git`。[S18] 本报告不把上层 JIT 的 Bisheng 编译称为已证明的“PTO AS 工作流”：实际命令是 C++/CCE 及头文件路径，PTO AS 的职责/兼容性须由独立工具链证据核实。[S17]

## 无 NPU 能做和不能验证什么

能直接参照：格式/静态检查；有可导入 host 构建的条件下，选取 hermetic 参数测试；MLIR 编译/IR lit/多架构 wheel 流程；有匹配 CANN、camodel 的条件下探索指定仿真入口。AscendC/PTO 默认安装/JIT 在无卡主机的可运行程度尚需小型独立验证，不能因没有显式 kernel launch 就承诺整个安装和测试全通。

不能验证：真实 NPU kernel 数值与同步正确性、设备 runtime/驱动/固件集成、实际时延/吞吐、设备内存压力及分布式通信。`get source`、编译成功、sim 成功、单设备 E2E、分布式成功是不同层次，不相互替代。

## 证据缺口与维护者确认问题

1. `ascend_cicd_a3` 与 MLIR A3 runner 的确切 SKU、逻辑设备到物理卡的映射、可用显存、host CPU/RAM、独占/共享政策是什么？标签含义不能自行解码。
2. AscendC/PTO 的 `tilelang_x1` 如何创建，镜像 digest、CANN/Bisheng、驱动/固件锁定组合、设备挂载与权限是什么？能否提供最小可复现环境清单？
3. 哪些 required checks 真正阻止合并？外部 fork PR 需要谁批准，评论重跑与自托管代码执行的权限边界是什么？本次不评估安全架构。
4. hermetic 测试是否有支持的 CPU-only 安装/运行命令？camodel 支持哪些目标/算子，有无公开 CI 样本及许可/下载限制？
5. shmem/dispatch-combine 被常规 CI 排除后，在哪里做多设备和网络测试？release、长期性能回归如何覆盖？
6. `tilelang-ascend:npuir` 与 `tilelang-mlir-ascend:main` 的维护/迁移政策是什么？主项目已经链接新仓库，但旧 README 链接仍并存。

## 对 tile-rs PTO 路径的条件性参照

若 tile-rs 将来存在明确的纯 host 降低/源码生成接口，可以参照“host 静态/IR/编译检查”和“NPU E2E”分层，降低每次检查都占用设备的需要。若最终依赖 PTO ISA C++ headers + Bisheng，则本项目是可追踪的相邻实践；若选择其他中间路径或 PTO AS，则不能直接复制其工具链要求。真机资源条件应以 tile-rs 实际执行路径、目标设备及维护者给出的版本清单为依据。本报告不选架构、型号、托管平台，不推算卡数或采购预算。

## 固定来源索引

所有源码均采集于 2026-09-22；分支见上表。运行链接可能随上游保留政策失去日志，正文保留了可复核的 job ID、时间和最小摘录。

- [S1 主项目 README](https://github.com/tile-ai/tilelang/blob/8cf0e7eff4593d8c0236bd45e74a5cacb797072f/README.md)
- [S2 AscendC/PTO README](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/README.md)；[仓库元数据 API](https://api.github.com/repos/tile-ai/tilelang-ascend)（动态）
- [S3 MLIR README](https://github.com/tile-ai/tilelang-mlir-ascend/blob/54a19e28dfd8756daa253b387e00d10ba7f952e7/README.md)
- [S4 AscendC/PTO 主 CI](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/.github/workflows/ci_cd.yml)
- [S5 格式 CI](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/.github/workflows/ci_ascend.yml)；[旧禁用链](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/.github/workflows/ci.yml)
- [S6 主项目 CI](https://github.com/tile-ai/tilelang/blob/8cf0e7eff4593d8c0236bd45e74a5cacb797072f/.github/workflows/ci.yml)
- [S7 bench_test.sh](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/examples/bench_test.sh)
- [S8 PTO GEMM](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/examples/gemm/example_gemm_pto_developer.py)
- [S9 A5 仿真模板](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/.agents/skills/tilelang-a5-sim-convert/scripts/run_a5_sim_template.py)
- [S10 MLIR Base CI](https://github.com/tile-ai/tilelang-mlir-ascend/blob/54a19e28dfd8756daa253b387e00d10ba7f952e7/.github/workflows/base_ci.yml)
- [S11 MLIR wheel 与 lit](https://github.com/tile-ai/tilelang-mlir-ascend/blob/54a19e28dfd8756daa253b387e00d10ba7f952e7/.github/workflows/build_tilelang_wheel.yml)
- [S12 MLIR NPU 测试](https://github.com/tile-ai/tilelang-mlir-ascend/blob/54a19e28dfd8756daa253b387e00d10ba7f952e7/.github/workflows/test_npuir_wheel.yml)
- [S13 旧 npuir 分支测试](https://github.com/tile-ai/tilelang-ascend/blob/bd7c60c312b0721885c78d1ce03c62be9225314c/.github/workflows/test_npuir_wheel.yml)
- [S14 NPUIR host 预构建](https://github.com/tile-ai/tilelang-mlir-ascend/blob/54a19e28dfd8756daa253b387e00d10ba7f952e7/.github/workflows/build_ascend_npuir.yml)
- [S15 AscendC/PTO host 安装](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/install_ascend.sh)
- [S16 hermetic 与 NPU 分层测试](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/testing/python/language/test_ascend_compile_flags.py)
- [S17 编译/加载及仿真库选择](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/tilelang/jit/adapter/libgen.py)
- [S18 子模块来源](https://github.com/tile-ai/tilelang-ascend/blob/585ad8c3b5fae175f1d1661059d347645f1f9536/.gitmodules)；[PTO ISA gitlink](https://github.com/tile-ai/tilelang-ascend/tree/585ad8c3b5fae175f1d1661059d347645f1f9536/3rdparty/pto-isa)

## Resolution comment 草稿（待批准，未发布）

已核实：TileLang 主项目的 CUDA CI 与 Ascend 实现分开。AscendC/PTO 有公开 PR/nightly NPU 执行记录；MLIR 实现有独立的 host 构建/IR 测试与 NPU wheel 测试。无卡可做静态及选定 host 检查，但不能替代真机数值、运行时与性能验证。runner 标签和逻辑设备可见性不足以推定物理卡数；容器、驱动/固件及准入缺口已列入报告。研究报告位于本地 `research/tilelang-ascend` 分支的 `docs/research/tilelang-ascend.md`，尚未推送；资产指针见票据正文。
