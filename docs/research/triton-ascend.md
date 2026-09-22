# triton-ascend CI 测试分层与 NPU 资源依赖

调查日期：2026-09-22（UTC）。票据：[核实 triton-ascend CI 的测试分层与 NPU 资源依赖](https://github.com/Jopqior/tile-rs/issues/50)。仅调查公开证据，不执行设备测试、不估算采购规模。

## 摘要

- 当前官方开发仓库是 `triton-lang/triton-ascend`。旧 `Ascend/triton-ascend` 的描述仍称 GitCode 镜像，但其 README 已明确迁移；不能把旧镜像的 Actions 当作当前 CI 全貌。[S1]
- 当前存在三条不同链路：CPU 静态/构建检查、GitHub Actions 直接运行 NPU pytest、CPU runner 上传 wheel 并委托外部 Blue/Yellow 流水线。最后一条已有公开成功回调，但不公开其设备执行细节。[S2–S7,R1,R2]
- 真机测试不是仅有脚本：公开运行 R1 同时包含 `npu-smi` 设备表、NPU pytest 用例通过记录和汇总。外部运行 R2 则只能确认委托任务被接收并返回成功，不能据此确认测试覆盖或设备池。
- 编译器构建、MLIR/FileCheck、compile-only 与设备执行需分开看。测试叫 unittest 不代表不需要设备；公共 fixture 会调用 `torch.npu.device_count/set_device`。[S8]
- 采集到的 main 配置、PR 实际执行的 merge commit 和 feature 分支配置并不相同。尤其 R1 的 MLIR 测试是 non-blocking，main 当前对应步骤是阻断失败的普通步骤。不能用绿色 run 推断所有 MLIR 测试成功。[S3,R1]

## 来源与版本

源码引用均固定 commit。以下编号链接在全文复用：

| 编号 | 来源、分支与版本 |
|---|---|
| S1 | 旧仓库 `main`，`865691e2e9b656bc58008170207b4108d92e8dd1`：[README 迁移声明](https://github.com/Ascend/triton-ascend/blob/865691e2e9b656bc58008170207b4108d92e8dd1/README.md)、[旧贡献指南](https://github.com/Ascend/triton-ascend/blob/865691e2e9b656bc58008170207b4108d92e8dd1/CONTRIBUTING.md)。旧指南混有 GitCode、GitHub、master/main，作为历史流程证据而非当前统一契约。 |
| S2 | 官方仓库 `main` 快照 `a9cf13bf67580a5054f3fe38f6cad367e7306941`：[CI 入口](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/ci.yml)、[runner matrix](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/runner-preparation.yml)。以下 S3–S11 同一 main 快照。 |
| S3 | [NPU integration workflow](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/integration-tests-ascend.yml) |
| S4 | [Pre-Commit](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/pre-commit.yml) |
| S5 | [Wheels](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/wheels.yml) |
| S6 | [Ascend950 外部 CI](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/Ascend950-ci.yml) |
| S7 | [DynamicCVPipeline CPU precheck](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/.github/workflows/DynamicCVPipeline-ci.yml) |
| S8 | [NPU fixture](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/third_party/ascend/unittest/conftest.py)、[MLIR lit 配置](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/third_party/ascend/unittest/lit.cfg.py) |
| S9 | [安装指南](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/docs/en/installation_guide.md) |
| S10 | [性能分析与仿真说明](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/docs/en/debug_guide/profiling.md) |
| S11 | [compile-only 文档](https://github.com/triton-lang/triton-ascend/blob/a9cf13bf67580a5054f3fe38f6cad367e7306941/docs/en/environment_variable_and_compiler_options_reference.md) |
| R1 | [Integration Tests run 35722034779](https://github.com/triton-lang/triton-ascend/actions/runs/35722034779)，2026-09-22；PR head `7cf04218a089994319759b063053aac054088656`，日志 checkout 实际 merge SHA `95219824477b7054026baf7c7da6d928be975af7`。[运行版本的 workflow](https://github.com/triton-lang/triton-ascend/blob/95219824477b7054026baf7c7da6d928be975af7/.github/workflows/integration-tests-ascend.yml)。 |
| R2 | [Ascend950 CI run 35575538513](https://github.com/triton-lang/triton-ascend/actions/runs/35575538513)，2026-09-21，`pull_request_target`，base 分支 `feature/simd-simt-compile-mode`，API head SHA `7d163d111a801ac198dacec20173c36527a32432`；日志上下文 PR 2192。[对应版本 workflow](https://github.com/triton-lang/triton-ascend/blob/7d163d111a801ac198dacec20173c36527a32432/.github/workflows/Ascend950-ci.yml)。注意此事件的 API head SHA 不能当作被测试 PR head。 |
| R3 | [Pre-Commit 成功运行](https://github.com/triton-lang/triton-ascend/actions/runs/35728011265)，2026-09-22，PR head `8ecf76729de6000f4ca10eeb0796f0daced81ba6`。 |
| R4 | [公开分支 ruleset](https://github.com/triton-lang/triton-ascend/rules/12500360)，[API](https://api.github.com/repos/triton-lang/triton-ascend/rulesets/12500360)，2026-09-22 采集，属于可变配置。 |

方法：读取官方仓库、GitHub REST、Actions 完整日志及 GitCode PR 服务端页面；当前工具无 web_search，未以搜索摘要代替证据。没有进入需要凭据的外部后台。日志由 `gh run view --log` 读取，报告保留必要摘录而非复制大量日志；公开日志可能过期。

## CI 入口、触发与门禁

### 当前 GitHub 路径

1. **Integration Tests（S2/S3）**：手动、符合路径过滤的 PR、main/release/**/**-dev push，以及这些分支的 merge_group。PR/手动启用 matrix；push 在构建依赖变更或距离最后成功运行至少四小时时启用。不是每天定时 NPU nightly。时间查询用了固定 workflow ID，不能把条件逻辑当作已经验证的调度 SLA。
2. **merge_group 特别限制**：入口有该事件，但 runner-preparation 只有 PR、手动和 push 的 enable 分支；在此快照下不能假设 merge queue 一定生成 NPU matrix。
3. **Pre-Commit（S4）**：PR、push、merge_group、手动，在 ubuntu-latest 执行格式检查。R3 是实际成功记录。
4. **Wheels（S5）**：每日 `0 8 * * *`、release/** push、特定路径 PR、手动。CPU 构建矩阵是 x86_64/aarch64、cp310–cp313、main/release/3.2.2。非 PR 上传 nightly OBS；这个 nightly 证明打包链路配置，不证明 nightly 真机验收。
5. **DynamicCVPipeline（S7）**：对应源码和测试目录 PR、手动；CPU runner 构建 base/PR，设置 `TRITON_COMPILE_ONLY=1` 生成中间产物并上传。不能把 compile-only 叫仿真执行。
6. **Ascend950 CI（S6）**：此快照仅对 `feature/simd-simt-compile-mode` 的匹配路径 `pull_request_target` 启动。尽管步骤中保留 push 分支逻辑，`on` 并没有 push，不能按注释扩大触发范围。先 CPU wheel，再外部流水线。

**门禁强度分三层**：S3 里的 pytest 非零退出会使 job 失败；R1 的 MLIR 步骤却显式 `continue-on-error: true`；是否阻止合并还需仓库保护设置。R4 active ruleset 公开要求两人批准、解决 review thread、线性历史等，但该返回中没有 `required_status_checks`。classic branch-protection API 返回 404，无法据此断言不存在其他限制。不能把“有 CI”写成“这些测试均为强制合并门禁”。release 分支触发构建也不等于正式发版前设备验收；未核实完整 release 审批链。

### 历史 GitCode/Jenkins 与当前外部 Blue/Yellow

旧贡献指南 S1 记载 Jenkins 自动创建 PR 流水线、评论 `/compile` 启动/重试、成功加 `ci-pipeline-passed`；静态检查 `compile#openlibing`、成功加 `SC-SUCC`，随后请求 committers review。

实际读取 [GitCode PR 891](https://gitcode.com/Ascend/triton-ascend/pull/891) 页面，可见 `SC-SUCC`、`ci-pipeline-passed` 和 [OpenLiBing 详情入口](https://www.openlibing.com/apps/transitionPrDetail?owner=Ascend&repo=triton-ascend&number=891&platform=gitcode&codeHostingPlatformFlag=gitcode)。OpenLiBing 响应仅得到前端应用壳，未取得任务日志、设备型号或测试清单。因此只确认历史外部检查入口和标签，不能证明该 PR 真机执行，更不将它等同于当前 GitHub CI。

当前 S6 是更明确的外部链路：

- `Wheels Build` 在 `linux-amd64-cpu-16` 构建 wheel，构建 job 不传外部服务 secrets；`Pipeline Tests` 在 `linux-aarch64-cpu-1` 下载 artifact 并上传 OBS。
- 用 `BLUE_YELLOW_BASE_URL/APP_CODE/APP_KEY/APP_SECRET`、`YELLOW_PIPELINE_ID/GROUP_ID` 调用 `/start` 和 `/query`。请求带 PR、仓库、分支、OBS wheel URL。控制 runner 的 CPU 标签不能代表远端执行设备。
- R2 [job 106259461630](https://github.com/triton-lang/triton-ascend/actions/runs/35575538513/job/106259461630) 日志：08:12:04 UTC `/start` 返回 `code:200`, `MQS send success`；08:54:06 第 42 次轮询返回 `status:SUCCESS`、`任务结束，运行成功`。不是只有脚本而没有运行记录。
- 回调给出 [CloudDragon 任务](https://clouddragon.huawei.com/pipeline/details/a086933980b040fea000837da6a6ccb9?jobId=fa55ace2600744f5aa2ae3f06940a1bf&welink_open_uri=no_mobile)。一次有限超时探测未取得可读详情，不继续追逐内部系统。成功回调不包含卡数、CANN、pytest 结果或 `npu-smi`，故远端是否真机、具体测试覆盖仍未知。

## 分阶段 NPU 必要性与证据矩阵

| 阶段 | 无 NPU 可否做 | 配置/脚本 | 公开执行证据与边界 |
|---|---|---|---|
| 格式与静态检查 | 可以 | S4，普通 Ubuntu + Python | R3 成功。不是算子正确性验证 |
| host 编译/wheel 打包 | 可以，不等于不需 Ascend 软件依赖 | S5/S6/S7 CPU runner；LLVM/C++、容器与依赖下载 | R2 Wheels Build 成功，job 106256446605 为 CPU 标签。NPU integration 的 build 步骤另有 `npu-smi`，不能原样移到无设备环境 |
| MLIR pass / FileCheck、C++ 编译器单测 | 原理上 host 可做，需先构建工具 | S8 lit 只注册 triton-opt/FileCheck；S3 ctest + lit | R1 实际运行 MLIR，但旧步骤只筛 DynamicCVPipeline、允许失败，日志有 FileCheck 错误。当前 main 更广的 blocking 步骤未以该 run 验证 |
| DynamicCVPipeline compile-only | 已配置 CPU 路径 | S7 `TRITON_COMPILE_ONLY=1` | 不运行算子，不证明输出正确性或真实性能；本研究未逐项审计所有 compile-only 用例 |
| pytest_ut/autotune_ut 整套 | 需要可用 NPU | S3/S8；fixture 设置设备，无设备时不能照搬整套 | R1 是直接设备执行证据，见下节。个别纯逻辑用例不等于整个目录可无 NPU 执行 |
| kernel 专项测试 | 测试代码存在不等于 CI 启用 | S3 注释掉 `kernels/test_triton_kernel.py`，称间歇失败 | 不计入当前启用覆盖 |
| 性能仿真 | 文档有工具路径，独立无设备条件未证明 | S10 `msprof op simulator --soc-version=...`，设置 CANN simulator 库路径 | 是本地 profiling 指南，不是本次审计到的 CI job；不可宣称仿真已替代真机 CI |
| 外部 Ascend950 pipeline | 本地控制/打包不需 NPU；远端需求未知 | S6 Blue/Yellow 请求与轮询 | R2 证明外部委托完成，不证明具体设备执行 |
| 多卡/分布式通信 | 本研究未证实专门测试覆盖 | S3 开启 `TRITON_BUILD_TD`；S8 多 worker 按设备数取模 | 构建 TD 和把独立用例分散到设备都不等于运行集合通信、多机网络测试 |

## 可证实的资源条件

### A. 观察事实：R1 真机运行

[py3.10 job 106727021017](https://github.com/triton-lang/triton-ascend/actions/runs/35722034779/job/106727021017) 与 [py3.11 job 106727044485](https://github.com/triton-lang/triton-ascend/actions/runs/35722034779/job/106727044485) 都成功；API labels 为 `linux-aarch64-a3-800i-4` 和 `cn12-001`。runner 名分别以 `...4njqz`、`...djp8j` 结尾。job 显示名仍是旧 `linux-aarch64-a3-4`，应优先保留 labels 与设备日志，不按名字猜设备。

- py3.10 `npu-smi` 在 11:35:06 输出 `npu-smi 26.0.rc3 Version:26.0.rc3`，Name 为 `Ascend910`；NPU 编号 6、7，各列 Chip 0、1，Phy-ID 12–15，每项 HBM 总量 65536 MB。
- py3.11 11:38:37 输出 `npu-smi 25.5.1 Version:25.5.1`，NPU 编号 0、1，各列 Chip 0、1，Phy-ID 0–3，每项 HBM 总量 65536 MB。
- 这是**每个 job 可见的两个 NPU 编号、四个 chip/Phy-ID 条目**，不擅自转换成四块物理卡，也不把两个 job 相加估算设备池。标签末尾 `-4` 不是独立卡数证明。仅日志 Name 不能进一步精确到某个 910 SKU。
- 日志工具版本不等于已独立核实的驱动包/固件版本；两 job 的版本差异也提示不能把某 runner 标签当作统一软件镜像/物理主机证明。
- py3.10 的 pytest_ut 摘录：`3127 passed, 108 skipped, 4 xfailed, 2 xpassed`；autotune_ut：`82 passed, 4 skipped`。py3.11 同样 3127 passed，但 xfail/xpass 数不同。这些只是本次运行计数，不是当前 main 的覆盖承诺。
- `npu-smi` 设备健康/温度/HBM 信息、实际 `PASSED ...test_atomic_add...` 等用例记录、NPU fixture 和运行步骤合在一起构成直接设备执行证据；不是单凭 runner 名下结论。

### B. 当前配置事实，不冒充已抽样执行

S2/S3 配置 A3 aarch64、CANN 8.5.0、Ubuntu 22.04、Python 3.10/3.11，以及 `linux-amd64-a5-4` 对应 Ascend 950、CANN 9.1.0、Python 3.12。容器为：

`swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/triton:<cann>-<chip>-<os>-py<python>-latest`。

镜像是浮动 latest 标签，不是 digest 锁定的环境合同。S3 按 `uname -m` 选 aarch64/x86_64 Bisheng 安装包并下载校验；NPU job 同时构建编译器和执行测试。R1 只抽样证明 A3 两个 Python job，不拿它证明当前 950 matrix 已运行成功。

网络涉及 GitHub/Actions artifact/cache、华为 SWR、OBS Bisheng 包、华为 PyPI 和 `cache-service.nginx-pypi-cache.svc.cluster.local` 内部 HTTP 缓存。最后一个是集群内部名称，外部 fork 无法假设可直连。S6 另外要求 OBS 写凭据和 Blue/Yellow API 凭据。公开 YAML 没有完整 NPU 设备挂载/runner Kubernetes 部署契约，runner 使用权、fork 准入、队列、配额及网络白名单均需询问维护者。

### C. 官方安装建议，不是 CI 实际库存

S9 支持 Atlas A2/A3/950，推荐每卡至少 32 GB；软件例子 CANN 9.1.0、Python 3.11、TorchNPU 2.7.1.post8，驱动/固件另指向 CANN 安装文档。可选 `TRITON_BUILD_NPUIR=ON` 要求 CANN 环境和大于 30 GB 可用磁盘。

S9 Docker 示例映射 `/dev/davinci0` 至 `7` 及管理设备、挂载 host driver/npu-smi，使用 root、seccomp=unconfined、`--shm-size=512g`。这是使用示例，不是最低八卡要求、512 GB 实际占用或 CI 安全基线；不能据此推算采购规模。支持型号清单也不是实际设备池。

## 无 NPU 能做与不能验证什么

可以有证据地把 host 构建/打包、格式检查、部分编译期/IR 检查单独理解为无设备阶段。工具链依赖、网络下载、CANN/Bisheng 软件获取仍需准备，不等于只需裸 CPU 和 Rust 工具链。MLIR 在 NPU job 中运行是当前编排选择，不代表 FileCheck 本身要 NPU。

没有 NPU 不能照搬当前整套 pytest fixture，不能确认设备 kernel 输出、精度/同步/内存访问、真实吞吐或驱动 runtime 兼容性。compile-only、仿真 profiling、外部 SUCCESS 回调分别验证不同事情，不能互换。

## 证据缺口与需维护者确认的问题

1. 哪些 checks 真正是 main/release 的 required checks？merge_group 为什么不显式启用 matrix？分支间 non-blocking MLIR 策略如何对齐？
2. Blue/Yellow/CloudDragon 执行什么测试、用真机还是仿真、设备型号和逻辑/物理划分是什么？能否公开脱敏 job manifest、pytest/JUnit 和 npu-smi/驱动/固件记录？
3. A3 与 950 runner 的挂载、隔离、设备共享规则、完整 CANN/Bisheng/TorchNPU/驱动/固件组合及不可变镜像 digest 是什么？同 labels 的 npu-smi 版本差异如何解释？
4. 测试独占性、CPU/RAM/磁盘最低条件、外部贡献者 runner 准入和网络访问流程是什么？目前不能从标签数字推导规格。
5. 有没有独立分布式/通信、性能回归、仿真 CI，以及 release 前真机验收？当前样本不足以回答；kernel 专项被禁用造成的缺口如何补足？

## 对 tile-rs PTO 路径的条件性参照

若 tile-rs 后续确认存在可独立运行的 PTO 编译/IR 校验步骤，可以参照这里的 host 构建与设备执行分层，并分别记录软件依赖与 NPU 访问条件。若还要验证算子结果、runtime 或真实性能，需要额外获得设备执行证据；不能用编译成功替代。

本调查不证明 Triton 的编译链等同 PTO ISA/PTO AS，不选择 tile-rs 的中间路径，不推荐具体设备、托管商或卡数。可向上反馈的已知边界是：同行既有 CPU 编译路径，也有可公开核实的设备测试；外部 CI 凭据、网络和执行环境是独立依赖，必须单独确认。

## Resolution comment 草稿（未经批准，不发布）

已完成调研。当前官方仓库是 triton-lang/triton-ascend，公开 CI 同时包含 CPU 构建、直接 NPU pytest 和外部 Blue/Yellow 流水线。已找到设备日志与用例结果；外部流水线只有成功回调，远端设备与覆盖仍未公开。报告区分 main 配置、实际运行版本、门禁和资源未知项，仅作 tile-rs PTO 路径的条件性参照。报告保存在本票资产指针所列本地分支，尚未推送。
