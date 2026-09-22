# torch_npu 的公开与外部 CI 资源要求

调查日期：2026-09-22。研究票：[核实 torch_npu 的公开与外部 CI 资源要求](https://github.com/Jopqior/tile-rs/issues/52)。范围是公开证据，不评估采购规模，不实施 CI。

## 摘要

- 用户指定的 [GitCode Ascend/pytorch](https://gitcode.com/Ascend/pytorch) 是主仓库；其 README 明确把 GitHub 标成 mirror。GitHub 是官方镜像，但也实际承载面向上游 PyTorch 的外部 CI，不是“镜像所以没有 CI”。本次两端 master 的 HEAD 相同。
- 公开 GitCode PR pipeline 分开了 CPU 静态检查、ARM/x86 构建、普通 UT 和 distributed UT。普通 UT 请求 `npu=1`，distributed UT 请求 `npu=2`。这是单 job 的资源标签，不是设备池数量。
- 官方开发镜像文档明确：TorchNPU 编译无需 CANN/驱动；运行、导入插件及调用 NPU 算子需要运行环境。CPU 构建成功不证明设备执行正确。[S5]
- 找到真实 GitHub NPU 测试 job：`npu-smi`、`torch.npu.is_available() == True`、`device_count() == 8` 和具体 `_npu` 测试 PASS 均在公开日志中。该数字是一个容器内可见的逻辑设备数，不是物理板卡总量，也不是设备池规模。[R1]
- 不可把绿色触发器当测试通过：另一个成功 run 的 build/test 全部 skipped。[R2] 实际测试 workflow 的结果回调也有 best-effort 处理，不能只看总 run 颜色。
- 没有确认 GitCode 当前 PR 的实际执行日志、分支保护所要求的 checks、完整 release 门禁，以及驱动/固件精确配套版本。没有证据表明仿真覆盖了此处真机验证。

## 来源与版本

本次从 GitCode 页面开始，用公开页面、`git ls-remote`、官方 GitHub mirror 源码和 GitHub Actions API/日志交叉核对。当前工具没有 web_search，以公开仓库/API 定向检索替代；未访问内部系统，未在本机运行 NPU 测试。

| 版本 | 采集结果与用途 |
|---|---|
| M：`master`，`b29d5da96474c2c2472bb1062ab7baf7698b3d4a` | GitCode `git ls-remote ... HEAD` 与 GitHub clone HEAD 相同；README、GitCode workflow、安装文档依据此版本。只证明这次快照一致，不保证镜像始终实时同步。 |
| C：GitHub `ci-test`，`70737b0684656fffa6b3958a2527afccf89693fe` | 采集时 branch HEAD；外部 CI 的可重用 workflow 和脚本分支，不应假定等于主库 master。 |
| R：`46ec3a747074660baa1a44d759ab613227d888c6` | R1 API 的 `referenced_workflows` 指定的实际 workflow SHA。R1 job 随后用浮动 `ref: ci-test` checkout 测试脚本，日志显示脚本 SHA 为 C。不能把运行中的脚本误固定到 R。 |

源码引用均为固定 commit permalink：

- [S1 README / 主库与镜像声明](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/README.md)。[GitHub 仓库 API](https://api.github.com/repos/Ascend/pytorch) 的 description 也写着 `Mirror of https://gitcode.com/Ascend/pytorch`；API `mirror_url=null` 不否定该明确声明。
- [S2 GitCode PR pipeline](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.gitcode/workflows/PR-pipeline_pytorch.yml)。虽然平台变量名是 `atomgit`，本报告仍按 `.gitcode` 配置和实际 GitCode 项目入口描述，不从变量名推断另一套部署。
- [S3 GitCode build job](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.gitcode/workflows/build_job.yml)。
- [S4 普通 UT](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.gitcode/workflows/ut_job.yml)、[distributed UT](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.gitcode/workflows/ut_job_distributed.yml)。
- [S5 开发镜像与编译/运行边界](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/docker/devel/README.md)、[Dockerfile](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/docker/devel/Dockerfile)。
- [S6 官方源码安装指南](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/docs/zh/installation_guide/building_from_source.md)。
- [S7 GitHub PR relay 入口](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.github/workflows/pytorch_ci_trigger_pr.yml)、[receive-trigger 实际运行版本 R](https://github.com/Ascend/pytorch/blob/46ec3a747074660baa1a44d759ab613227d888c6/.github/workflows/receive-trigger.yml)。
- [S8 实际运行版本 R 的 test-category](https://github.com/Ascend/pytorch/blob/46ec3a747074660baa1a44d759ab613227d888c6/.github/workflows/_test-category.yml)、[test-collect](https://github.com/Ascend/pytorch/blob/46ec3a747074660baa1a44d759ab613227d888c6/.github/workflows/_test-collect.yml)、[实际 checkout 的测试执行脚本 C](https://github.com/Ascend/pytorch/blob/70737b0684656fffa6b3958a2527afccf89693fe/.github/scripts/run_npu_test_shard.py)。
- [S9 all-reduce 测试](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/test/distributed/test_allreduce.py#L110-L136)、[UT 调度器](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/ci/access_control_test.py#L299-L317)。

## CI 入口、触发、门禁

### GitCode 项目本身的 PR

S2 的 `pr_comment` / `pull_request_comment` 以 `^compile$` 触发，覆盖 master 和所列版本分支。修改内容仅为文档等文件时，可跳过 build/UT。阶段包括 CodeCheck（lintrunner、SCA、恶意代码扫描）、Build、UT。流水线中 `pre-pipeline` / `query-pipeline` 使用 token 并申请更新 PR 标签。配置只能证明声明了这些步骤，无法由此证明每个 PR 都强制执行、谁有触发资格或哪些检查是合并硬门禁。

`check_distribution`、`check_inductor`、测试选择与分片决定是否运行相关 UT；因此普通 PR 不等于完整 distributed 回归。S4 distributed job 还显式移除部分同质测试文件，使用 disabled-tests 配置，不能概括为“所有分布式测试都跑”。

GitCode `/actions` 页面本次只取到前端壳，未取得可核实的某次 run/job 日志。此处对 GitCode CI 的结论限定为**公开配置**，不冒充真实运行证明。

### GitHub 面向上游 PyTorch 的外部 CI

S7 master 接受 `repository_dispatch` 的 `pull_request` 类型，转发至 `ci-test` reusable workflow。R/C 中接收器要求 `closed` 且有 `Merged` 标签才构建；配置解释这是适配 PyTorch MergeBot。它主要是上游合入后的验证，不应凭文件名“PR”称为每次 PR 提交的 pre-merge gate。接收器还定义了 `ciflow/trunk` tag push 分支，但采集时 [Actions workflows API](https://api.github.com/repos/Ascend/pytorch/actions/workflows) 把 `pytorch_ci_trigger_push.yml` 标为 `disabled_manually`，不能把这条代码路径当当前已启用的合入门禁。

接收器在 CPU runner 上构建 PyTorch/torch_npu，再用 NPU runner collect 和按类别分片执行。它通过固定版本的 `pytorch/test-infra` `cross-repo-ci-relay-callback` 向上游回报。S8 回调 `continue-on-error: true`，根据报告的失败计数决定回调 conclusion；测试脚本 exit code 先被保存为 output。因此“workflow 成功”“回调成功”“全部测试通过”必须分别核查。

### Nightly、手动与 release

- [coverage nightly](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.github/workflows/pytorch_coverage_nightly.yml)：声明每日 UTC 19:00 和 `workflow_dispatch`，调用 `ci-test` 的 coverage workflow。采样 API 最新几次记录实际是 `push` 事件，不把这些样本当 schedule 已成功执行的证明。
- [manual](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.github/workflows/manual.yml)：手动传入 runner、devices、image；调用 [旧 build-and-test](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.github/workflows/_build-and-test.yml)，构建后执行 `ci/access_control_test.py`。
- [periodic](https://github.com/Ascend/pytorch/blob/b29d5da96474c2c2472bb1062ab7baf7698b3d4a/.github/workflows/periodic.yml) 虽有定时声明、`self-hosted` 和 `/dev/davinci6`，API 状态是 `disabled_manually`。设备索引 6 不能证明当前池内有七张卡，更不能代表采购需求。
- 有历史 workflow 名称/运行记录并不意味着其仍是当前门禁。未确认完整 release 验证、发布前的设备矩阵或必过 checks。

## 按阶段的 NPU 必要性与证据矩阵

| 阶段 | NPU 必要性 | 证据等级与边界 |
|---|---|---|
| 文档差异检测、lint、SCA | 不需要设备执行 | S2 在 `[self-hosted, default, x86, cpu=16]` 声明这些任务；是配置事实，不是实际 host 盘点。 |
| torch_npu wheel 编译 | 不需要真机；官方 builder 连 CANN/driver 也不含 | S5 明确陈述，S3 分 ARM/x86 请求 CPU 标签。编译依赖 CPU PyTorch、工具链与源码/子模块。没有本次本地构建实测。 |
| 插件 import、安装后的 `.npu()` smoke | NPU 运行栈；实际设备运算需真机 | S5 指出 import/运行依赖 CANN、driver；S6 安装验证创建 NPU tensor。不能以“能生成 wheel”替代它。 |
| 普通算子/core UT | 请求单 NPU 的配置已公开 | S4 `npu=1`；是否所有用例都单设备、缺卡是否 skip，须按用例筛选。 |
| distributed/HCCL UT | 至少部分用例要求 2 个可用 NPU | S4 `npu=2`；S9 `skipIfUnsupportMultiNPU(2)`、`ranks=[2]`，初始化 `backend='hccl'`。仅有一张卡不能验证此类通信。 |
| 外部 PyTorch NPU collect | 该实现放在 NPU runner 上 | S7/S8 及 R1 collect job；不要笼统假定 pytest collect 必然 CPU-only。 |
| 外部 PyTorch core/tensor/graph 执行 | 真实 NPU 日志已核实 | R1 验证可见设备、运行 `_npu` 用例。该样本没有 distributed job，不证明外部全套多卡通信回归已执行。 |
| 仿真 | 未确认能替代上述阶段 | 未找到此证据链上的仿真 workflow 或模拟器结果，不声称项目完全没有仿真能力。 |

**特别区分数量含义：** S2 的 `TEST_WORLD_SIZE=3/4` 最终进入 S9 的按时间/轮转测试分片，不是 HCCL 通信 world size。S9 all-reduce 中 `world_size=2` 才是该用例的通信进程数。GitCode `npu=1/2` 是调度标签；GitHub runner 名字中的 `-8-` 是标签字符串；R1 `device_count=8` 才是该次进程可见逻辑 NPU 数。四者都不能相加推导设备池规模。

## 公开运行证据

### R1：确有 NPU 测试执行

[run 35723340208](https://github.com/Ascend/pytorch/actions/runs/35723340208)，2026-09-22，`repository_dispatch`，标题 `PyTorch CI Trigger PR - PR 198007`，head SHA=M。可用 [run API](https://api.github.com/repos/Ascend/pytorch/actions/runs/35723340208) 与 [jobs API](https://api.github.com/repos/Ascend/pytorch/actions/runs/35723340208/jobs?per_page=100) 复核。

- build job 使用 `linux-aarch64-cpu-24`；collect 使用 `linux-aarch64-a3-800i-2-cn12-001`；core/tensor/graph shard 使用 `linux-aarch64-a3-800i-8-cn12-001`。这些是观测到的 runner labels，不把标签编码当供应商硬件规格证明。
- [core job 106741416058](https://github.com/Ascend/pytorch/actions/runs/35723340208/job/106741416058) 的日志在 UTC 12:26:22 打印 `npu-smi 26.0.rc3`、`Version: 26.0.rc3`、设备名称 `Ascend910`；12:26:27 打印 `NPU available: True`、`NPU count: 8`。
- `npu-smi` 表里 NPU ID 4、5、6、7 各出现 chip 行，进一步说明不宜把 8 个逻辑设备直接叫作 8 张物理板卡。精确板卡 SKU/拓扑仍未知。
- 12:28:52 的结果摘要为 `2476 passed, 0 failed, 0 errors, 0 timeout, 418 skipped, 2894 total`。日志含具体 `_npu` 用例 PASS，例如 `test_conv_backend_empty_batch_channel3d_has_bias_True_strided_True_contiguous_False_npu`。这支持“实际运行部分 NPU 测试”，不支持“全部用例通过”或“整个产品矩阵验证完成”。
- 容器 tag 含 `aarch64-cann-a3-py3.10-torch-master-cann9.2.0-20260910200430-202609140905-0ba22e2b`。它是观测到的配置/tag，不是通过安装清单核验的全套 CANN、ops、NNAL、驱动、固件版本。日志的 npu-smi 版本也不能独立填满该配套矩阵。

### R2：成功但未执行测试

[run 35728517112](https://github.com/Ascend/pytorch/actions/runs/35728517112) 总体 success，[jobs API](https://api.github.com/repos/Ascend/pytorch/actions/runs/35728517112/jobs) 显示只有 `forward / extract` 在 `ubuntu-latest` 成功，build/test skipped。因此 README badge 或 trigger 成功列表不足以证明真机测试。

## 公开可证实的资源条件

| 项目 | 观察事实或官方要求 | 不能据此推出的内容 |
|---|---|---|
| Host 架构/CPU | S3 构建分别 ARM/x86，默认 `cpu=32`；S2 lint 为 `cpu=16`；R1 build label 为 `linux-aarch64-cpu-24`。S6 列 x86_64/AArch64 编译环境。 | 不把不同路径 CPU 标签统一成最低 CPU 要求；不知道物理核、内存和池容量。 |
| NPU 型号 | S6 支持表含 Ascend 950DT、Atlas A3/A2 训练系列及 Atlas 训练系列。R1 日志为 Ascend910，runner tag 含 a3/800i。 | 支持列表不是 CI 实际覆盖型号；tag 不是精确 SKU 证明。 |
| NPU 数量 | S4 请求单 NPU 与双 NPU job；S9 双设备 HCCL 用例；R1 有 8 个逻辑设备可见。 | 不推物理板卡数、同时可借资源数、机器数或采购规模。 |
| CANN/驱动/固件 | S1 二进制安装示例 CANN 9.1.0；S5 dev 默认 9.1.0/910b，驱动不含在镜像中；S6 要求匹配硬件的驱动固件及 Toolkit/ops/NNAL。R1 image tag 标 CANN 9.2.0。 | 示例与外部 CI 是不同版本路径；没有统一精确 driver/firmware 版本证据，不拼成一套已验证 BOM。 |
| 容器 | S5 builder 无 CANN，dev 有 CANN；R1/S8 测试容器来自 SWR，`--network host`。旧 build-and-test 明确挂载 driver lib、dcmi、npu-smi 和 davinci 管理节点。 | 容器不自带宿主机驱动，不因镜像公开就获得设备。新 runner 的完整设备注入机制未公开确认。 |
| 权限 | S5 给出 privileged 的方便用法，同时明确可用逐个 `--device` 的较小权限方案。S2 checkout 允许 unsafe PR，需 `GIT_TOKEN`、OIDC；S8 OBS 读源码/产物用 `ACCESSKEY`/`SECRETACCESSKEY`，回调用 OIDC。 | privileged 不是所有路径的必要最小权限；不能假定任意 fork/外部 PR 能拿到这些 secret 或 self-hosted runner。 |
| 网络 | 配置访问 GitCode/GitHub、SWR、OBS、PyPI/华为/清华源；S8 用 host network 并向上游 relay 回报。 | 不能推断已经有离线镜像、公共匿名 OBS 访问、跨机 HCCL 网络带宽或 RDMA 拓扑。 |

## 无 NPU 能做什么、不能验证什么

可以做文档/静态检查、生成与编译 wheel、检查接口与打包依赖；S5 为“不含 CANN 的 host builder”提供了比 self-hosted 标签更直接的官方依据。外部 CI 的 CPU build 与设备 test 分离也给出实践参照。

不能以这些结果证明 `.npu()` 执行、算子数值正确性、设备内存/stream 行为、HCCL 多卡通信、性能或特定型号适配。一个 NPU 的 smoke/算子测试不能替代 S9 的双设备通信测试；有八个可见设备的普通测试 job 也不能证明已跑多机分布式测试。插件 import 不是当然安全的纯 CPU 检查。

## 公开证据缺口与需维护者确认的问题

1. GitCode `compile` 的触发授权、实际执行样本、分支保护 required checks、文档跳过策略如何影响合入门禁？
2. `npu=1/2` 指逻辑设备、芯片还是板卡资源？runner 标签如何对应设备隔离和具体 SKU？外部 label 的资源语义是什么？不询问或推断采购数量。
3. 哪些 distributed 用例需要 2、更多设备或多机？缺卡 skip 是否会被当作合格？本次外部样本无 distributed job，需另取真实运行记录。
4. 指定发布分支的 CANN Toolkit/ops/NNAL、驱动、固件、内核与 OS 精确兼容矩阵是什么？公开镜像是否可匿名拉取，是否能固定 digest？
5. 新 GitHub runner 的 NPU 注入、host-network、安全隔离、fork PR 准入与 secret 管理由谁负责？OBS/relay 是否能向外部贡献者开放？
6. GitHub relay 回报在上游属于 advisory、post-merge 还是 required gate？release/nightly 的完整覆盖、跳过列表和日志保留策略是什么？
7. 是否有官方认可的仿真替代阶段，以及其明确不覆盖的设备语义？本次未取得这类证据。

## 对 tile-rs PTO 路径的条件性参照

只有在 tile-rs 的实际中间产物和 PTO 工具链边界明确后，才能映射测试资源。torch_npu 提供的参照是：把 host 检查/编译、设备 smoke/数值测试、通信/多卡测试分别列证据和失败条件；它不证明 PTO 编译器本身需要 torch_npu/CANN，也不替 tile-rs 选择架构、NPU 型号或 runner 平台。

若 tile-rs 只覆盖单设备 kernel，torch_npu 的 distributed 配额不能直接移植；若未来确有跨设备语义，再针对那一阶段确认资源条件。公开配置、runner 标签和单次可见设备数均不能用来估算共享设备池或采购规模。

## Resolution comment 草稿（未发布，待批准）

已核实 GitCode 主库与 GitHub 官方镜像关系，并找到 GitCode PR 的单 NPU/双 NPU job 配置，以及 GitHub 外部 CI 的真实 NPU 测试日志。官方文档明确区分无 CANN/驱动的 host 编译与设备运行；成功的 trigger run 也可能完全跳过测试。报告区分了配置、实际运行、单卡和分布式要求，没有由 runner 标签或可见设备数推导设备池规模。GitCode 实际门禁、精确驱动固件配套和外部 runner 准入仍需维护者确认。研究报告及未推送的本地提交见本票资产指针。
