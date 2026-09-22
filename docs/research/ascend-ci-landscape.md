# 其他 Ascend / PTO CI 项目：广度筛选与强候选初查

采集日期：2026-09-22。研究票：[筛选其他可借鉴的 Ascend / PTO CI 项目](https://github.com/Jopqior/tile-rs/issues/54)。范围是资源依据，不实施 CI，不选择 tile-rs 编译架构，不估算采购规模。

## 摘要

- 最贴近 PTO 编译链的新增候选是 **hw-native-sys/pypto、pypto-lib**。PyPTO 自称社区实现，不能因名称或组织名就认定为华为官方项目。公开配置、运行 job 和部分日志能串起 CPU 单测、codegen-only、仿真、真机四层证据。[P1][P2][P3][R1]
- **CATLASS** 是值得保留的 Ascend C Tile 算子模板参照：官方仓库明确提供不依赖 NPU 的 host stub 单测；但本次没有取得它的公开 CI 调度与真实运行记录，不能把可执行测试当成已接入 PR 门禁。[C1]
- **AscendNPU-IR** 是相关性较高的编译器候选。镜像可见 tag wheel 构建和 CANN 编译器依赖，另有 NPU 端到端示例；二者不是同一份“设备 CI”证据。[I1][I2]
- **FlagGems** 有明确的 Ascend backend、NPU 检查脚本、910B weekly 容器配置，故纳入，而不是拿其其他 GPU 后端凑数。[F1][F2][F3]
- **Ascend-CI** 提供最清楚的一组公共设备运行初查：Liger Ascend 测试 job 日志显示两条 910B3 设备记录和真实 pytest 执行。另有 llama.cpp CANN 测试，但算子失败被汇总后继续，不能以 workflow 绿色等同所有算子通过。[A1][A2][R3]
- **vllm-ascend** 只作 CI 运营层参照，图编译和 PTO 相关性弱于上述候选；其 CPU/NPU 分层和 PR 标签准入值得参考，不能把推理服务多卡需求转成 tile-rs 需求。[V1]

## 来源、版本与方法

没有可用的 web_search 工具，本次使用 GitHub repository search、Ascend 组织仓库枚举、项目 README 指向的 canonical 仓库、公开 Actions API 和浅克隆。搜索角度包括 `Ascend compiler`、`Ascend operators`、`PTO Ascend`、`Ascend CI`、`pypto`、`ascendc`、`cann ops`、`catlass`、`ascend compiler in:readme`。同时检查 AKG、TorchAir、OpPlugin 和 samples，未把用户原列三项当作完整清单。搜索排序偏向 GitHub，GitCode 项目可能漏检；这是一轮有边界的筛选，不是 Ascend 生态完整目录。

以下 commit 为采集时指定分支的快照，所有源码引文固定到该快照。运行记录可能使用较早 commit，单列，不声称与当前源码完全一致。

| 项目 / 仓库身份 | 分支 | 固定 commit |
|---|---|---|
| [PyPTO](https://github.com/hw-native-sys/pypto)，GitHub 仓库自称 community-driven implementation | main | `8a944cf211a847b9d39d35a38791fd5d5bd2120d` |
| [pypto-lib](https://github.com/hw-native-sys/pypto-lib)，同组织的 PTO kernels / models | main | `133fd2273e96f9da62ec7b87adfd87d728e813cb` |
| [CATLASS](https://gitcode.com/cann/catlass)，直接克隆官方文档所指 GitCode 仓库，不用个人 GitHub 同名仓库 | master | `7a9c18a2c38b8cf1905aacbbf3006db1a9468206` |
| [AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)，GitHub Ascend/AscendNPU-IR API 明示 mirror | master | `8d49415f4d75d53dc983353e49b6c99004b0f17c` |
| [FlagGems](https://github.com/flagos-ai/FlagGems)，旧 FlagOpen URL 重定向至 flagos-ai | master | `1f85ac4ee27a1d2236099600dcbaa90e9068d836` |
| [Ascend-CI](https://github.com/Ascend/Ascend-CI)，Ascend 组织公共 CI 项目 | main | `17946c98dac6db21b4f722ee9b8dd94df33442ca` |
| [vllm-ascend](https://github.com/vllm-project/vllm-ascend)，社区维护的 Ascend 硬件插件 | main | `1ebf3a6c3a887b00bf680a2184b78af7aacae2ac` |
| TorchAir，README 指向 https://gitcode.com/ascend/torchair | master | GitHub 快照 `e1118e8b0fc1b95190ab53b732c0872989550e20` |
| OpPlugin，README 指向 https://gitcode.com/Ascend/op-plugin | master | GitHub 快照 `7c5bd436230b03318b690b8fb769151be7966c61` |
| mindspore-ai/akg，GitHub metadata homepage 指向 Gitee MindSpore | master | `5aa15f3fe4d856eb60cd3b27447e04f5f1fb4dff` |
| Ascend/samples，README 历史版本链接指向 Gitee | master | `fc2aea93e48be8f8cee51d0c6c1b13a08155f83a` |

GitHub 镜像缺少 workflow 或运行数据，只能描述该镜像观察结果，不能代表 GitCode/Gitee 或内部 CI 没有 NPU 测试。

## 候选纳入 / 暂缓 / 排除

| 候选 | 判定与理由 | 证据强度与限制 |
|---|---|---|
| PyPTO | 优先纳入。直接产出 PTO，CI 分层与目标最接近 | 配置、required checks API、CPU/仿真/NPU job、NPU 测试日志均可见；不代表 tile-rs 应选择其 IR/runtime |
| pypto-lib | 纳入，作为 PyPTO 下游算子及模型压力测试 | 真机与仿真拆分，设备数按用例声明；LLM 多卡负载不是基础 kernel CI 最小需求 |
| CATLASS | 纳入无 NPU 测试设计参照；公开 CI 强度低 | 官方 stub UT 文档和脚本明确，未证实公开门禁或 runner |
| AscendNPU-IR | 纳入编译器 host 构建参照 | MLIR/BiSheng 编译器、wheel workflow、NPU VecAdd 示例；未核实完整 NPU CI |
| FlagGems Ascend backend | 纳入算子 CI 参照 | 专属 Ascend runner/config/script，weekly 设备挂载明确；本次抽样未确认 Ascend job 成功记录 |
| Ascend-CI 的 Liger / llama.cpp CANN | 纳入外部 CI 运营参照 | 是明确 Ascend 执行，不泛引 Liger 或 llama.cpp 的 CUDA CI；非 PTO 编译链 |
| vllm-ascend | 次级纳入：标签准入、CPU UT、NPU 精选测试 | 模型服务层资源重，不把它的设备数量用于 tile-rs |
| TorchAir | 保留后续候选，不在本轮深挖 | 编译器相关，但 `build.sh` 只证明有 build/UT/ST 入口和 SDK/stub 路径，不证明外部 CI 真机执行 [T1] |
| AKG | 保留后续候选 | README 明确 AKG-MLIR 接 AscendNPU IR、AKG-AGENT 接 Triton-Ascend；本次未找到足够公开 Ascend CI runner/运行证据 [K1] |
| OpPlugin | 本轮不独立深挖 | 明确为 TorchNPU 算子适配子仓，沿用主仓安装/贡献流程；与另票 torch_npu 研究高度重叠 [O1] |
| Ascend/samples | 暂不作当前 CI 资源主证据 | 有 sample/ST 目录，但所检 README 配套表较旧；脚本存在不证明当前持续门禁 [S1] |
| 个人 cann-ops、catlass 镜像及算子生成 agent | 不用作官方 CI 证据 | 身份和同步关系不足；CATLASS 已找到 canonical 源。不否定这些项目价值 |
| triton-ascend、TileLang/Ascend、torch_npu、PTO ISA/AS | 已分票，不重复细研 | 本报告仅在 PyPTO 实际调用依赖处记录工具链 pin，不重做其自身 CI 研究 |
| 泛 CUDA/GPU 项目 | 排除泛引 | 只在有专门 Ascend workflow、backend 或运行证据时纳入 |

## CI 入口、触发与门禁

### PyPTO / pypto-lib

PyPTO `ci.yml` 在 main push、面向 main 的 PR、手动触发时运行。文档-only 分类可跳过重型检查；`pre-commit`、`clang-tidy` 位于大部分测试之前。`codegen-tests` 明确 `--codegen-only`；设备系统测试调用 `task-submit` 和 pytest；`system-tests-a5sim` 在 GitHub hosted Ubuntu 容器运行。配置中的注释有历史设计和计划，**以实际 runs-on、needs、命令为准**。例如部分注释讨论未来 CPU runner 动态借卡，当前 YAML 仍是 `npu` runner，不能按计划描述现状。[P1]

2026-09-22 查询 [ruleset 11556890](https://api.github.com/repos/hw-native-sys/pypto/rulesets/11556890) 得到 active，默认分支 required checks 包括 `clang-tidy`、`pre-commit`、`system-tests`、`unit-tests (ubuntu-latest, 3.10)`、`system-tests-a5sim`、`dist-system-tests`、`pypto-lib-model`、`codegen-tests`；不包括全部 workflow job。规则是采集时的可变配置，不是源码承诺，仍有 GitHub skip 语义及规则绕过权限影响。

Daily CI 在 UTC 18:00 及手动触发：A5 系统测试和 Qwen3 性能采样。性能 step 标记 `continue-on-error`，guard 是 warn-only，所以 nightly 绿色不能证明没有性能回退。[P2]

pypto-lib 的 PR/push/manual CI 有 CPU 单测、按改动选择的 `a2a3sim/a5sim` 和 `a2a3`；部分模型 PR 另外触发 serving。`# ci: no-sim` 用例被仿真 job 跳过；设备 job 用 `# ci: devices=N` 申请数量，默认一设备。serving 特例和 daily 的更大 world size 不能当作所有算子必需。本轮未查询其 branch protection。[P3]

### 其他候选

- AscendNPU-IR：x86_64 wheel workflow 仅 tag push 触发，Ubuntu hosted + manylinux，下载 CANN 9.1.0 并提取 BiSheng，再构建编译器和 Python wheel。不是已证明的 PR NPU 门禁。aarch64 workflow 也存在，本轮不据文件名推断实际设备池。[I1]
- FlagGems：`unittest.yaml` 从 `.github/backends.json` 生成后端矩阵；Ascend 使用 `ascend-cann900` / `cann9`，调用 `backend-test.yaml`，setup 检查设备后执行 `tools/test-op.sh`。weekly 为周三、周六 UTC 13:30 或手动触发，910B 配置独立于其他厂商。未核实 required checks。[F1][F2][F4]
- Ascend-CI Liger：自身 workflow 文件改动的 push/PR、手动、UTC 08:00/22:00 定时；先 Ubuntu checkstyle，再 NPU transformers / benchmarks。checkout 是 `linkedin/Liger-Kernel` 的 `refs/pull/1392/head`，**不是证明 Liger 主分支的所有 PR 已有 Ascend 门禁**。[A1]
- Ascend-CI llama.cpp：自身 workflow 改动、手动、UTC 12:00 定时。构建 `GGML_CANN=on`，运行后端算子、模型精度及 benchmark。后端算子循环捕获失败后继续并生成支持报告，没有最终失败退出，因此其门禁含义弱于“每个算子必须通过”。[A2]
- vllm-ascend：PR E2E 工作流使用 CPU lint、明确 `num_npus: 0` 的 CPU UT 和 NPU 选择测试。`ready-precise`、`ready-all`、`main2main` 等标签配合 `ci-gate` 控制源码变更是否具备执行条件。另有 nightly/weekly/release 工作流，本轮只确认入口存在，不深入资源池。[V1]

## 按阶段的 NPU 必要性与证据矩阵

证据等级不是优劣排名：配置 C、可执行脚本 S、项目要求 D、公开 job J、日志 L 分开记录。未运行这些项目本身的测试。

| 项目 / 阶段 | NPU 必要性 | 证据及可以证明什么 | 不能据此证明什么 |
|---|---|---|---|
| PyPTO lint / IR unit tests / codegen-only | 不需要真机 | C+J：CPU torch、pytest UT，CPU runner `--codegen-only` [P1][R1] | 不验证 PTOAS 组装、真实 kernel 执行与性能 |
| PyPTO a2a3sim / a5sim | 不需要真机 | C+J：明确 simulator platform、`needs-device: false` 或 hosted runner [P1][R1] | 仿真通过不等于芯片数值/同步/调度/性能完全正确 |
| PyPTO system-tests / direct / A5 | 需要真机 | C+S+J，system-tests 另有 L：设备命令和真实 pytest 结果 [P1][R1] | job success 不等于每项测试都执行，日志有 skipped/deselected |
| PyPTO dist tests | 需要多设备 | C+J：具体步骤 `--device-num 4`、另有 `--device-num 2` [P1][R1] | 不证明跨主机、多节点网络，也不推导池总量 |
| pypto-lib CPU tests / simulator | 不需要真机 | C+J：CPU torch、两个 simulator job 成功 [P3][R2] | 不覆盖 no-sim 标记用例 |
| pypto-lib a2a3 / serving | 真机；依用例单设备或多设备 | C+J：a2a3 成功；serving 特例本次 run 为 skipped [P3][R2] | 不能用该 run 证明 serving 8 卡实际执行 |
| CATLASS Tile stub UT | 明确无需 NPU | D+S：替换 AscendC API，检查模板路由和调用参数 [C1] | stub 不是 NPU 指令仿真，不验证真实计算、带宽与同步；无公开 run 初证 |
| AscendNPU-IR tag 构建 | 所查 job 不使用 NPU | C：host build、CANN 编译器提取 [I1] | CANN 依赖不等于设备依赖，也不证明产物在芯片上正确 |
| AscendNPU-IR VecAdd E2E | 运行需要 NPU | D+S：先 `bishengir-compile` 生成 kernel，再 host wrapper 启动 [I2] | 示例不证明由公开 CI 执行 |
| FlagGems Ascend backend / weekly | 测试需要 NPU | C+S：`npu-smi`、检查可用 HBM、设备挂载 [F2][F3][F4] | runner 名和配置不能证明一次真实成功，也不能证明 8 卡集体通信 |
| Ascend-CI Liger Ascend | 真机 | C+J+L：NPU inventory、实际 pytest [A1][R3] | 两设备可见不等于每个用例需两卡，不等于设备池总量 |
| vllm-ascend CPU UT / NPU E2E | CPU UT 无需；NPU E2E 需要 | C+J：抽样 run 只有 CPU 分支成功，NPU 分支 skipped [V1][R4] | 此次成功不是 NPU 成功证据 |

## 公开可证实的资源条件

下表严格区分「配置/日志观察」和「项目要求」。空白信息记为未知，不从支持列表推断实际设备池。

| 项目 | 配置或运行观察 | 官方/项目文档要求 | 未知 / 适用边界 |
|---|---|---|---|
| PyPTO | runner 标签 `self-hosted,linux,cpu/npu/npu-a5`；预构建矩阵 arm64/x64；examples 仿真当前固定 arm64；真实 system job 使用任务队列。setup 依赖 `CI_CACHE_ROOT`、`CANN_ROOT`、`DEVICE_ID`，设备分支 source CANN；CPU 和部分仿真 host-native，A5 simulator 用 `ghcr.io/hw-native-sys/pypto/github-ci:latest`。日志 PTOAS `v0.65` | README 开发容器要求 driver `26.0.rc1`、Docker 和 Ascend Docker Runtime，允许匿名拉取开发镜像 [P4] | README 开发容器条件不是 CI 驱动盘点；CI 确切固件、每台 CANN/驱动、芯片子型号未完整查证。`latest` 镜像不能视为内容固定 |
| pypto-lib | a2a3 和 sim 配置 arm64，A5 runner 不固定 host arch；HCCL 路径采用 host-native，代码注释说明容器 comm_init 问题。设备数量来自用例标记 [P3] | 本轮不额外推导硬件要求 | HCCL 软件/驱动/拓扑以及多节点网络未核实 |
| CATLASS | 官方文档明确 host stub；gcc 7.5 至 12.0、CMake >=3.16、googletest >=1.14.0 [C1] | 上述是 UT 构建要求 | 无公开 runner、实际 CPU arch、设备池、CI CANN/驱动/固件证据；支持产品列表不作盘点 |
| AscendNPU-IR | x86_64 tag wheel 配置 `manylinux_2_28_x86_64:latest`、CANN 9.1.0 下载，Python 3.10–3.13 wheel 矩阵 [I1] | E2E 文档需安装 CANN、设置 `ASCEND_HOME_PATH` [I2] | 对应设备型号/卡数/驱动/固件和真实执行记录未知 |
| FlagGems | 910B weekly 声明 `/dev/davinci0..7`、`test_gpus: 0..7`，privileged、host pid/ipc 和 driver/add-ons 挂载；镜像为 BAAI harbor 的 `huawei-flaggems-test-910b-flagtree0.6.1rc1-ascend3.5:202608191554`。可用性脚本阈值 30000、120 秒重试、最多 600 秒 [F2][F3] | 脚本阈值是调度条件，不是所有 kernel 的内存最低要求 | 8 个设备编号是配置可见范围，不证明测试要求同步用 8 卡或实际池规模；确切 CANN/驱动/固件、host arch、registry 准入未知 |
| Ascend-CI Liger | 默认标签 `linux-aarch64-a2-2`；CANN 镜像 tag `9.1.0-910b-ubuntu22.04-py3.12`；privileged + host IPC + manager/devmm/hisi 设备挂载；运行日志显示设备 0/1 为 910B3，`npu-smi 25.5.2 Version:25.5.2` [A1][R3] | 当前 workflow 安装 torch/torch_npu 2.9.0、triton-ascend 3.2.2 | npu-smi 的版本字符串不另认作固件版本；具体驱动包/固件未知。需可访问 SWR、PyPI/CPU torch 和 triton-ascend 软件源 |
| vllm-ascend | CPU runner `linux-amd64-cpu-8-hk`，CPU UT 配置 CANN 镜像仍是 `num_npus:0`；workflow 有集群内部 PyPI cache 域名及 OBS cache endpoint [V1] | 未把推理模型环境要求移植给 tile-rs | 外部贡献者能否直接使用 runner、网络/镜像/cache secrets 准入需维护者确认 |

尤其不能把 PyPTO 文档中的 driver 版本、Ascend-CI 的 CANN 镜像、FlagGems 的挂载范围拼成一个“推荐标准环境”：它们是不同项目、不同 job 的条件。

## 公开运行抽查

- [R1：PyPTO PR run 35722991008](https://github.com/hw-native-sys/pypto/actions/runs/35722991008)，2026-09-22，head `a35665bfe070669896610df0b95c98cf43e49518`。job API 显示 codegen-tests、system-tests、dist-system-tests、A5 onboard、A5 sim、CPU UT 等 success；serving-guard skipped。`system-tests` 的 [job 106730995142](https://github.com/hw-native-sys/pypto/actions/runs/35722991008/job/106730995142) 日志显示设备分派并结束于 **154 passed, 7 skipped, 279 deselected**。这是该次测试集合的事实，不是全部测试覆盖证明；当前 main 快照与 run head 不同。
- [R2：pypto-lib push run 35728234153](https://github.com/hw-native-sys/pypto-lib/actions/runs/35728234153)，2026-09-22，head 与本报告快照相同。CPU UT、a2a3sim、a5sim、a2a3 成功；serving 和 A5 专项跳过。只取得 job 级证据，没有逐用例日志审计。
- [R3：Ascend-CI Liger schedule run 33696502262](https://github.com/Ascend/Ascend-CI/actions/runs/33696502262)，2026-09-02 起跑，head `a6d1b216fee7e0bc9d1fa5d881d59d878e778ff5`。已另外读取该 commit 的 workflow；两个 NPU job 成功。[transformers job](https://github.com/Ascend/Ascend-CI/actions/runs/33696502262/job/100466452778) 日志中 `npu-smi` 显示两条 910B3，pytest 为 **1527 passed, 682 skipped, 67 deselected, 12 xfailed, 2 xpassed**。这是专用 Ascend 运行证据，而非从 Liger 支持宣传推断。
- [R4：vllm-ascend PR run 35726101891](https://github.com/vllm-project/vllm-ascend/actions/runs/35726101891)，head `47ea8174a98ea15f6e6f7ba5640452e5503b0797`。抽样成功 run 的 NPU 选择测试全为 skipped，仅 CPU 分支执行，不能报为 NPU 成功。
- AscendNPU-IR Actions API 返回 total_count=4，成功过滤无条目。未继续诊断失败原因，不声称它从不运行、也不把配置当成成功发布证据。
- FlagGems 抽样成功 run [34125321039](https://github.com/flagos-ai/FlagGems/actions/runs/34125321039) 的 job 查询没有找到 Ascend/cann 匹配；本轮不把它计入真实 Ascend 运行证据。没有无限追查历史或内部调度系统。

## 没有 NPU 能做什么，不能验证什么

可参照已有项目完成：格式/静态检查、IR 和变换单测、codegen 文本断言、host wheel 构建、明确实现的 stub API 契约检查，以及工具链支持的 simulator 数值测试。这里的“无 NPU”不等于“无 SDK”：AscendNPU-IR host build 下载 BiSheng；PyPTO codegen-only 与 simulator 的依赖也不同。

没有真机不能用这些结果替代：实际驱动/runtime 加载、kernel launch、设备内存和同步行为、真实数值差异、性能回退、HCCL 集体通信与多设备调度。仿真设备编号和真实设备编号不能混为一谈；多设备单机测试也不能证明跨节点网络。

## 证据缺口与后续维护者问题

1. PyPTO：各 runner 的真实芯片子型号、CANN/driver/firmware 精确版本如何记录？`task-submit` 的租约、隔离和 untrusted PR 准入是什么？A5 与 a2a3 simulator 各有哪些已知不等价行为？CPU 架构限制是临时 workaround 还是支持边界？
2. CATLASS / AscendNPU-IR / AKG / TorchAir：canonical GitCode/Gitee 是否有可公开的 PR/nightly job、日志和设备矩阵？host UT、编译回归、设备回归分别是不是 required checks？镜像 Actions 覆盖到哪一层？
3. FlagGems：910B weekly 的八设备是并行分片还是集体通信？可取得哪一次 Ascend job 日志及环境 inventory？`cann9` 与 weekly image 的版本关系是什么？
4. Ascend-CI：Liger PR ref 的固定 head SHA 和上游合入状态如何记录？哪些失败被容忍？runner 标签后缀与实际设备可见范围的契约是什么？容器 driver 挂载/注入由谁维护？
5. 所有项目：是否允许外部仓库复用服务？runner 准入、密钥、容器拉取、下载镜像及数据、网络 ACL 由谁审批？公共 YAML 不等于外部项目有访问权。

## 对 tile-rs PTO 路径的条件性参照

若后续确定中间路径确实产出 PTO，PyPTO 的 **IR/codegen → simulator → 单设备 → 多设备特例** 分层是直接可比较的证据；这不证明 PyPTO 本身适合作为 tile-rs 中间层。若只关心编译器产物，AscendNPU-IR 的 host 构建与 CATLASS 的 stub UT 能帮助定义“无卡门禁到底证明了什么”。若需要运营已有 NPU 池，Ascend-CI、FlagGems、vllm-ascend 可提供容器、队列/标签准入和测试选择的实例。

应先明确要验证的 kernel、精度、设备代际、是否需要分布式，再向对应维护者核实资源。不能从任何一项多卡配置推导采购数量，也不能因已有 simulator 就断言 tile-rs 无需 NPU。

## 固定源码证据索引

[P1]: https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml
[P2]: https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/daily_ci.yml
[P3]: https://github.com/hw-native-sys/pypto-lib/blob/133fd2273e96f9da62ec7b87adfd87d728e813cb/.github/workflows/ci.yml
[P4]: https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/README.md
[C1]: https://gitcode.com/cann/catlass/blob/7a9c18a2c38b8cf1905aacbbf3006db1a9468206/tests/unittest/catlass/gemm/tile/README.md
[I1]: https://github.com/Ascend/AscendNPU-IR/blob/8d49415f4d75d53dc983353e49b6c99004b0f17c/.github/workflows/wheel-x86_64.yml
[I2]: https://github.com/Ascend/AscendNPU-IR/blob/8d49415f4d75d53dc983353e49b6c99004b0f17c/bishengir/test/Integration/HIVM/VecAdd/README.md
[F1]: https://github.com/flagos-ai/FlagGems/blob/1f85ac4ee27a1d2236099600dcbaa90e9068d836/.github/backends.json
[F2]: https://github.com/flagos-ai/FlagGems/blob/1f85ac4ee27a1d2236099600dcbaa90e9068d836/.github/configs/weekly/910B.yml
[F3]: https://github.com/flagos-ai/FlagGems/blob/1f85ac4ee27a1d2236099600dcbaa90e9068d836/tools/gpu_check_ascend.sh
[F4]: https://github.com/flagos-ai/FlagGems/blob/1f85ac4ee27a1d2236099600dcbaa90e9068d836/.github/workflows/backend-test.yaml
[A1]: https://github.com/Ascend/Ascend-CI/blob/17946c98dac6db21b4f722ee9b8dd94df33442ca/.github/workflows/liger_kernel.yml
[A2]: https://github.com/Ascend/Ascend-CI/blob/17946c98dac6db21b4f722ee9b8dd94df33442ca/.github/workflows/llamacpp_test.yml
[V1]: https://github.com/vllm-project/vllm-ascend/blob/1ebf3a6c3a887b00bf680a2184b78af7aacae2ac/.github/workflows/pr_test.yaml
[T1]: https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/build.sh
[K1]: https://github.com/mindspore-ai/akg/blob/5aa15f3fe4d856eb60cd3b27447e04f5f1fb4dff/README.md
[O1]: https://github.com/Ascend/op-plugin/blob/7c5bd436230b03318b690b8fb769151be7966c61/README.zh.md
[S1]: https://github.com/Ascend/samples/blob/fc2aea93e48be8f8cee51d0c6c1b13a08155f83a/README.md
[R1]: https://github.com/hw-native-sys/pypto/actions/runs/35722991008
[R2]: https://github.com/hw-native-sys/pypto-lib/actions/runs/35728234153
[R3]: https://github.com/Ascend/Ascend-CI/actions/runs/33696502262
[R4]: https://github.com/vllm-project/vllm-ascend/actions/runs/35726101891

补充固定入口：[PyPTO setup-ci-job](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/actions/setup-ci-job/action.yml)、[FlagGems PR workflow](https://github.com/flagos-ai/FlagGems/blob/1f85ac4ee27a1d2236099600dcbaa90e9068d836/.github/workflows/unittest.yaml)、[FlagGems weekly workflow](https://github.com/flagos-ai/FlagGems/blob/1f85ac4ee27a1d2236099600dcbaa90e9068d836/.github/workflows/weekly.yaml)、[Liger 采样运行对应配置](https://github.com/Ascend/Ascend-CI/blob/a6d1b216fee7e0bc9d1fa5d881d59d878e778ff5/.github/workflows/liger_kernel.yml)。

## Resolution comment 草稿（待批准，不发布）

已完成其他 Ascend/PTO 项目的广度筛选。优先候选是 PyPTO/pypto-lib；CATLASS、AscendNPU-IR、FlagGems 和 Ascend-CI 提供不同层次的补充证据，vllm-ascend 仅作 CI 运营参照。报告区分了 host 构建、stub、仿真、真机及多设备测试，也标明哪些只有脚本或配置、哪些查到真实运行。没有从支持列表或 runner 标签推算设备池及采购数量。研究报告位于本票资产指针所列的本地分支，尚未推送。
