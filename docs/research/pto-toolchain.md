# PTO ISA / PTO AS 验证环节的资源边界

调查日期：2026-09-22（UTC）。对应票：[核实 PTO ISA 与 PTO AS 验证环节的资源边界](https://github.com/Jopqior/tile-rs/issues/53)。本报告是资源与证据调查，不确定 tile-rs 的后端架构，不估算采购数量。

## 摘要

- **PTO ISA 是 Ascend CANN 定义的 Parallel Tile Operation 虚拟指令集**；`pto-isa` 仓库提供 tile 指令的 C++ 实现、CPU 模拟器、文档及测试。它不是一份可以直接装载到 NPU 的最终机器码。[I1]
- **PTOAS 是 PTO Assembler & Optimizer**，基于 LLVM/MLIR，解析和验证 PTO IR、做优化与同步插入、lower 到 EmitC 等并生成调用 `pto-isa` C++ 库的代码；另有 `ptobc` 编解码 PTO-BC。这里的“汇编”不等于完成设备二进制编译或真机验证。[A1][A2]
- 不用 NPU 可以做 host 工具构建、IR/bytecode 检查、代码生成、PTO CPU-SIM 功能测试；安装匹配的 CANN/Bisheng 后，还可以在无卡机器上做设备目标的 compile-only，以及指定的 CANN simulator 测试。[I2][A3]
- **存在公开真机 CI，不只是测试脚本**：PTO ISA 的 PR A5 Smoke 有成功执行记录；PTOAS 有 A3/A5 nightly board workflows 和实际执行日志。相反，最新 A3 nightly 在 host configure 就失败，board 被跳过，不能把 workflow 存在当成每次都上板。[R2][R3][R4][R5]
- 真机 PASS 的含义也要细分：PTOAS 无独立 golden 时重复执行同一 NPU kernel，只验证重复输出一致。采样 A5 run 的 654 个 PASS 中 550 个标为 `DETERMINISM_ONLY`，不能称为全部数值正确。[A7][R4]
- CPU-SIM、CANN simulator、真机执行是三类资源合同。支持 A2/A3/A5 的列表不是设备池；单设备 smoke 也不覆盖多卡通信、性能或完整分布式路径。

## 1. 项目身份、来源与版本

### 官方入口和镜像边界

1. `pto-isa` 的官方入门文档给出的 clone 地址是 <https://gitcode.com/cann/pto-isa.git>。公开 GitHub 对应项目现在是 <https://github.com/hw-native-sys/pto-isa>；旧地址 <https://github.com/PTO-ISA/pto-isa> 实测 HTTP 301 到该地址。身份同时有 CANN 定义、版权/许可证、GitCode clone 地址和同生态项目互链支撑，不依据名称相似判断。[I1][I2]
2. PTOAS 的 GitHub 仓库描述明确写着 **read-only mirror of GitCode，PR 提交到 <https://gitcode.com/cann/pto-as>**。当前可访问的源码和 GitHub CI 在 <https://github.com/hw-native-sys/PTOAS>；PTO ISA README 也将 PTOAS 列为关联 assembler/compiler backend。[A1][M1]
3. 本次 GitCode PTOAS 页面请求返回 403，未访问其内部 CI、动态生成脚本及运行日志。因此下文 GitCode CI 部分是 **GitHub 快照中 `.gitcode/` 配置的观察**，不是 GitCode 当前分支和实际执行状态的认证。GitHub/GitCode 不能混用 SHA：官方 compile-only 文档特意区分 `gitcode-default`、`github-ci-sim`、`cann90-dev` 三套 pin。[A3]
4. 名称存在历史表述：PTO ISA 文档将 PTO 展开为 Parallel Tile Operation；PTOAS README 对 PTO Bytecode 另写 Programming Tiling Operator Bytecode。本文按具体工具、IR、库职责区分，不强行把所有历史缩写统一，也不把 README roadmap 当已完成能力。[I1][A1]

### 固定快照

| 对象 | 分支 / 固定 commit | 用途 |
|---|---|---|
| GitHub PTO ISA | `main` / `82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19` | 库、文档、GitHub/GitCode CI 配置 |
| GitHub PTOAS | `master` / `97e9ecb247744c587171c73ab58ac2423bba3197` | 工具、compile-only、CI 和验证脚本 |
| 实际 CI runs | 下节逐项记录 head SHA / URL | 运行证据独立于上述源码快照 |

使用 `gh search repos`、GitHub REST、公开 git clone 和官方仓库文档交叉核对。未执行本地 LLVM/CANN 构建，也未占用 NPU；结果来自源码审阅和公开运行记录。本会话无 web_search 工具，检索使用 GitHub 第一方接口，未引用搜索摘要作为事实依据。

## 2. 实际工具关系与分阶段证据矩阵

已证实的一个工具流是 `PTO IR (.pto) → ptoas → C++（调用 PTO Tile Library）→ CANN/Bisheng → 可执行验证工程 → simulator 或 NPU`；`ptobc encode/decode` 是 IR bytecode 的另一条序列化支路。该工具流不证明 tile-rs 已产生 PTO IR，也不证明必须采用此流。[A1][A2][A3]

| 阶段 | 物理 NPU 必要性 | 可证实依赖 / 入口 | 通过能说明什么；不能说明什么 |
|---|---|---|---|
| PTOAS / LLVM host 构建及 Python 包 | 不需要 | LLVM19 `vpto-dev/llvm-project:feature-vpto`；CMake/Ninja、C++、Python；host CI 构建无 CANN 步骤 [A1][A4] | 工具可构建、导入、安装；不是设备代码有效性 |
| PTO IR 解析、op verifier、pass/lowering | 不需要 | `ptoas`、`ninja ... check-pto`、lit/CTest；samples 的 py→pto→cpp [A1][A4] | 已实现的静态约束、转换和回归；不是所有动态语义的证明 |
| PTO-BC 编解码/往返兼容 | 不需要 | `ptobc encode/decode`；CI 的 `ptobc_` CTest、跨版本 reader 检查 [A2][A4] | IR 序列化/反序列化兼容；不是 NPU 指令直接执行 |
| 生成 C++ 的目标编译及 template 静态检查 | 不需要，但需要设备工具链 | CANN Toolkit/Bisheng + 匹配 `pto-isa/include`、`tests/common`；`STAGE=build` [A3][A7] | 当前目标工具链接受代码和静态布局；不验证 ACL/驱动、权限、数值、真机死锁 |
| PTO CPU-SIM | 不需要；不要求 CANN | `tests/run_cpu.py` / `tests/run_cpu_tests.sh`，C++20、Python/CMake [I2][I3] | CPU 模型覆盖范围内的功能/trace；不是芯片性能或硬件时序 |
| CANN simulator / camodel | 指定 simulator 路径不需要物理 NPU；需要匹配软件组件 | `run_st.py -r sim`；PTOAS `ci_sim.yml` 的 CANN、dav_3510 camodel、PyPTO/PTODSL ST [I2][A6] | 模拟目标的执行与比较；不是实际硬件/驱动配置证明。不能只凭 self-hosted 标签判断有卡 |
| 真机 kernel 运行与数值验证 | 需要 | `STAGE=run RUN_MODE=npu`，ACL runtime、可访问设备、驱动/固件，独立 golden [A7][I2] | 被执行用例的设备运行/比较；无 golden 时可能只有确定性验证 |
| 跨 NPU 通信及性能 | 需要与该用例匹配的多设备/拓扑 | TGET 示例要求至少 2 NPU、MPICH；异步 TGET 要 CANN ≥9.0.0 [I5] | 该通信场景与测量；不能从单卡 smoke 或 CPU 模型外推 |

“静态验证”在这里指 MLIR verifier、lit/FileCheck、C++ template 约束等可观察机制，不声称存在完备形式验证或对任意 kernel 的安全证明。

## 3. CI 入口、触发和门禁

### GitHub 配置

| 入口 | 触发 | runner / 实际工作 | 门禁边界 |
|---|---|---|---|
| PTO ISA `ci.yml` [I3] | `main` push、面向 `main` 的 PR、手动 | `ubuntu-latest`；pre-commit、docs、CPU SIM smoke/trace/full ST | full ST 依赖前面三个 job；不执行真机 |
| PTO ISA `a5-smoke.yml` [I4] | PR opened/synchronize/reopened/ready_for_review、手动 | `[self-hosted,a5]`，TaskQueue 调 `build.sh --run_simple --a5 --npu` | 真机 PR smoke；不是完整 ST 或分布式测试 |
| PTOAS `ci.yml` [A4] | PR、`main` push、每日 UTC 19:00、手动 | host：`ubuntu-22.04`；A3：`[self-hosted,Linux,ARM64,ptoas-a3]` | A3 只对本仓库 schedule/dispatch，且 `needs: build-and-test`；默认 `npu`，手动可选 build/run、npu/sim；不是每个 PR 板测 |
| PTOAS `ci_a5.yml` [A5] | 每日 UTC 19:00、手动 | `[self-hosted,Linux,ARM64,ptoas-a5]`；本机生成 A5 payload，TaskQueue 上板，上传结果 | `Ascend950` / `RUN_MODE=npu`；维护跳过名单；不是 PR 通用 gate |
| PTOAS `ci_sim.yml` [A6] | PR、每日 UTC 14:00、手动 | `[self-hosted,Linux,X64,label-1]`；PyPTO、VPTO、TileLib、PTODSL simulator 测试 | PR 按路径过滤；不是所有 PR 都运行完整 simulator，部分 harness 是 smoke |
| PTOAS `build_wheel.yml` [A8] | PR、`main` push、UTC 17:10 nightly、手动、release | x86_64 `ubuntu-latest` / aarch64 `ubuntu-24.04-arm`；manylinux 容器 | PR Python 3.11；nightly/release/手动为 3.10–3.12。构建发布包不是板测通过证明 |

重要配置不一致：采样时 PTOAS default branch 是 `master`，但 host CI 和 wheel 的 push filter 仍为 `main`。不能概括成“每次默认分支 push 都跑这些任务”。cron 的配置时间也不等于保证准时运行；下表观察到延后启动。[M1][A4][A8]

工作流内的 `needs`、失败退出和跳过条件可确认；GitHub branch-protection API 对两仓库返回 404，不能据此判定 required status checks 是否开启，也不能断言某个失败一定阻止合并。GitHub Actions API 还列出历史 `Remote NPU Validation`，但当前快照已无该文件，唯一查询到的历史 run 是 2026-02-07 failure；不能拿 API 中的 active 标签当当前 CI 入口。[M2]

### GitCode 配置可见，但运行证据未取得

- 两项目 `.gitcode/workflows/` 都有以 compile 评论正则触发的 PR pipeline；PTOAS 另有手动入口。compile 配置区分 `arch=x64`、`arch=arm64`，通过 SWR 容器、BuildAccelerate、OBS 分发构建产物。[G1][G2]
- PTO ISA 有 UT、automode A3/A5、KIRIN build 等 job，但具体 `ut.sh`、`compile.sh` 从 OBS 下载，**job 名不证明实际使用哪类硬件或执行了哪些测试**。[G1][G3]
- 两者有面向 `master` 的 PreSmoke，runner 标签 `self-hosted,npu=1,smoke=204`，执行 OBS 下发的 `pre_smoke.sh`。这里 `npu=1` 是标签文本，不能推导真实卡数；三个 part 也不是三台或三卡的证明。[G1][G2]
- PTOAS 的 `SIM_VALIDATION` 明写 observation phase：缺环境时警告并跳过而不是硬失败。不能把它当已强制执行的 simulator gate。[G2]
- 镜像地址、OBS AK/SK、GIT_TOKEN、共享 actions 可在配置中看到；账号准入、动态镜像 digest、实际设备池和真实运行结果未核实。未追逐不可访问的内部系统。

## 4. 公开实际运行记录

以下是采样，不是可用率统计；run 对应 SHA 与当前审阅快照不同。

| 项目 / 日期 | run / head SHA | 已观察事实 |
|---|---|---|
| ISA host，09-21 | [35567976227][R1]，`82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19` | push run success，与当前 ISA 快照相同 |
| ISA A5 PR，09-21 | [35565316201][R2]，`e1fde026389ed447731eb824ab9d9198ee99bcaf` | job runner `pto-isa-a5-39-upstream-root`；A5 smoke step success；已读完整下载日志，含 NPU 路径测试和 GTest PASS。09-22 新一轮 [35691189240](https://github.com/hw-native-sys/pto-isa/actions/runs/35691189240) 为 failure，不能只报成功样本 |
| PTOAS A3 nightly，09-02 | [33685098919][R3]，`bdcb319d6ad43fe4a562e8911e05aebca228b848` | runner `ptoas-a3-01`；TaskQueue board step success；日志 `MATCHED=363 PASS=323 CORRECTNESS_OK=90 DETERMINISM_ONLY=233 FAIL=0 SKIP=40` |
| PTOAS A5 nightly，09-21 至 09-22 | [35663047961][R4]，`6d744afb538af093c29f89e92a84863c93b4068a` | runner `ptoas-a5-01`；payload 生成成功、board step failure；日志 `MATCHED=708 PASS=654 CORRECTNESS_OK=104 DETERMINISM_ONLY=550 FAIL=1 SKIP=53`，失败 case 为 scatter。不是“根本没跑板测” |
| PTOAS 最新 A3 nightly，09-21 | [35662292225][R5]，同上 `6d744afb…` | host `Validate strict CMake 4 configure` failure；`remote-npu-validation` skipped，没有分配 A3 runner |
| PTOAS simulator nightly，09-21 | [35641893749][R6]，同上 `6d744afb…` | runner `b0`、X64/label-1；job 执行至 `Run PTODSL ST CI` 失败。只核对 job/step 状态，未据此宣称所有 simulator 用例正确 |
| PTOAS wheel PR，09-22 | [35707189673][R7]，`058422d894921a917be96193c87fe3ef1c9e4b9a` | x86_64 与 aarch64 两个 Python 3.11 wheel job success；发布 job skipped，非真机测试 |

### 为什么 PASS 不等于数值正确

`run_remote_npu_validation.sh` 的 `GOLDEN_MODE=npu` 分支先运行 kernel；若找不到独立 golden，则将首次输出复制为 golden，再次生成输入并运行，同步记录 `.determinism_only`，最终报告 `validation=determinism-only`。[A7] 上表 A3/A5 日志均实证该分支发生。`GOLDEN_MODE=skip` 更明确跳过比较。

所以必须分别读取 `STAGE`、`RUN_MODE`、golden 模式和结果分类；不能只看 workflow 绿色或汇总 `PASS`。无独立 oracle 时，资源上确实消耗了 NPU，但获得的结论不是算子数学正确性。

## 5. 公开可证实的资源条件

| 项目 | 官方要求或配置事实 | 不能据此推出 / 未知 |
|---|---|---|
| host CPU 架构 | ISA CPU 支持 x86_64/AArch64；CPU 入门覆盖 Linux/macOS/Windows。PTOAS wheel 配置且实际成功覆盖 Linux x86_64/aarch64；A3/A5 board job 显式检查 `uname -m == aarch64` [I1][I2][A5][R7] | 不是 tile-rs 必须使用 ARM host；不等于所有平台、所有测试均被 CI 覆盖 |
| 编译软件 | ISA CPU：Python ≥3.11、CMake ≥3.16、C++20（Linux 文档 GCC13+/Clang15+）。PTOAS：Linux/Ubuntu20.04+ 推荐、CMake≥3.20、Python≥3.10、LLVM19 VPTO、自身 README 写 C++17；host CI 使用 Clang15 [I2][A1][A4] | LLVM 资源消耗、最小 RAM/CPU 核/磁盘没有本次可复核下限；`-j`/jobs 值不是硬件需求 |
| CANN | ISA NPU/模拟器入门写 ≥8.5.0；A5 smoke 默认路径包含 `cann-9.0.0/aarch64-linux`。PTOAS compile-only 强调 A5 CANN 与 ISA pin 对齐 [I2][I4][A3] | 8.5.0 不应当作所有 A5 用例的充分版本；路径不是全部实际安装包版本清单；不同 target pin 不可混用 |
| 模拟器环境 | ISA 说明只构建或模拟时可跳过驱动/固件；PTOAS simulator job 要预装 CANN、torch/torch_npu、camodel，PyPTO smoke 另要求 GCC15 [I2][A6] | 安装 torch_npu 不意味着调用了 NPU；CANN 模拟器不是纯 C++ CPU-SIM；未核实所有 host 上可获得对应组件 |
| 设备型号 | ISA 文档支持 A2=910B、A3=910C、A5=950。PTOAS A5 workflow 目标 `Ascend950`；A3 workflow 默认 `Ascend910`，README 上板示例用 `Ascend910B1` [I1][A1][A4][A5] | runner 名、目标 SoC 编译字符串和实际物理 SKU 不是同一事实；无法由此锁定 A3 芯片具体变体或设备总量 |
| 单卡数量边界 | A3 TaskQueue 默认选择 `auto`；A5 默认 device id `1`，ISA smoke 默认 `0`，验证程序选择一个设备 [A4][A5][I4] | id 不是数量；默认 1 不代表需要两卡，不得推算池规模 |
| 多卡场景 | TGET 带宽示例明确两 NPU 点对点，要求 ≥2 Ascend NPU、MPICH；TGET ≥8.5.0 / TGET_ASYNC ≥9.0.0 [I5] | 这是该示例条件，不是所有 PTO 测试需求；未确认它被上述 CI 执行，也未证明跨主机拓扑 |
| 驱动/固件 | 真机官方要求驱动与固件；脚本核查 `/dev/davinciN` 读写权限并处理物理/逻辑设备映射 [I2][A7] | 精确驱动/固件版本和 CANN 兼容矩阵没有在本次采样中确认，不能给出采购或部署基线 |
| 容器 | wheel 使用 `quay.io/pypa/manylinux_2_34_{x86_64,aarch64}`；GitCode SWR image 由共享步骤解析；GitHub board jobs 无 `container:` 配置 [A8][G1][G2] | 无 job container 不证明底层宿主绝无容器；动态 tag 不等于可复现 digest |
| 权限/网络 | host CI 需要 apt/pip、GitHub LLVM 源码和 Actions artifacts；board 下载 payload、访问 GitCode pin，必须有 TaskQueue 和暂时设备权限；ISA smoke 注释说明 non-root runner 经 TaskQueue 提交给 root；GitCode 使用 SWR/OBS/密钥 [I4][A4][A5][G2] | 不是所有 PTO 用户都必须 root；task-submit 是该共享设备池的调度合同。公开 runner 名不提供外部访问或使用授权 |

## 6. 对无 NPU 环境与 tile-rs 的条件性参照

若将来选择产生 PTO IR，可以在无 NPU host 上检查工具构建、IR verifier、bytecode round-trip、lowering 输出，并在 **匹配工具链可获得** 的前提下追加 Bisheng compile-only。若选择直接产生 PTO Tile Library C++，则未必需要 PTOAS/bytecode 这一段。本报告不替这两种或其他路径作选择。

如果要复用模拟器结果，应明确实际调用的是 PTO CPU-SIM 还是 CANN camodel，以及它覆盖的指令、数据类型和同步模型。不能把 CPU-SIM 通过当成目标 C++ 必然可编译，或把目标 compile-only 当成真机无死锁和数值正确。

真机需求至少应绑定到一个明确产物和测试合同：目标 SoC、CANN/ISA/compiler pin、执行入口、设备访问方式、独立 golden/误差规范、是否要求多卡与性能。只有单设备功能用例时，不应从通信扩展的存在扩大到多卡需求；反之涉及通信时，单设备 smoke 不能代替。

## 7. 公开证据缺口与需维护者确认的问题

1. GitCode canonical 分支与 GitHub 快照的同步关系、CANN 版本及三个 pin 的兼容维护责任；默认 `master` 与部分 GitHub push filter `main` 的当前意图。
2. 真正的 required checks、PR fork 审批、自托管代码执行信任边界，以及 GitCode simulator observation phase 何时转强制 gate。
3. A3/A5 实物 SKU、可申请设备数、是否共享、排队/超时/清理合同、驱动/固件精确版本。公开配置和 runner 名不能替代设备清单。
4. CANN simulator 组件的可下载版本、许可/网络准入、host 架构适配；CPU-SIM 与 simulator 对未来 tile-rs 目标指令的覆盖差异。
5. 哪些 board cases 有独立 golden，哪些仅确定性；跳过名单、已知失败、模型精度和同步测试的验收边界。
6. GitCode OBS 动态 `compile.sh`/`ut.sh`/`pre_smoke.sh` 的固定版本与可公开运行记录；镜像 digest、秘密管理及设备透传合同。
7. 多卡通信用例是否进入 PR/nightly/release；所需拓扑、MPI/HCCL 版本、跨机网络及权限。本次不从支持列表或参考性能图推导配置。

## 来源索引

以下源码链接均固定到本次采集 commit；所在分支与日期见第 1 节。API 和 Actions run 是动态服务，run URL 固定标识一次执行，日志有保留期。

[I1]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/README.md
[I2]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/docs/getting-started.md
[I3]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/.github/workflows/ci.yml
[I4]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/.github/workflows/a5-smoke.yml
[I5]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/kernels/manual/a2a3/tget_bandwidth/README.md
[A1]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/README_en.md
[A2]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/tools/ptobc/README.md
[A3]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/docs/no_npu_compile_only_guide_zh.md
[A4]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.github/workflows/ci.yml
[A5]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.github/workflows/ci_a5.yml
[A6]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.github/workflows/ci_sim.yml
[A7]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/test/npu_validation/scripts/run_remote_npu_validation.sh#L838-L951
[A8]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.github/workflows/build_wheel.yml
[G1]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/.gitcode/workflows/PR-pipeline_pto-isa.yml
[G2]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.gitcode/workflows/pto-as_action.yml
[G3]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/.gitcode/workflows/llt_action.yml
[M1]: https://api.github.com/repos/hw-native-sys/PTOAS
[M2]: https://github.com/hw-native-sys/PTOAS/actions/runs/21777246614
[R1]: https://github.com/hw-native-sys/pto-isa/actions/runs/35567976227
[R2]: https://github.com/hw-native-sys/pto-isa/actions/runs/35565316201
[R3]: https://github.com/hw-native-sys/PTOAS/actions/runs/33685098919
[R4]: https://github.com/hw-native-sys/PTOAS/actions/runs/35663047961
[R5]: https://github.com/hw-native-sys/PTOAS/actions/runs/35662292225
[R6]: https://github.com/hw-native-sys/PTOAS/actions/runs/35641893749
[R7]: https://github.com/hw-native-sys/PTOAS/actions/runs/35707189673

## Resolution comment 草稿（待批准，未发布）

已核实 PTO ISA 与 PTOAS 的身份、工具职责及无卡/模拟/真机边界。无 NPU 可做 host 构建、IR/bytecode 检查、CPU-SIM；匹配 CANN/Bisheng 后还可做目标 compile-only 和对应模拟器验证。公开 CI 确有 A5 PR smoke、A3/A5 nightly 板测，但上板 PASS 可能仅表示重复输出一致，不能替代独立 golden 验证。

报告位于 `research/pto-toolchain` 的 `docs/research/pto-toolchain.md`，记录固定源码、实际 runs、GitCode 可见配置与资源缺口。报告仅条件性参照 tile-rs，不确定编译路径或采购规模。资产尚未推送，精确本地 commit/路径见本票资产指针。
