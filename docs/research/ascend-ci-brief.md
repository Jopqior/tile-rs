# PTO / Ascend CI 调研与无卡模拟验证简报

证据采集于 2026-09-22。范围：同类项目公开 CI 与两个独立最小原型，不涉及采购规模或正式 CI 部署。

## 结论

**无物理 NPU 可以做算子结果比较。** 本次在 GitHub 托管 Ubuntu runner 上，分别用 PTO CPU-SIM 和 CANN camodel 跑通最小 f32 add；两者都覆盖手写 PTO C++、独立 `.pto → PTOAS → C++` 两种入口。正常输出与独立 CPU 参考一致，故意破坏输出后 CI 失败，恢复后再次通过。

**这还不是 tile-rs 的 Rust→PTO 全链路验证，也不能替代真机。** 同类项目通常分开 host 检查、目标编译、模拟和 NPU 执行。实际驱动/runtime 集成、芯片上的数值与同步行为、性能仍需兼容真机；编译通过、模拟通过和真机通过分别提供不同证据。

## 同类项目怎么测，真机从哪里来

下表是已查快照与运行样本，不代表全部测试覆盖或持续可用性。真机来自项目接入的专用 runner 或设备池，不是所查 GitHub 标准托管 Ubuntu runner；设备产权均未核实，不能把 self-hosted 写成“项目自有”。

| 项目 | CI 与正确性测试 | 实际执行证据及设备提供方式 |
|---|---|---|
| [triton-ascend](triton-ascend.md) | CPU 静态/构建、NPU pytest，另有外部 Blue/Yellow 流水线 | 已取得 NPU 设备日志和用例通过记录；专用 Ascend runner。外部流水线仅取得成功回调，远端设备与覆盖未知。 |
| [TileLang 及 Ascend 实现](tilelang-ascend.md) | 主项目与 Ascend CI 分开。AscendC/PTO 有 PR/定时算子测试；MLIR 实现分 host 构建/IR 检查与设备测试 | 两个 Ascend 实现均有 NPU 执行记录，包含参考结果比较；使用 self-hosted 容器或专用 NPU runner，设备明细未完整公开。 |
| [torch_npu](torch-npu.md) | GitCode 配置有 CPU 构建、普通 UT、分布式 UT；GitHub 镜像另承载上游 PyTorch 外部 CI | GitHub 日志证实 NPU 可用及具体用例执行，设备由专用 runner 提供；GitCode 本轮只取得配置，不能以此声称其某次 UT 已运行。 |
| [PTO ISA / PTOAS](pto-toolchain.md) | host/IR、CPU-SIM、CANN simulator、真机 smoke/nightly 分层 | 已取得真机记录，使用 self-hosted 与 TaskQueue 设备调度。PTOAS 部分 PASS 仅是重复输出一致，须与独立参考比较分开。 |
| [PyPTO / pypto-lib](ascend-ci-landscape.md) | CPU 单测/codegen、CPU-SIM、真机系统测试 | 有模拟与真机 job 记录，PyPTO 另取得真实 pytest 日志；真机走专用 runner/任务队列。所查 `a2a3sim/a5sim` 是 CPU-SIM，不是 camodel。[更正依据][camodel] |
| [其他参照](ascend-ci-landscape.md) | CATLASS 有 host stub UT；AscendNPU-IR 有 host 构建及 NPU 示例；FlagGems 有 Ascend 算子测试配置；vllm-ascend 分 CPU/NPU 测试 | 本轮未取得这些候选的实际 NPU 测试日志；stub、示例或配置不当作真机成功，抽样中跳过的 NPU job 也不计入。 |

**Ascend-CI 是公共 CI 项目，不是已确认向任意仓库开放的 NPU 服务。** 已调查其 Liger Ascend 与 llama.cpp CANN 工作流：Liger 日志有两条 910B3 设备记录及真实 pytest；llama.cpp 算子循环会汇总失败后继续，workflow 绿色不等于所有算子通过。它使用接入的 NPU runner 与设备容器。可借鉴公开配置，但 runner 准入、设备使用权、网络和凭据需另行确认，设备产权未知。[调查与运行证据](ascend-ci-landscape.md)

## 两个模拟器及本次原型结果

| | PTO CPU-SIM | CANN simulator / camodel |
|---|---|---|
| 来源与职责 | `pto-isa` 提供的 PTO C++ CPU 模型，用普通 CPU 编译器执行其支持的操作；不直接解释 `.pto` | CANN Toolkit 提供的目标模拟器/runtime，执行经 Bisheng 目标编译的 kernel |
| 本次环境 | GitHub 托管 Ubuntu 24.04 x86_64、GCC 13；不安装 CANN 或驱动 | GitHub 托管 Ubuntu 22.04 x86_64、无驱动 CANN 9.1 Toolkit、Bisheng；Ascend910B1 / dav_2201 模型 |
| 覆盖与边界 | 可比较模型支持范围内的功能结果；不验证设备二进制、硬件时序或性能 | 可比较指定目标模型的执行结果，覆盖此次目标编译和模拟运行；不证明真实驱动、硬件行为或性能 |
| 两入口运行结果 | [正常通过][cpu-ok] / [负例失败][cpu-bad] / [恢复通过][cpu-restored] | [正常通过][cam-ok] / [负例失败][cam-bad] / [恢复通过][cam-restored] |

两个原型均使用 PTOAS 0.65 和固定 PTO ISA 版本。生成 C++ 原样使用，只补 host 输入输出与调用。输入为固定 `[1,256]` f32 add，值及其和均可精确表示；独立 Python 标量参考逐元素比较，并检查长度与有限性。负例在计算完成后将首个输出加 1，两入口均因比较不匹配退出 1，验证错误能传递为 CI 失败，不是故意制造下载或编译故障。

该结果只证明此最小用例可行，不覆盖其他 shape、dtype、一般浮点舍入、NaN/Inf 运算或多卡通信。CPU-SIM 无需 SDK；camodel 需要匹配的 Toolkit、下载网络、安装空间及许可条件，不能把“无卡”写成“无软件依赖”。本次未再分发 SDK。版本、命令和长期证据保留在 [CPU-SIM 实验记录][cpu] 与 [camodel 实验记录][camodel]，避免只依赖有保留期的 Actions 日志。

## 接入 tile-rs 还缺什么

仓库已有 PTO 产物和运行记录，但核心 Rust→MLIR/PTO 后端不公开；所查发布标签未提供 Linux 后端包。两个原型的 IR 均为独立手写样例，尚未测试 tile-rs 实际输出。要建立可追溯的端到端结果，仍需确认：

- 可获取、匹配 host 平台和 rustc ABI 的后端，以及当前 Rust kernel 生成实际 PTO 产物的可复现入口。
- 实际产物与 PTOAS、PTO headers、CANN 的版本兼容，host 调用/launch ABI，以及对应算子的独立参考和误差规则。
- 若验证真机，目标设备、驱动/固件、运行库与访问条件；同行 runner 标签或设备数量不能直接变成 tile-rs 的环境要求。

上述缺口不影响“两个独立最小无卡原型可行”的结论，但限制了向完整链路的推广。[tile-rs 路径核查](tile-rs-pto-path.md)。原型保留在独立实验分支，未接入 main 正式门禁；本简报不提出硬件选型、采购数量或环境协调请求。

[cpu]: https://github.com/Jopqior/tile-rs/blob/50eaa4be2fe0e59e464dc7d9bba31bcd054452c6/docs/research/pto-cpu-sim-prototype.md
[camodel]: https://github.com/Jopqior/tile-rs/blob/4d875e05c82afe0e6f1036cc5caeb386cb40b9e6/docs/research/camodel-prototype.md
[cpu-ok]: https://github.com/Jopqior/tile-rs/actions/runs/35753705495
[cpu-bad]: https://github.com/Jopqior/tile-rs/actions/runs/35753801302
[cpu-restored]: https://github.com/Jopqior/tile-rs/actions/runs/35753920477
[cam-ok]: https://github.com/Jopqior/tile-rs/actions/runs/35761288167
[cam-bad]: https://github.com/Jopqior/tile-rs/actions/runs/35761853455
[cam-restored]: https://github.com/Jopqior/tile-rs/actions/runs/35762428063
