# 申请 Ascend NPU 访问时可引用的两份 CI 实例

核对日期：2026-09-27；以下源码均固定到提交，运行日志另按 run/job 固定。用途是为 yijunyu/tile-rs 的 Rust→PTO 全链路正确性测试说明所需设备访问，不是多卡采购依据。

后续已把真实 Rust add 接到 PTOAS/CANN/camodel，并通过独立 CPU 比较：[完整运行](https://github.com/Jopqior/tile-rs/actions/runs/36337430892)、[原型与实际接入变量](https://github.com/Jopqior/tile-rs/blob/14910d6/docs/research/rust-pto-ascend-prototype.md)。Rust 前端放在 macOS，使用配套 release DSL；Linux NPU host 不需要 Rust 后端。以下同行配置是参照，不替代该原型的实际环境合同。

## 1. PyPTO：PTO add 真机正确性实例

[PyPTO CI 固定配置](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml#L1228-L1323) 的 `system-tests-direct` 使用 `[self-hosted, linux, npu]`，`needs-device: true`，先运行 [setup-ci-job](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/actions/setup-ci-job/action.yml#L133-L265) 完成检出、工具链安装、PyPTO 扩展构建/安装，再运行 pytest。runner 必须提供 `CI_CACHE_ROOT`、`CANN_ROOT`、`DEVICE_ID` 以及 bundle 工具链；action 在 device job 的 `activate.sh` 中 source `$CANN_ROOT/set_env.sh`，安装按版本及 SHA256 校验的 PTOAS wheel、固定 PTO ISA commit，并[构建/安装 PyPTO 与 simpler](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/actions/setup-ci-job/action.yml#L516-L704)。这些是该项目自己的 provision 契约，不可仅复制一条 pytest 命令到空白 runner。CI 当前是**无 job 容器、host-native**；注释里“将来改 CPU runner/auto 借卡”不是现状。`CANN_ROOT=/path/to/Ascend/cann-9.0.0` 是 action 报错中的示例路径，不是所有 runner 的已核实安装版本。

该 job 实际执行的命令（位于已安装且已 `source activate.sh` 的 checkout，`DEVICE_ID` 非空）是：

```sh
python -m pytest tests/st/runtime/ops tests/st/runtime/cross_core \
  -v -m device_batch \
  --precompile-workers=32 --strict-case-discovery \
  --execute-via-task-submit --execute-batch-size=64 \
  --task-queue-timeout=1800 --task-max-time=600 --task-submit-device="$DEVICE_ID"
```

[PyPTO 该提交的 add 用例](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/tests/st/runtime/ops/test_elementwise.py#L26-L52) 包含 64×64/128×128 f32 `tile_add`，使用 `golden=lambda _: a + b` 和 `case_run.assert_passed()`；[PR run 35722991008 的 system-tests-direct job 日志](https://github.com/hw-native-sys/pypto/actions/runs/35722991008/job/106730996219) 明列两个 `test_tile_add` **PASSED**，设备号 10。该运行的 head 为 `a35665bfe070669896610df0b95c98cf43e49518`，与上面 CI 快照不同；配置在本报告固定快照所述，日志是一次独立样本。只调试 add 时，可在**同一已 provision 的 PyPTO runner** 把上述 pytest 两个目录换成 `tests/st/runtime/ops/test_elementwise.py::test_tile_add`，其余参数不变；这是从已存在的测试和 CLI 参数缩小选择范围，**不是原 workflow 原样运行的命令，更不测试 tile-rs**。[pytest 参数定义](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/tests/st/conftest.py#L123-L247) 中 `--codegen-only` 会跳过 runtime，绝不能用于声称真机正确性。

## 2. Ascend-CI：可见的容器/设备接入实例（非 PTO）

[Ascend-CI 的 Liger workflow 固定配置](https://github.com/Ascend/Ascend-CI/blob/17946c98dac6db21b4f722ee9b8dd94df33442ca/.github/workflows/liger_kernel.yml#L1-L126) 使用默认 `linux-aarch64-a2-2` runner、`swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.1.0-910b-ubuntu22.04-py3.12`，容器选项 `--privileged`、`--ipc=host`，挂载 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`；先 `npu-smi info`、读取 toolkit 目录/版本，再装依赖并执行 `python -m pytest test/transformers ...`。它测试 Liger/Triton-Ascend，**不编译 Rust→PTO**，因此镜像及 PyTorch/triton 依赖不是 tile-rs 的既定要求。[schedule run 33696502262 的 transformers job](https://github.com/Ascend/Ascend-CI/actions/runs/33696502262/job/100466452778) 在两条设备记录中看到 `910B3`，结束于 1527 passed、682 skipped、67 deselected；run head `a6d1b216fee7e0bc9d1fa5d881d59d878e778ff5` 的 workflow 与所引快照一致。日志所见两设备不代表最小 add 要两卡，`npu-smi 25.5.2` 亦不等于已确认驱动/固件完整版本。

## tile-rs 申请范围与前置缺口（提案，不是已存在的脚本）

建议申请**一种明确 SoC 的单设备真机执行机会**及配套同机/共享路径的 host 编译环境；先要求维护方确认实际 SKU、可用卡号/时段、host 架构、CANN Toolkit/Bisheng 与 PTOAS/PTO ISA pin、驱动/固件兼容性、设备节点/权限、是否提供 `task-submit` 或容器运行时、镜像/SDK 网络与许可证、超时/隔离/清理方式。PyPTO host-native 的 `CANN_ROOT` 和 Ascend-CI 910B 容器是两种**不同**实现，不能拼成一套已经验证的 tile-rs 环境。

目标验收合同是：从固定 Rust add kernel 与工具链/后端提交构建，留下 Rust→MLIR→PTO、PTOAS 后处理、PTO→AscendC/CANN 编译的产物及版本；在获授权 NPU 上初始化 ACL、launch/sync、D2H；对长度/shape 与每个 f32 输出用**独立 CPU 加法参考**逐元素比较，检查有限值与误差阈值，故意破坏一个输出应以非零退出。单设备功能正确性足够作为首个请求，不把分布式、性能或 PyPTO 的其他算子纳入最小额度；编译/仿真通过不能替代真机数值结果。

务必先打通当前上游：旧报告 `tile-rs-pto-path.md` 所引旧 softmax、`tools/ascend/pto_run.cpp` 和旧 `KernelBuilder` 命令**不是** [yijunyu/tile-rs 最新已核查提交 `6fbebdb`](https://github.com/yijunyu/tile-rs/tree/6fbebdb0e7c1f6d453c32fe34f146fd6490decf1) 的可执行入口。新提交公开了 [PTO emitter](https://github.com/yijunyu/tile-rs/blob/6fbebdb0e7c1f6d453c32fe34f146fd6490decf1/crates/rustc_codegen_tile/src/mlir_to_pto.rs)、[`tile` CLI 帮助/`-r` 接口](https://github.com/yijunyu/tile-rs/blob/6fbebdb0e7c1f6d453c32fe34f146fd6490decf1/crates/tile_cli/src/cli.rs#L355-L405) 和 [预构建 `pto_to_ascendc` 的输入合同](https://github.com/yijunyu/tile-rs/blob/6fbebdb0e7c1f6d453c32fe34f146fd6490decf1/vendor/pto_to_ascendc/README.md#L1-L25)：它只接受经 `ptoas --enable-insert-sync --emit-pto-ir` 后具有 `addr =` 的 PTO IR，二进制源码不在公开树。实施前没有针对当前提交验证整条链。后续原型已验证实际 Rust→MLIR→当前 PTO emitter→PTOAS→CANN/camodel→独立比较，固定配套 release DSL，复用原 camodel ACL host；不需要 Linux Rust 后端。真机 job 已有实际脚本，但还没有物理 NPU 运行结果。上面的 PyPTO 命令是已证实的**硬件验收参照**，不是全链路替身。

安全/触发：PyPTO 的 [`push main`/`pull_request main`/手动](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml#L1-L10) 会调度自托管 NPU job（除 docs-only 等条件）；device job 显式 `permissions: contents: read`、checkout `persist-credentials: false`，且通过 `task-submit` 排队并清理孤儿任务，但**运行不可信 PR 代码仍可触及 host/设备**。Ascend-CI 仅在其自身 workflow 路径变化的 push/PR、定时或手动时触发，checkout 固定的 `linkedin/Liger-Kernel` PR ref；容器 `--privileged` 更须限定准入。申请时要求设备管理员说明 fork PR 审批、允许的触发/作者、token/secrets 最小权限、runner 与宿主隔离、并发配额及日志保留。公开 workflow/runner 名**不授予 yijunyu/tile-rs 使用这些他人设备的权限**；不建议直接把未审查的外部 PR 接上特权设备。
