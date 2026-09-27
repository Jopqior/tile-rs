# Rust→PTO→Ascend CI 原型

归档：[补录新版 Rust→PTO→camodel 验证与 NPU 资源申请草稿](https://github.com/Jopqior/tile-rs/issues/61)。

独立分支：`prototype/61-rust-pto-ascend-ci`。不接入 main，不注册或调用未获授权的真机。

## 验证范围

前端已在 [macOS ARM Actions](https://github.com/Jopqior/tile-rs/actions/runs/36337202111) 编译实际 Rust add，产出 MLIR，再用当前公开 PTO emitter 生成 PTO。本地对下载产物执行固定 PTOAS 0.65 的 `--pto-arch=a3 --enable-insert-sync` 已成功生成 `add_generated(float*, float*, float*)` C++。没有用手写 MLIR/PTO 替代编译输出。

完整模拟链已在 [Actions](https://github.com/Jopqior/tile-rs/actions/runs/36337430892) 通过，执行提交 `75d56a6b57b0102e05c24116b2edeb48cb9ee2b2`。两个 job 分别完成真实 Rust→MLIR→PTO 和 PTOAS→Bisheng→ACL/camodel→独立 CPU 比较，结果为 `256/256 exact matches, atol=rtol=0`。生成 C++ 的 SHA-256 为 `4b03c12b9582fdbf1dcd738fd36e743f8676b1a1bcaab3e0ccf49ee9c14a2661`。关键输出留在 `prototypes/rust-pto-ascend/evidence/`。

真机 job 默认关闭，在该运行中是 skipped，尚未执行；模拟通过不表述为驱动集成或真实芯片验证。本轮未另跑完整链的故障注入，比较器保留了旧原型已有的注入入口。

前端使用后端发布包 v0.0.2 和对应 release 源码 `c3c8ec0169bd02757b51370ba9c6ec3b116b833d` 中的 DSL/宏；CLI 和公开 PTO emitter 来自 upstream `6fbebdb0e7c1f6d453c32fe34f146fd6490decf1`。未修改发布包或配套 DSL。当前 DSL 与旧后端混用会失败，GmView 入口另有未定义 PTO 操作数；成功原型使用该 release 支持的 raw-pointer kernel 入口。具体运行和失败记录见 [前端核查](rust-pto-frontend-probe.md)。

原地图的调查固定在旧公开树 `e8d1acdf...`，是路径审阅和已有记录核查，没有复跑整条链。本原型补充运行证据，不把旧调查改写成当时已验证过。

## 文件与运行

- `.github/workflows/prototype-rust-pto-ascend.yml`：macOS Rust 前端、Ubuntu camodel，以及手动选择的 NPU job。
- `.github/workflows/prototype-rust-pto-frontend.yml`：实际 Rust 编译和当前源码 PTO 转换。
- `prototypes/rust-pto-ascend/frontend/add_generated.rs`：三个指针、1×256 f32 add。
- `prototypes/rust-pto-ascend/device/`：固定 PTOAS/PTO ISA、Bisheng 编译、ACL host、独立 Python 比较器。由 camodel 原型 `4d875e05` 的 host/checker 适配而来。

```sh
# 无物理 NPU；实际 Rust 产物进入 CANN simulator。
gh workflow run prototype-rust-pto-ascend.yml \
  --repo Jopqior/tile-rs --ref prototype/61-rust-pto-ascend-ci

# 仅在 runner 注册、权限与下表变量就绪后使用。
gh workflow run prototype-rust-pto-ascend.yml \
  --repo Jopqior/tile-rs --ref prototype/61-rust-pto-ascend-ci -f run_npu=true
```

## 真机接入合同

本原型的设备编译目标限定为 Ascend 910B 系列的 `dav-c220-vec`。camodel 是 Ascend910B1，使用 CANN 9.1.0；不同 SoC 或 Toolkit 不自动视为兼容。

设备管理员和目标仓库管理员共同把 Linux NPU host 注册为该仓库获准使用的 self-hosted runner，增加 `ascend-910b` 标签。runner 账户能使用分配的 NPU，机器预装匹配驱动/固件、CANN Toolkit/Bisheng、g++、Python 3、git。需要访问 GitHub Actions、artifact 服务和固定 PTO ISA 源码；不需要在 workflow 中提供 SSH 主机或私钥。

仓库 Actions variables：

| 名称 | 内容 |
|---|---|
| `ASCEND_CANN_ENV` | NPU host 上可 source 的实际 CANN 环境脚本绝对路径。示例 `/usr/local/Ascend/cann/set_env.sh`，按安装位置填写。 |
| `ASCEND_DEVICE_ID` | 分配给 runner 进程的逻辑 ACL 设备号，例如 `0`。填写实际映射，不假定等于物理卡编号。 |
| `ASCEND_SOC` | 实际 SoC，例如 `Ascend910B1`。此原型仅接受 `Ascend910B*`。 |

环境脚本负责设置 `ASCEND_HOME_PATH`、Bisheng 的 PATH 和真实 runtime 的库路径。设备 host 无需 Rust 后端或 PTOAS，前序 job 已上传生成 C++ 和摘要；NPU job 下载同一份源码，在目标 host 本地编译。它明确链接真实 `runtime`，ACL host 检查实际加载的关键符号不是 camodel，再选择配置的设备运行。设置错误或设备缺失均失败，不跳过为通过。

真机只由 `workflow_dispatch` 的 `run_npu=true` 启用。公共 PR 不自动调度设备。仓库 concurrency 防止本仓库重叠使用；与其他仓库共用设备时，还需设备方提供独占分配或外部互斥。

接入方式参照 [PyPTO self-hosted NPU job](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml#L1242-L1262) 与 [GitHub runner 注册说明](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)。上述三个变量是本原型实际读取的合同，不是 PyPTO 的变量原样照搬。
