# 请求 Ascend NPU 真机访问，用于 tile-rs→PTO 正确性 CI

为 tile-rs→PTO→Ascend NPU 链路的正确性 CI，请求一个可供 GitHub Actions 使用的 Ascend 910B 真机执行环境。

## 接入后运行的 workflow

[完整 workflow](https://github.com/Jopqior/tile-rs/blob/7da5fb0fb5a921c0b4120a5bb8cd9b70584fd5a5/.github/workflows/prototype-rust-pto-ascend.yml) 已准备在 fork 中。[最新无卡运行](https://github.com/Jopqior/tile-rs/actions/runs/36399385961) 在提交 `7da5fb0fb5a921c0b4120a5bb8cd9b70584fd5a5` 上通过离线测试、Rust 前端编译和 camodel 执行。真实 Rust add 生成 MLIR/PTO，经 PTOAS、CANN 编译后执行，256 个 f32 输出全部与独立 CPU 参考精确一致。该次运行的 NPU job 按 push 触发规则跳过，尚未验证物理设备。

用例统一放在 `cases/<name>/`：Rust kernel 和前端期望是必需文件，需要执行验证的用例另行提供 `device/kernel.cpp`、`host.cpp` 和 `check.py`。runner 自动发现用例，统一校验产物、编译执行并汇总结果；当前真实执行用例是 `add_generated`。目录结构和新增用例方式见[测试总览](https://github.com/Jopqior/tile-rs/blob/7da5fb0fb5a921c0b4120a5bb8cd9b70584fd5a5/prototypes/rust-pto-ascend/README.md)。

真机接入后，手动运行同一 workflow 并选择 `run_npu=true`。下面的 job 接收模拟器阶段逐用例生成的 C++，校验后在 NPU host 上重新编译、执行并比较结果。它不重新生成 PTO，也不复用模拟器二进制。[运行脚本和产物说明](https://github.com/Jopqior/tile-rs/blob/7da5fb0fb5a921c0b4120a5bb8cd9b70584fd5a5/prototypes/rust-pto-ascend/device/README.md)已在原型分支中；真机路径尚待有卡后验证。

```yaml
  npu:
    if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_npu }}
    needs: simulator
    runs-on: [self-hosted, linux, ascend-910b]
    timeout-minutes: 30
    concurrency:
      group: ascend-910b-device
      cancel-in-progress: false
    env:
      CANN_ENV: ${{ vars.ASCEND_CANN_ENV }}
      ASCEND_DEVICE_ID: ${{ vars.ASCEND_DEVICE_ID }}
      ASCEND_SOC: ${{ vars.ASCEND_SOC }}
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4
        with:
          persist-credentials: false
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4.3.0
        with:
          name: rust-pto-device-input
          path: device-input
      - name: Set isolated directories
        run: |
          echo "BUILD_DIR=$RUNNER_TEMP/rust-pto-npu-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" >> "$GITHUB_ENV"
          echo "TOOLS_DIR=$RUNNER_TEMP/rust-pto-tools-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" >> "$GITHUB_ENV"
      - name: Install pinned PTO headers
        run: bash prototypes/rust-pto-ascend/device/install-tools.sh npu "$TOOLS_DIR"
      - name: Run NPU suite using the same generated sources
        run: |
          bash prototypes/rust-pto-ascend/device/run.sh \
            --mode npu --input "$GITHUB_WORKSPACE/device-input" \
            --output "$BUILD_DIR" --tools "$TOOLS_DIR"
```

上面摘录的是完整文件中的 NPU job，省略了末尾的日志上传步骤。公开产物按用例保存，包含生成源码及校验和、编译日志和数值结果；SDK、安装日志和原始运行时日志不上传。

## 接入需要提供什么

采用 [PyPTO 的 self-hosted NPU runner 接入方式](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml#L1242-L1262)：将设备 host 注册或授权给执行 workflow 的仓库，并配置 `self-hosted`、`linux`、`ascend-910b` 标签。runner 账户需要访问分配的 NPU；机器需能连接 GitHub Actions 和下载构建依赖。无需在 workflow 中提供 SSH 地址或私钥。

需要填写的仓库 Actions variables：

| 变量 | 需要提供的值 |
|---|---|
| `ASCEND_CANN_ENV` | 机器上 CANN 环境脚本的绝对路径，例如 `/usr/local/Ascend/cann/set_env.sh`。 |
| `ASCEND_DEVICE_ID` | 分配给 runner 的逻辑 ACL 设备号，例如 `0`。 |
| `ASCEND_SOC` | 实际 SoC，例如 `Ascend910B1`。 |

还需确认 host 架构、驱动/固件版本，以及与 CANN 9.1.0 配套的设备环境。机器预装 CANN Toolkit/Bisheng、g++、Python 3 和 git 即可；Rust 编译和 PTOAS 在前序托管 runner 上完成，真机 job 仅获取固定版本 PTO-ISA 头文件。当前真机 job 面向 910B 系列，仅手动触发，不自动运行外部 PR。相关 PR 只跑无需 SDK 和硬件的离线测试；同一并发组的真机任务串行执行，不取消正在运行的任务。
