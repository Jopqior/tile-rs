# 请求 Ascend NPU 真机访问，用于 tile-rs→PTO 正确性 CI

为 tile-rs→PTO→Ascend NPU 链路的正确性 CI，请求一个可供 GitHub Actions 使用的 Ascend 910B 真机执行环境。

## 接入后运行的 workflow

[完整 workflow](https://github.com/Jopqior/tile-rs/blob/75d56a6b57b0102e05c24116b2edeb48cb9ee2b2/.github/workflows/prototype-rust-pto-ascend.yml) 已准备在 fork 中。[无卡运行](https://github.com/Jopqior/tile-rs/actions/runs/36337430892) 已从真实 Rust add 生成 MLIR/PTO，经 PTOAS、CANN 编译后在 camodel 执行，并通过独立 CPU 参考比较。

真机接入后，手动运行同一 workflow 并选择 `run_npu=true`。下面的 job 会接收前序 job 生成的 C++，在 NPU host 上编译、执行并比较结果。调用的 [run.sh](https://github.com/Jopqior/tile-rs/blob/75d56a6b57b0102e05c24116b2edeb48cb9ee2b2/prototypes/rust-pto-ascend/device/run.sh)、ACL host 和比较器均已在原型分支中；真机分支尚待有卡后验证。

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
      EXECUTION_MODE: npu
      CANN_ENV: ${{ vars.ASCEND_CANN_ENV }}
      ASCEND_DEVICE_ID: ${{ vars.ASCEND_DEVICE_ID }}
      ASCEND_SOC: ${{ vars.ASCEND_SOC }}
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          persist-credentials: false
      - uses: actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093
        with:
          name: rust-pto-device-input
          path: device-input
      - name: Check generated source and set paths
        run: |
          (cd device-input && sha256sum -c generated.sha256)
          echo "BUILD_DIR=$RUNNER_TEMP/rust-pto-npu-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" >> "$GITHUB_ENV"
          echo "GENERATED_CPP=$GITHUB_WORKSPACE/device-input/generated.cpp" >> "$GITHUB_ENV"
      - name: Compile, execute on NPU and compare with CPU reference
        run: bash prototypes/rust-pto-ascend/device/run.sh
```

上面摘录的是完整文件中的 NPU job，省略了末尾的日志上传步骤。

## 接入需要提供什么

采用 [PyPTO 的 self-hosted NPU runner 接入方式](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml#L1242-L1262)：将设备 host 注册或授权给执行 workflow 的仓库，并配置 `self-hosted`、`linux`、`ascend-910b` 标签。runner 账户需要访问分配的 NPU；机器需能连接 GitHub Actions 和下载构建依赖。无需在 workflow 中提供 SSH 地址或私钥。

需要填写的仓库 Actions variables：

| 变量 | 需要提供的值 |
|---|---|
| `ASCEND_CANN_ENV` | 机器上 CANN 环境脚本的绝对路径，例如 `/usr/local/Ascend/cann/set_env.sh`。 |
| `ASCEND_DEVICE_ID` | 分配给 runner 的逻辑 ACL 设备号，例如 `0`。 |
| `ASCEND_SOC` | 实际 SoC，例如 `Ascend910B1`。 |

还需确认 host 架构、驱动/固件版本，以及与 CANN 9.1.0 配套的设备环境。机器预装 CANN Toolkit/Bisheng、g++、Python 3 和 git 即可；Rust 编译和 PTOAS 在前序托管 runner 上完成。当前真机 job 面向 910B 系列，仅手动触发，不自动运行外部 PR。
