# PTO CPU-SIM add 原型

Throwaway，仅回答[验证 PTO CPU-SIM 的 f32 add 无卡正确性 CI](https://github.com/Jopqior/tile-rs/issues/58)。不接入 main，不是 Rust→PTO 链路或 NPU 二进制验证。

## 运行

需要 Git、Python 3、GCC 13（C++23）和 GitHub 网络访问，无 pip、CANN、GoogleTest 或 NPU 依赖。

```sh
bash prototypes/pto-cpu-sim/run.sh
INJECT_ERROR=1 bash prototypes/pto-cpu-sim/run.sh # 预期退出 1
```

`CXX` 可覆盖编译器。脚本在临时目录下载固定 PTO ISA commit `82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19`，以 `-D__CPU_SIM -std=c++23 -O0` 编译，结束清理临时目录。上游源码及许可证原样保留，不复制模拟实现。

## 实验

- 独立 C++ kernel 调用真实上游 `TLOAD → TADD → TSTORE`，单 tile，f32，shape `[1,256]`。
- Python 标量参考端生成 256 对正、负、零、抵消及分数输入，通过文件传给 kernel；参考不调用 PTO，也不以 kernel 输出生成 golden。
- 输入为小范围 1/8、1/16 的倍数，和可在 f32 精确表示。因此逐元素精确比较，`atol=rtol=0`，并检查长度与有限值。此规则不能推广到任意浮点输入或其他算子。
- 负例在 kernel 执行完成后将输出首元素加 1，再走同一比较器，非强制退出、编译失败或模拟器 mock。workflow 不吞掉失败。
- workflow 只在本 throwaway 分支的原型文件 push 时运行。使用 GitHub-hosted `ubuntu-24.04`，记录 runner 镜像、PCI 清单、设备节点、安装目录及软件版本。没有暴露的 Ascend 设备且无需设备依赖，不声称检查了云宿主的物理资产。
- PTO ISA 和 checkout action 按 SHA 固定；runner 镜像及系统软件随 GitHub 更新，实际版本留在日志，不宣称整个环境按 digest 冻结。

## 边界

成功仅说明该固定模型和输入集的最小功能正确性 CI 可行。不覆盖 CANN camodel、设备目标编译、硬件时序、同步、性能、多卡、所有 shape/dtype，或 tile-rs 端到端编译链。

正常、负例与恢复运行的链接、提交及环境版本见[实验记录](../../docs/research/pto-cpu-sim-prototype.md)。
