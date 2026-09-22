# PTO CPU-SIM f32 add 无卡 CI 原型

实验日期：2026-09-22 UTC。对应[验证 PTO CPU-SIM 的 f32 add 无卡正确性 CI](https://github.com/Jopqior/tile-rs/issues/58)。资产分支：`prototype/58-pto-cpu-sim`。技术结果已取得，票据结论待现场审阅确认。

## 观察结果

真实上游 PTO CPU-SIM 可在 GitHub-hosted Ubuntu x86_64 runner 上执行独立 f32 add kernel，并与 Python 标量参考比较。正常、负例、恢复三次运行分别为 success、failure、success。负例失败来自结果不匹配，不是下载、编译或启动故障。

| 实验 | 固定提交 | Actions | 观察 |
|---|---|---|---|
| 正常 | `71202dbda5fbdb06cc4786edc2988d42e462464e` | [正常运行](https://github.com/Jopqior/tile-rs/actions/runs/35751052050) | 256/256 精确匹配，success |
| 破坏输出 | `59a075ff67e4947e2316b9398a58c93da407b3c9` | [负例运行](https://github.com/Jopqior/tile-rs/actions/runs/35751118666) | 首元素 actual=1、reference=0，退出 1，failure |
| 恢复 | `040743e4d6026ca13b7c10d2f6a45056d2bc6662` | [恢复运行](https://github.com/Jopqior/tile-rs/actions/runs/35751216271) | 256/256 精确匹配，success |

负例与正常实现的唯一代码差异是 workflow 环境变量 `INJECT_ERROR` 从 0 改为 1；恢复提交将其改回 0。没有 `continue-on-error`，也没有跳过比较。

## 版本与资源

- PTO ISA 固定到 `82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19`，下载后校验 HEAD；上游头文件原样使用。
- checkout action 固定到 `11bd71901bbe5b1630ceea73d27597364c9af683`（v4.2.2）。GitHub 日志提示该 action 的 Node 20 声明被平台按 Node 24 运行，未阻塞原型。
- 正常运行的 runner 镜像为 `ubuntu-24.04` / `20260920.314.1`，x86_64，Ubuntu 24.04.5，内核 `6.17.0-1022-azure`。
- GCC / libstdc++ 开发包 `13.3.0-6ubuntu2~24.04.1`；Python `3.12.3`，系统包 `3.12.3-0ubuntu2.1`；Git `2.55.0`，系统包 `1:2.55.0-0ppa1~ubuntu24.04.2`。
- 未安装 CANN、驱动、固件、camodel、NumPy 或 GoogleTest。只用 runner 已有编译器、Python 标准库、Git，以及公开上游下载。
- 环境检查未发现 `/dev/davinci*`、`/dev/devmm_svm`、`/dev/hisi_hdc` 或 `/usr/local/Ascend*`；PCI 清单仅列出 Microsoft 网络与 NVMe 控制器。结合 GitHub-hosted runner 与纯 CPU 编译路径，支持本实验不依赖真实 NPU。它不是对云供应商底层物理资产的盘点。
- 源码/action 已按 SHA 固定；系统镜像与包版本仅记录实际值，未按容器 digest 或完整工具链 lock 冻结。后续复现须重新记录环境，不能称逐位可复现。

## 输入、比较及运行方式

代码位于 [`prototypes/pto-cpu-sim/`](../../prototypes/pto-cpu-sim/)，专用 workflow 位于 [`.github/workflows/prototype-pto-cpu-sim.yml`](../../.github/workflows/prototype-pto-cpu-sim.yml)。

```sh
bash prototypes/pto-cpu-sim/run.sh
INJECT_ERROR=1 bash prototypes/pto-cpu-sim/run.sh # 预期失败
```

`CXX` 默认 `g++-13`，编译参数 `-std=c++23 -O0 -D__CPU_SIM`。单 tile，f32，shape `[1,256]`，调用真实上游 `TLOAD → TADD → TSTORE`，没有 mock。独立 Python 程序生成输入文件并以标量加法计算参考，kernel 只读输入、写输出。

输入是小范围的 1/8、1/16 倍数，包含正负、零、抵消和分数；输入与和均可在 f32 精确表示。比较长度、有限性及全部 256 个值，`atol=rtol=0`。输出先填 NaN，未写入的结果不能误通过。Python 的 double 精度在这一输入集上不造成 oracle 舍入差异。该精确比较规则不推广到任意浮点加法。

负例在真实 kernel 完成后读取输出，把第 0 个值加 1，再交同一个比较器。这验证数值错误会让 CI 变红，不证明所有种类的 kernel bug 都会被当前输入捕获。

本地另用 GCC 15.2.0 跑过相同正常与负例。首次编译发现上游头文件依赖先提供整数类型声明；本原型在 PTO include 前包含 `<cstdint>` 和 `<cstddef>`，没有修改上游代码。

## 复用条件与边界

可作为 tile-rs 后续无卡功能测试的最小参照：前提是待测产物能调用该 CPU-SIM 支持的 PTO 接口，并为实际 shape、dtype、算子语义另设输入和独立参考。这次未测试 Rust→PTO 编译链，没有加载最终 NPU 二进制，也未测试 CANN camodel、真机同步、性能或多卡。

workflow 仅在 throwaway 分支的原型路径 push 时执行，不成为 main 正式门禁。无需新凭据、付费资源或外部授权。

## 日志保留

GitHub 日志有保留期。以下文本保存从三次完整日志中筛出的原始环境及结果行，不冒充完整日志：

- [正常日志节选](pto-cpu-sim-evidence/positive.txt)
- [负例日志节选](pto-cpu-sim-evidence/negative.txt)
- [恢复日志节选](pto-cpu-sim-evidence/restored.txt)
