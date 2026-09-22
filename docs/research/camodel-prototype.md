# camodel f32 add 无卡 CI 原型

对应票：[验证 camodel 的 f32 add 无卡正确性 CI](https://github.com/Jopqior/tile-rs/issues/59)。调查日期：2026-09-22。

## 当前状态

已在 GitHub 托管 Ubuntu 22.04 x86_64 runner 完成无驱动 Toolkit 安装。手写 PTO C++ 和 `.pto → PTOAS → C++` 两入口均经真实 Bisheng/camodel 执行，256/256 精确匹配；破坏输出后两 job 均因数值比较退出 1，恢复后再次通过。这个最小独立 kernel 的无卡 CI 可行，不是 CPU-SIM 或仅编译成功。

用户确认覆盖两个入口：独立手写 PTO C++，以及独立 `.pto → PTOAS → C++`。两者均须经设备编译器和 camodel 执行，与独立 CPU 参考比较，并故意破坏输出验证失败退出。不是 tile-rs 全链路测试。

分支：`prototype/59-camodel`。不接入 main，不申请 NPU，不发布 SDK 二进制。用户先授权下载和解包盘点，查看结果后另行确认采用无驱动完整 Toolkit 基线，继续安装并运行两入口正常/负例 CI。

## 对既有调查的更正

[PyPTO 固定 workflow](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/workflows/ci.yml) 的 `system-tests-a5sim` 在 GitHub 托管机器运行，但不能作为 camodel 先例。该版本的 runtime gitlink 指向 simpler `6e383fc57c12a8bd4dce045289b8b4f5d55a15f1`；[kernel compiler](https://github.com/hw-native-sys/simpler/blob/6e383fc57c12a8bd4dce045289b8b4f5d55a15f1/simpler_setup/kernel_compiler.py) 将 `a2a3sim`、`a5sim` 分派给 `HOST_GXX_15`，使用 `__CPU_SIM`。这是 PTO CPU-SIM，不是 CANN camodel。

`ghcr.io/hw-native-sys/pypto/github-ci:latest` 是 GitHub 容器镜像仓库中的环境包。[固定 Dockerfile](https://github.com/hw-native-sys/pypto/blob/8a944cf211a847b9d39d35a38791fd5d5bd2120d/.github/docker/github_ci.Dockerfile) 不含 CANN/camodel/Bisheng 安装步骤。本次匿名 token endpoint 返回 401，未取得 digest、未拉取镜像。无须继续追此镜像来完成 camodel 实验。

## 真正 camodel 的公开入口

[PTOAS simulator workflow](https://github.com/hw-native-sys/PTOAS/blob/66bd855ed4a860df11c14393a6e64219c48aa723/.github/workflows/ci_sim.yml) 在自托管 Linux X64 runner 使用预装 CANN，搜索 `*/simulator/dav_3510/lib`。这不是全新 GitHub 托管机器上的 SDK 获取/安装配方；workflow 中也混有 CPU-SIM 步骤，不能整体统称 camodel。

[PTO ISA A2/A3 ST CMake](https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/tests/npu/a2a3/src/st/testcase/CMakeLists.txt) 的 vector kernel 使用 Bisheng `dav-c220-vec`；sim host 链接 `runtime_camodel`。目标候选为 A2/A3 普通 f32 add，不要求 A5。具体 SoC 目录与工具版本仍须按安装包实物匹配。

## Toolkit 获取实证

[AscendNPU-IR 固定 wheel workflow](https://github.com/Ascend/AscendNPU-IR/blob/8d49415f4d75d53dc983353e49b6c99004b0f17c/.github/workflows/wheel-x86_64.yml) 引用 CANN 9.1 官方 OBS 下载。它提取 Bisheng 做构建，不是 camodel 运行证据。

本次从同一官方 bucket 下载独立 Toolkit，无凭据，未安装驱动或固件：

```text
https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-x86_64.run
size: 1298374045 bytes
sha256: 985b8c7b68a5f85af7c28c3514f3d3f6baec1e0784669f2bba0cce2b502dabe9
```

此 SHA256 是下载后计算的内容固定标识，不是独立官方签名认证。官方 east-2 endpoint 的同版本同名包大小不同，不能无条件互换下载端点或 hash。

[官方安装参数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0043.html) 说明 Toolkit 支持无驱动 `--full` 安装。实物外层脚本的 `devtools` 白名单仅包含 oam-tools、asc-tools，不含 simulator，不能据文档摘要用这个白名单代替完整安装。

## 解包库存

只使用 Python 安全提取和系统 readelf 静态读取，未执行包内程序。Bisheng、npu-runtime、simulator 的 version.info 均声明 9.1.0，时间戳 `20260730_231653901`。这些不是实际执行的版本输出。

- Bisheng ELF SHA256：`647f5772aab1b93dd59f088721c94c3edae1212c92b6dc4117b3c5742548252a`。
- A2/A3 `libruntime_camodel.so`：x86-64 ELF，SHA256 `f2dcf3deaf635d217b64747d15ba2bb72e52682f654188f684e150a681464415`。
- simulator 安装 manifest 明确映射原始 `lib64/Ascend910B1/lib` 到 `x86_64-linux/simulator/dav_2201/lib`，并创建 Ascend910B1 等别名；不是猜测软链接。
- ACL 存在 `libascendcl.so → libascend_dump.so → libruntime.so / libascend_hal.so` 的静态依赖链。不能仅凭链接 `runtime_camodel` 就断言实际运行不会误用硬件 runtime，须以动态加载和无卡执行核实。
- 完整 Toolkit 已在隔离 Ubuntu 22.04 容器完成 root 和非 root 安装；未安装到开发机全局环境，容器无特权、无 NPU 设备透传。
- 实际 Bisheng banner：clang 15.0.5，clang/flang build `5c68a1cb1231`，时间戳 `2026-07-30T20:53:21+08:00`。旧 PTO ISA pin 与 9.1 的两种最小 kernel 均兼容。
- 本地私有动态绑定检查确认关键 ACL/launch 调用解析到 camodel；硬件 runtime/hal 共享库仍会间接加载。原型 host 检查五个关键 rt 符号的进程全局解析，不能写成“硬件 runtime 库完全未加载”。

本地 SDK 与解包证据保留在 `/home/whh/.cache/camodel-prototype-59/`，不入 Git。

## 许可边界

外层实际安装脚本展示 `script/eula_cn.txt` 的企业 EULA（2023-11-09）。simulator 的 `cannsim-0.1.0` wheel 内另含 CANN Open Software License v2.0，不能据此把全部闭源模型 ELF 当作相同许可。已读包内文本没有明确点名禁止自有正确性 PASS/FAIL 发表，但组件细分归属及用途仍不是法律保证。

公开网页 [CANN 软件许可](https://www.hiascend.com/legal/cannua-download)、[软件许可协议](https://www.hiascend.com/legal/softlicense) 和企业 EULA 不是同一协议。第二项的数据披露条款未在本次盘点中绑定到具体 camodel ELF，也不能保证它必不适用。用户获知这些边界后确认继续独立 Ascend kernel 实验。

公开下载 URL 不代表可以再分发。实验不上传 Toolkit、Bisheng、camodel 库或打包镜像，也不将它们放入公开缓存；不发布模型内部 trace、寄存器 dump 或性能测量。

## GitHub Actions 实测证据

下表每次均独立运行 handwritten/generated 两个 matrix job；正常和恢复时两者成功，负例时两者都在比较步骤失败，安装成功且没有 skip 或 continue-on-error。

| 运行 | 固定提交 | 结果 |
|---|---|---|
| [正常](https://github.com/Jopqior/tile-rs/actions/runs/35761288167) | `135f2ddbc4f3e35ec560c17de534d642912a97c5` | 两入口 256/256 精确匹配 |
| [负例](https://github.com/Jopqior/tile-rs/actions/runs/35761853455) | `1b14de4bc63641581cc6a1dd0c4c94c1a9b40d48` | 首元素加 1，actual=1.0、reference=0.0，两 job exit 1 |
| [恢复](https://github.com/Jopqior/tile-rs/actions/runs/35762428063) | `02325b6b675c79ca956fee6aef4acb98d7bd1232` | 两入口再次精确匹配 |

[首次 workflow 验证失败](https://github.com/Jopqior/tile-rs/actions/runs/35761177708)发生在 runner 分配前：job 级 env 不接受 `runner.temp`。将路径设置移入首个步骤后修复，不算 camodel 运行失败或负例证据。

已下载六份数值 artifact 再次独立重算，两个入口的正常/负例/恢复分别只有 0/1/0 个不匹配元素，负例仅 index 0。三次 generated.cpp 字节一致。长期证据保存在 [camodel-evidence/](camodel-evidence/)：输入输出、比较文本、host/工具版本和未修改生成 C++；Actions artifact 保留 14 天，链接日志也受平台保留期限制。

GitHub runner inventory：`ImageOS=ubuntu22`，`ImageVersion=20260907.292.1`，x86_64；Python 3.10.12，GCC/G++ 11 默认发行版工具，Bisheng clang 15.0.5。无 exposed Ascend 设备，未安装 driver/firmware。CANN/模型 ELF hash 与上述本地包库存一致。

## 数值合同与两入口

- 输入是两个 f32 `[1,256]` tile，含零、正负数、抵消及非整数二进制精确分数。独立 Python 标量相加作为参考；输入及和在 binary32 中精确可表示，`atol=rtol=0` 合理，但不是通用浮点误差规范。
- 输出先写入 NaN，返回后检查长度和有限性，再比较全部元素；没有把首次模拟输出当 golden。
- 手写 kernel 使用 PTO TLOAD/TADD/TSTORE 和设备流水同步；generated 使用 PTOAS 0.65 `--pto-arch=a3 --enable-insert-sync` 生成 C++，原样 include，再由 Bisheng `dav-c220-vec` 编译，在 Ascend910B1/dav_2201 camodel 执行。不是所有 PTOAS target 的覆盖证明。
- 两者共用 ACL host 和独立 oracle，检查 ACL 调用、launch last-error、stream sync，任何失败非零退出；没有 `__CPU_SIM` 定义。
- 负例只破坏返回数值并验证 oracle/CI 的失败传播，不是向设备 kernel 注入错误，也不证明能发现所有可能的 kernel 缺陷。
- host 强制检查五个关键 rt 符号全局解析到 camodel；真实调用的额外动态绑定核对来自本地私有日志，不是公开 CI 调用跟踪证据。

## 可复用条件和覆盖边界

复用入口见 [原型 README](../../prototypes/pto-camodel/README.md)。需要获准使用的 CANN、官方 OBS/GitHub/Ubuntu/Python 软件源网络访问、x86_64 Ubuntu 22.04、至少 20 GiB 安装前可用空间和匹配工具链。安装完整 Toolkit，没有镜像预装、SDK cache、真实 NPU 或外部账号凭据；软件分发和使用仍遵循许可。包/PTOAS wheel 用固定 SHA256，PTO ISA 固定 commit；Ubuntu 发行版依赖仍可更新，不承诺逐字节重现全部环境。

本次成功只覆盖固定 shape 和精确可表示输入的 f32 add，不覆盖 NaN/Inf/subnormal 算术、其他 dtype/shape/指令、一般舍入误差、并发/多卡通信、真实驱动/硬件正确性或性能。独立手写 IR 不是 tile-rs 产物，不证明 Rust→PTO 链路或正式门禁可直接启用。

模型独立 build 版本、其他 host/SoC/CANN 版本的兼容性和闭源组件更细许可归属仍未解决，不能从本次 PASS 外推。正式 CI 设计、依赖裁剪和生产 runner 选择不在本票范围。
