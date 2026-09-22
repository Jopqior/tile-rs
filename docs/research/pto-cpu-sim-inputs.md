# PTO CPU-SIM 接受什么程序输入

调查日期：2026-09-22。仅审阅第一方文档和源码，未编译或执行 kernel、未触发 CI、未更新 issue。ISA 本地 HEAD 已核对为 `82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19`；PTOAS 另取固定快照 `97e9ecb247744c587171c73ab58ac2423bba3197`。**两者分别固定，不代表上游认证的兼容组合。**既有 `pto-toolchain.md`、`tile-rs-pto-path.md` 仅作背景；下列结论回溯到固定上游源文件。

## 结论与输入边界

CPU-SIM 是 **PTO C++ 库的 CPU 后端**，不是接收任意 PTO 文件或 NPU 二进制的解释器。官方启用合同是普通 CPU 编译器 + `__CPU_SIM`；`pto-inst.hpp` 据此包含 CPU stubs / 指令路径。[I1][I2]

| 程序产物 | 能否直接作为 CPU-SIM 程序输入 | 实际接口 |
|---|---|---|
| 手写 PTO Tile Library C++ | 是，作为待编译源码；不是交给模拟器命令行解释 | 包含 PTO headers，用 GCC/Clang 编译 CPU executable，host 调用 kernel。[I1][I3] |
| `.pto`，即 MLIR PTO IR | 不能直接执行 | 先由 PTOAS lower 为调用 PTO 库的 C++，再接 CPU harness。上游明确记录此工作流。[A1][A2] |
| PTO-BC / `.ptobc` | 未见 CPU-SIM 直接 reader/执行入口 | 文档化接口为 `ptobc decode out.ptobc -o out.pto`，然后走上行。PTO-BC 是编码/解码格式，不是 CPU 或 NPU 机器码。[A3] |
| PTOAS 生成的 C++ | 是候选源码，不保证原样可编译运行 | 检查生成签名、includes、目标宏和指令覆盖，补薄 launch wrapper / host I/O。不是所有产物只加 `__CPU_SIM` 就够。[A2][A4] |
| NPU object、device binary、ACL 打包 `.so` | 不是该 CPU 后端的程序入口 | 应取得 C++ 并重新做 CPU 编译；NPU 编译、launch 属另一链。没有依据声称 CPU-SIM 装载并模拟设备 ISA 二进制。[I1][I3][A1] |

这里“不能直接”限定于所审阅、文档化的 CPU-SIM 接口，不是对所有外部模拟器作不存在证明。CPU 编译得到的 host executable / host library 与设备二进制也不可混称。

## 文档化 lowering 与适配路线

```text
手写 PTO C++ ───────────────────────────────┐
.ptobc → ptobc decode → .pto → ptoas → C++ ─┤
                                          ↓
               CPU harness + PTO headers + __CPU_SIM
                                          ↓
                        CPU 编译、链接 → host 执行与 oracle 比较
```

PTOAS README 给出 `ptoas input.pto -o output.cpp`、可选 `--enable-insert-sync`、`--pto-arch=a5`；生成实现确有 EmitC preparation、`emitc::translateToCpp` 和 C++ 后处理。[A1][A5] 这描述的是 **EmitC/C++ 路线**，不把当前 PTOAS 的其他 backend 都概括成 C++。

尤其重要：PTOAS 自带 CPU-SIM 集成说明明确要求先生成 C++，再移植到 `pto-isa/tests/cpu/st/testcase/<case>`，或使用 standalone harness；建议尽量保留 kernel body、用薄 wrapper 对齐参数与 host。它还专门说明无 GM 输入/输出、仅有内部 tile 地址的生成函数不适合机械套用 tadd harness。[A2][A4] **因此“存在文档化适配路线”已核实；“这份生成 C++ 与选定 ISA pin 兼容、数值正确”未测试。**文档提到的本地 scratch 示例不当作本次成功执行证据。

## 后续接入所需的具体接口合同

1. **产物与版本。**保留 `.pto`、原始生成 C++、PTOAS/ISA commit、完整 flags、明确 arch 与 build level，以及 wrapper 的差异。README 写 `level3` 禁用 PlanMemory/InsertSync；不能随意删掉地址分配或同步代码来“让 CPU 通过”。[A1]
2. **CPU 构建。**提供 PTO `include`，定义 `__CPU_SIM`，使用可满足库要求的 CPU C++ 工具链。官方 standalone CMake 用 C++20（GCC≥14 时用23）、Threads，Unix 的 `m`、Linux 的 `dl`；此分支在进入 CANN/Bisheng 配置前返回，不要求设备 SDK。[I3] stubs 已将 `AICORE`、`__global__`、`__gm__` 等处理为 CPU 可用形式，不需要一概删除这些标记；但不能据此认定所有 CCE builtin/launch 语法均被兼容。[I2][I4]
3. **调用与数据 ABI。**明确符号名、参数顺序/类型、shape/stride/valid shape、buffer 大小与归属、输出可观察位置。CPU tadd 的 `LaunchTAdd` 直接调用普通 C++ kernel，并对 half 作类型适配；不是 NPU `<<<...>>>` launch。[I5] 若保留 ACL 风格 host，使用受支持 CPU stubs，或替换为 CPU 分配/拷贝，不能继续依赖真实设备 loader。[I1][A4]
4. **内存与目标。**普通 Tile 必须 `TASSIGN`，或显式选择 `__PTO_AUTO__` lazy storage。后者是独立 host 内存，不保真模拟 UB/L1 等地址、aliasing。CPU 默认 A2A3；需要 A5 时，在使用 Tile/启动线程之前调用对应 `NPU_MEMORY_INIT` / arch 初始化。PTOAS 的 `--pto-arch=a5` 或外部 `a5sim` 名称不自动配置 CPU runtime。[I1]
5. **执行上下文。**单函数直接调用不能自动获得多核调度；依赖 block/subblock 的 kernel 需 `LaunchKernelMultiCore` 或等价 runtime 上下文。A2/A3 FFTS 还要求明确 `__DAV_CUBE__` / `__DAV_VEC__` 角色、公共同步 API、跨 worker 共享存储与 lane identity；旧生成器输出 compiler-private FFTS 名称需修改生成器后重生成，不能空 stub 过去。[I1]
6. **测试入口与 oracle。**采用 CPU ST 时需 `<case>_kernel.cpp`、`main.cpp`、`gen_data.py`、CMake 注册；生成数据的工作目录和 `Suite.case` 目录名须匹配。standalone 则自供输入、调用与独立 reference、容差及失败退出。`run_cpu.py --testcase <case>` 选的是 C++ 测试工程，**不是 `.pto` 文件参数**。[A2][A4]

## 不可外推的兼容性与语义

- CPU op 同步执行，线程共享 Tile 有限制；`SYNCALL` 是兼容 no-op，不能当 CPU worker barrier。GlobalData FIFO 流未提供，CCU gather/scatter 不具备功能实现；不同 dtype/layout/op 必须逐项查覆盖。[I1]
- CPU 与 NPU 的 scratch、布局检查、浮点 reduction 和 BF16 支持存在差异；文档不保证完整 bit-exact。CPU PASS 不证明设备同步、性能或目标编译正确。[I1]
- 本次未运行 PTOAS、CPU compiler、tile-rs 后端或 CI。**不证明 tile-rs 的旧 `.pto` 被此 PTOAS 接受，不证明其留存 C++ 能接此 ISA，也不证明 Rust→PTO→CPU-SIM 路线已通。**后续最小待验对象应是一份有来源的 IR/生成 C++ 加明确 ABI、目标与独立 oracle，而不是“PTO 支持”这一总标签。

## 固定来源

[I1]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/docs/coding/cpu_sim.md
[I2]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/include/pto/pto-inst.hpp
[I3]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/kernels/custom/fused_add_relu_mul/CMakeLists.txt
[I4]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/include/pto/common/cpu_stub.hpp
[I5]: https://github.com/hw-native-sys/pto-isa/blob/82361dd56d4ea5d0dc99f14028fe5eb8b5c46d19/tests/cpu/st/testcase/tadd/tadd_kernel.cpp
[A1]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/README_en.md#L247-L306
[A2]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.agents/skills/pto-isa-cpu-sim-kernel-test/SKILL.md
[A3]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/tools/ptobc/README.md
[A4]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/.agents/skills/pto-isa-cpu-sim-kernel-test/references/cpu-st-patterns.md
[A5]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/tools/ptoas/ptoas_pipeline.cpp#L1569-L1655
