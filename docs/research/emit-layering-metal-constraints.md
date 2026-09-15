# 多目标 emit 分层机制与 Metal 约束

服务于研究票：[研究适用于 tile-rs 多后端 emit 的分层机制与 Metal 约束](https://github.com/Jopqior/tile-rs/issues/15)，所属地图：[多后端 emit 分层探索：先厘清领域模型，再形成整体判断](https://github.com/Jopqior/tile-rs/issues/12)。

研究资产分支：`research/15-emit-layering-metal`；路径：`docs/research/emit-layering-metal-constraints.md`。领域词汇与现状地图在 `wayfinder/12-emit-layering`，本分支不复制它们。

本票是 AFK research。记录外部编译器与 MLIR 生态里能解释前置调查问题的机制，以及 Metal 目标链路的真实支持、限制和引入成本。不选择分层方案，不制定实施计划，不修改生产代码。

## 证据边界

- 主仓库对照版本：`Jopqior/tile-rs@45a3cd148bba0c953a8b03071038ea46a9d08b93`（`wayfinder/12-emit-layering` HEAD）。职责调查资产：[emitter-responsibilities.md](https://github.com/Jopqior/tile-rs/blob/45a3cd148bba0c953a8b03071038ea46a9d08b93/docs/analysis/emitter-responsibilities.md)；领域地图：[tile-rs-source-map.md](https://github.com/Jopqior/tile-rs/blob/45a3cd148bba0c953a8b03071038ea46a9d08b93/docs/domain/tile-rs-source-map.md)。
- 外部源码与文档抓取日期：2026-09-15。固定版本见文末「外部版本」。行号相对这些 SHA，不随后续提交更新。
- 只读官方文档、规范与公开源码。未运行 `mlir-opt`、`llc`、SPIRV-Cross、naga、`xcrun metal`、IREE 或任何目标硬件。
- 当前公开生成器的输入是 MLIR **文本**（`llvm.func`、`llvm.call @__tile_*`、`hacc.entry`），由 `parse_module` 做文本解析。不能假定这份输入可被上游 `mlir-opt` 当完整 LLVM 方言模块消费，也不能把生成器文件头注释里的工具命令当成已执行的目标编译。
- 不把 GPU/NPU/TPU 名称当作抽象成立的证据。不把生态中存在某个 dialect 当作已有可用 Metal 通路。
- 聚焦能解释前置职责问题的候选，不穷举框架。Triton、TVM、XLA、Mesa NIR 等未纳入，不是因为已证无关，而是超出本票边界。

### 证据分级

| 标记 | 含义 |
|------|------|
| **一手事实** | 官方文档、规范或固定 SHA 源码里读到的声明、目录、类型或调用 |
| **适用性推断** | 把外部机制对照 tile-rs 现状得出的含义，不是外部项目自己的承诺 |
| **未核实** | 逻辑上可能，但本次未解析、未编译、未上机 |

---

## 1. 前置调查钉住的问题

#14 已确认的现状（决议：[issuecomment-5676642389](https://github.com/Jopqior/tile-rs/issues/14#issuecomment-5676642389)）：

- 已有共享：`mlir_parse` 文本解析与助手；MUSA 对 CUDA 的整生成器委托；`CodegenTarget::emit` 调用与输出包装。这些共享的是解析结果和调用缝，不是各目标内部的操作分析。
- 跨目标维护联动来自独立的操作识别、参数读取和组合分析。SiLU→Mul 检测规则已经分叉，不能把差异全部归因于硬件或实现缺陷。
- Metal 体量包含专用计算正文，以及分类、操作链分析、入口约定和输出选择之间的联动。减少文件行数不等于减少这些成本。
- 值得研究的共性：共同输入语义及数据依赖分析，能否与目标实现选择分别变化。
- 不能被简单抹平的差异：目标原语、存储与同步、入口和工作分配契约、由哪个阶段展开算法。现存算法选择不全是硬件必需。

领域地图对输入与 Metal 终点的约束（#13）：

- 公开生成器消费的是 LLVM 方言写法的 MLIR 文本，函数体主要是文本行，不是完整 MLIR API 对象，也不是独立的 Tile IR。
- `llvm.*` 不等于 LLVM IR 文本；`__tile_*` 是调用符号，不足以证明存在 `tile.*` 方言。
- Metal 生成器返回 MSL 字符串。`xcrun metal` / `new_library_with_source` 在主仓是注释；关联仓有运行时加载源码，中间 rustc 编译器后端不可见。
- 名为 `spirv` 的生成器输出 GLSL compute 文本，不是 SPIR-V 二进制。

后续机制研究必须能回答：某种分层是把「识别输入」从「写出目标实现」里拆开，还是只换了一种目标语言中转；以及这条中转对 Metal 是否真的接到了 MSL 工具链。

---

## 2. 四种组织方式

四种方式在外部编译器里都能找到实例。它们解决的问题不同，对 tile-rs 的覆盖范围也不同。

### 2.1 按硬件家族组织

**机制（一手事实）**

LLVM 把 GPU 相关代码生成放进独立 Target，与 CPU Target 并列。2026-09-15 的 `llvm/lib/Target` 目录含 `NVPTX`、`AMDGPU`、`SPIRV`、`DirectX`，没有 `Metal` 或 `AIR`。文档对应：

- NVPTX：消费 LLVM IR 子集，用 `ptx_kernel` 等约定表示 kernel，产出 PTX。[NVPTXUsage.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/llvm/docs/NVPTXUsage.md)
- SPIR-V Target：`llc -mtriple=spirv32/spirv64/...` 或 `clang --target=spirv64`，面向 OpenCL / Vulkan 运行时，不是 Metal。[SPIRVUsage.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/llvm/docs/SPIRVUsage.md)
- DirectX Target：架构为 `dxil`，OS 为 `shadermodel`，环境含 `compute`。[DirectXUsage.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/llvm/docs/DirectXUsage.md)

MLIR `gpu` 方言自称「类似 CUDA 或 OpenCL 的编程模型」，提供 `gpu.module` / `gpu.func` / `gpu.launch` 以及 thread/block/subgroup 原语。官方编译示例是 `nvvm-attach-target` → `convert-gpu-to-nvvm` → `gpu-to-llvm` → `gpu-module-to-binary`。文档写明该默认 NVVM 流水线「要求已经显式并行的 IR，自己不做 GPU 并行化」。[GPU.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/GPU.md)

`mlir/lib/Conversion` 在同一 SHA 下有 `GPUToNVVM`、`GPUToROCDL`、`GPUToSPIRV`、`GPUToLLVMSPV`，没有 `GPUToMetal`。

IREE 把 CUDA、ROCM、VulkanSPIRV、MetalSPIRV、WebGPUSPIRV 做成并列的 HAL target 插件。[compiler/plugins/target](https://github.com/iree-org/iree/tree/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/plugins/target) 目录列出这些名字。Metal 插件复用 SPIR-V codegen 流水线，再交叉编译，见第 3 节。Apple 目标细节里 `mmaCount = 0`，feature 字符串是 `spirv:v1.3,cap:Shader`。[KnownTargets.cpp](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/src/iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.cpp)（约 901–915、1430–1433 行）

**适用性推断**

「GPU 家族」能解释：NVPTX / AMDGPU / SPIR-V 为何共享一份 LLVM IR 或一份 `gpu` 方言，再分叉到不同序列化。它解释不了 Metal 作为 LLVM/MLIR 一等目标，因为公开树里没有这条分叉。

对照 tile-rs：公开生成器覆盖 CUDA、MUSA、GLSL、MSL，也覆盖 NKI、TPU/Pallas、PTO、AIE、BANG、Gaudi 等非 GPU 目标。按 GPU 家族切一层，最多覆盖其中一部分；NKI/TPU/PTO 的目标原语与工作分配不在 CUDA/OpenCL 模型里。MUSA 已经用整生成器委托贴在 CUDA 上，这是家族内局部共享，不是新的 GPU IR。

**不能复用的职责**

即便只讨论 GPU 子集，Metal 的入口属性、threadgroup 内存、barrier、simdgroup matrix、以及 host 侧 `dispatch_thread_groups` 契约仍要单独表达。IREE 自己把 Apple WGP 的 MMA 计数写成 0，并注明这些数值未经核对。tile-rs 的 MSL 生成器里存在 simdgroup 矩阵乘与 partition-cell 路径（#14），不能假定「GPU 家族 lowering」会自动发出这些原语。

### 2.2 按编译职责或能力组织

**机制（一手事实）**

MLIR 把「写程序」和「对某个环境合法」分开。`gpu` 方言描述 launch 与线程原语；`gpu-module-to-binary` 按挂在 `gpu.module` 上的 Target 属性做序列化；Offloading 属性再把 binary 和 `gpu.launch_func` 译成 LLVM 指令。[GPU.md Compilation](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/GPU.md)

SPIR-V 方言用 `spirv.target_env` 携带版本、扩展、capability 和资源限制。`SPIRVConversionTarget` 按该环境检查 op 合法性；缺扩展时 `SPIRVTypeConverter::convertType()` 返回空类型。[SPIR-V.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/SPIR-V.md)（Target environment / Conversions）

同一份文档把 shader ABI 写成编译路径的外部信息：descriptor set/binding、push constant 必须与 host 资源布局一致。SPIR-V 只表达 device 计算，不够运行。

IREE 再拆一层：`TargetBackend` 负责可执行文件翻译与序列化，HAL driver 负责 `MTLDevice` / `MTLCommandBuffer` / `MTLLibrary` 等运行时对象。Metal 的编译后端名叫 `metal-spirv`，设备名叫 `metal`。[MetalSPIRVTarget.cpp](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/plugins/target/MetalSPIRV/MetalSPIRVTarget.cpp)；运行时设计：[metal-hal-driver.md](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/docs/website/docs/developers/design-docs/metal-hal-driver.md)

tile-rs 公开缝已经按职责切开：`CodegenTarget::emit` 只返回源码，目标工具链调用故意不放进该 trait。[target.rs](https://github.com/Jopqior/tile-rs/blob/45a3cd148bba0c953a8b03071038ea46a9d08b93/crates/tile_codegen/src/target.rs)

Apple 把 MSL 编译分成在线（API 编译源字符串）和离线（先得到二进制再加载）。[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) Version 4.1（2026-06-04）§1.6。运行时入口：`MTLDevice.makeLibrary(source:options:)`，「同步地通过编译源字符串中的函数创建 Metal library」。[文档](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:))。离线入口见第 3 节 IREE 的 `xcrun metal | metallib`。

**适用性推断**

这一维直接对应 #14 里「由哪个阶段展开算法」以及「入口和工作分配契约」。SPIR-V `target_env` 和 Metal GPU family 是「这个目标允许哪些 op」的合法性表，和 tile-rs 各文件里的 `KernelType` / `classify_body` 不是同一件事：前者是目标能力，后者是对输入文本的模式识别。

tile-rs 已经把 emit 与工具链调用分开。再按职责分层，增量主要在 emit 内部：识别输入、选择实现、写出目标语言。外部机制能证明这种切开是常见做法，不能证明必须再引入 MLIR pass manager。

**不能复用的职责**

Vulkan 把 workgroup size 编码进 SPIR-V，Metal 在 `dispatchThreads:threadsPerThreadgroup:` 同时指定 grid 与 threadgroup。IREE 因此把 threadgroup size 从 SPIR-V 抽出来另存 FlatBuffer，dispatch 时再交给 Metal。[metal-hal-driver.md § Workgroup/threadgroup size](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/docs/website/docs/developers/design-docs/metal-hal-driver.md)。tile-rs 的 MSL 头注释用 `[[max_threads_per_threadgroup(N)]]` 对位 GLSL 的 `layout(local_size_x=N)`，host 例子则写死 `dispatch_thread_groups(1 组 × 256 线程)`。无论职责怎么切，这份契约要在生成签名和 host 绑定两侧同时改。

Metal 命令缓冲是 encoder 分层模型（blit/compute encoder），IREE HAL 是扁平 Vulkan 风格记录。IREE 为此做了分段延迟记录，并注明 fill/copy 对齐限制要用 polyfill kernel。这是运行时职责，不是 emitter 文本能消掉的。

### 2.3 增加共享 IR 或 MLIR lowering 阶段

**机制（一手事实）**

MLIR 的渐进 lowering：Arith / GPU 等方言 → SPIR-V 方言 → 序列化为 SPIR-V 二进制。当前文档列出的转到 SPIR-V 的转换是 Arith 与 GPU：`gpu.module` 变成 `spirv.module`，其中的 `gpu.func` 降为入口函数。[SPIR-V.md Current conversions](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/SPIR-V.md)；pass 声明要求 `gpu.func` 带 `spirv.entry_point_abi`。[GPUToSPIRVPass.h](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/include/mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h)

LLVM 方言把 LLVM IR 映射进 MLIR，`llvm.func` 是一等操作，指针语法为 `!llvm.ptr` 及可选地址空间 `!llvm.ptr<N>`。[LLVM.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/LLVM.md)。这与 tile-rs 测试里的 `!llvm.ptr<1>` 写法一致，只说明类型记号同类，不说明整模块可被 `mlir-opt` 验证。

tile-rs 已有一条 **egress** 式 linalg 文本生成：把 `__tile_*` 写成 `linalg.softmax` / `linalg.matmul` 等 named operation，生成器自己不展开线程与同步（#14 Softmax / matmul 对照）。文件头称其面向「linalg-on-tensors」。[mlir_to_linalg.rs](https://github.com/Jopqior/tile-rs/blob/45a3cd148bba0c953a8b03071038ea46a9d08b93/crates/rustc_codegen_tile/src/mlir_to_linalg.rs)。PTO 是另一条可输出的 MLIR 方言文本，领域地图写明它不是各目标必经阶段。

Khronos 对 SPIR-V 的定位：供 Vulkan、OpenGL、OpenCL 驱动消费的中间形式。[Khronos SPIR](https://www.khronos.org/spir/)（2026-09-15 抓取）。Metal 不在该客户 API 列表里。

**适用性推断**

共享 IR 能解释「识别与数据依赖」和「目标实现」如何分开：在 GPU/SPIR-V 方言里，softmax 可以仍是 `gpu`/`linalg` 操作，由后续 pass 决定展开成 warp 归约还是 subgroup op。这正是 #14 对 linalg 生成器的观察。

对当前 tile-rs 输入，这一步的前置成本是 **raising**：公开生成器看到的是 `llvm.call @__tile_add_f32` 这类符号，不是 `gpu.func` 或 `linalg.add`。`parse_module` 明确是 std-only 文本扫描，禁止 MLIR/LLVM 依赖。[mlir_parse.rs](https://github.com/Jopqior/tile-rs/blob/45a3cd148bba0c953a8b03071038ea46a9d08b93/crates/rustc_codegen_tile/src/mlir_parse.rs)。接入 `convert-gpu-to-spirv` 之前，至少要先得到带 `spirv.entry_point_abi` 的 `gpu.func`，并处理 `__tile_*` 的语义。本次未尝试用 `mlir-opt` 解析测试片段。

linalg 生成器已经演示「保留 named operation、把展开留给后续工具」。它是单独目标，不是 CUDA/MSL/NKI 的公共前置。把它改成公共层，等于改变所有目标的输入契约，也把线程/同步职责从各 emitter 挪走；对 Metal 仍要有人写出 threadgroup 归约或调用后续 GPU 流水线。

**不能复用的职责**

- 从 `__tile_*` 文本到可验证 MLIR 的 raising、合法性和地址空间模型。
- 非 GPU 目标（PTO/NKI/TPU）并不消费 `gpu` 或 SPIR-V 方言。
- Metal 若停在 SPIR-V 方言，仍然没有官方序列化到 MSL；必须再接第 3 节的交叉编译。
- 发布 rustc 编译器后端内部的 MIR→MLIR 不可见。若上游开始产出真正的 `gpu`/`linalg` 模块，公开生成器的输入假设会变；本票不能把那种未来输入当成现状。

### 2.4 直接 emit 与局部共享

**机制（一手事实）**

tile-rs 现状就是这一类：每个目标一个 `convert_mlir_to_*`，共享文本 parser；MUSA 对 CUDA 做头文件与标识符替换。[mlir_to_musa.rs](https://github.com/Jopqior/tile-rs/blob/45a3cd148bba0c953a8b03071038ea46a9d08b93/crates/rustc_codegen_tile/src/mlir_to_musa.rs)

生态里对等的「局部转换器」：

- SPIRV-Cross：解析 SPIR-V，编译到 GLSL / MSL / HLSL。MSL 是一级后端，不是实验项。[README](https://github.com/KhronosGroup/SPIRV-Cross/blob/be71ee8c12cd7dc5ca8fa9581f708c2e8561fe2a/README.md)。`CompilerMSL` 继承 `CompilerGLSL`，选项含 platform（iOS/macOS）、`msl_version`（默认 1.2）、`argument_buffers`、`emulate_subgroups`、`ios_use_simdgroup_functions`、`fixed_subgroup_size`。[spirv_msl.hpp](https://github.com/KhronosGroup/SPIRV-Cross/blob/be71ee8c12cd7dc5ca8fa9581f708c2e8561fe2a/spirv_msl.hpp)
- naga（wgpu 树内）：前端 SPIR-V / WGSL / GLSL，后端含 `msl-out`（主支持）。绑定模型是 Metal 扁平 per-resource，需要单独的 descriptor set 映射。明确不支持 `SHADER_INT64_ATOMIC_ALL_OPS`。[naga/README.md](https://github.com/gfx-rs/wgpu/blob/044e92665050a08110cf1da3809f543719bd1ad0/naga/README.md)；[back/msl/mod.rs](https://github.com/gfx-rs/wgpu/blob/044e92665050a08110cf1da3809f543719bd1ad0/naga/src/back/msl/mod.rs)
- MoltenVK：Vulkan 1.4 在 Metal 上的分层实现，运行时把 SPIR-V 转成 MSL。另有独立命令行 `MoltenVKShaderConverter`。[README](https://github.com/KhronosGroup/MoltenVK/blob/4aaf714aa1b3e78e26ecfcefa9c75e9a576c500b/README.md)。这是 Vulkan 运行时，不是 tile-rs 生成器输出 MSL 的缝。

**适用性推断**

局部共享能解释 MUSA，也能解释「先有一份 SPIR-V，再分叉出 MSL/HLSL/GLSL」。它不自动共享 #14 里的 SiLU 识别或 matvec 链分析：那些发生在生成目标文本之前。SPIRV-Cross/naga 接的是已经合法的着色器 IR，不是 `__tile_*` 文本。

MSL 头注释声称复用 SPIR-V 分类，#14 已核对两份 `KernelType` 与 `classify_body` 各自定义。这是「注释当共享」失败的实例，属于直接 emit 下局部复制的维护成本。

**不能复用的职责**

SPIRV-Cross 文档写明：跨语言时并非一方的特性都能被目标原生支持，需要调用方做 binding 重映射、entry point 名称、clip-space 等处理。MSL 2.0 argument buffer 下，descriptor set 变成 argument buffer，数组资源会占用多个 id，Vulkan 不会。[README Implementation notes](https://github.com/KhronosGroup/SPIRV-Cross/blob/be71ee8c12cd7dc5ca8fa9581f708c2e8561fe2a/README.md)

IREE 在序列化里写 TODO：因为 spirv-cross，当前是每个函数一个 module，无法像 Vulkan SPIR-V 目标那样链接多个 `spirv::ModuleOp`，「really bad」。[MetalSPIRVTarget.cpp](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/plugins/target/MetalSPIRV/MetalSPIRVTarget.cpp)（`serializeExecutable` 前部注释）

naga 要求调用方提供 binding map；MSL 入口参数与 IR 侧不同，它在入口处重组。这些是转换器接口的一部分，不会替 tile-rs 决定 softmax 用 `simd_sum` 还是循环树。

---

## 3. Metal 通路核对

只记录能从输入语义接到 MSL 工具链或 `MTLLibrary` 的路径。存在 `gpu` 或 SPIR-V **方言**本身不构成 Metal 通路。

| 路径 | 输入语义 | 目标源码 / 工具链 | 限制 | 判定 |
|------|----------|-------------------|------|------|
| A. 直接写 MSL | MSL 源字符串（kernel、buffer、threadgroup 属性） | 在线：`makeLibrary(source:)`；离线：`xcrun metal -c` → AIR → `metallib` | 公共接口是 MSL，不是 LLVM IR / SPIR-V | **真实**。tile-rs 生成器的输出形态与此对齐；主仓未执行工具链 |
| B. SPIR-V 二进制 → SPIRV-Cross → MSL → A | SPIR-V 模块，compute 入口，descriptor set/binding | `CompilerMSL::compile()` 得 MSL，再走 A | 要 SPIR-V 二进制而非 GLSL；binding/workgroup/entry 名需调用方配置；subgroup 在部分设备上要模拟 | **真实**。IREE `metal-spirv` 与 MoltenVK 走这条 |
| C. MLIR GPU → SPIR-V 方言 → 序列化 → B | `gpu.module`/`gpu.func` + `spirv.entry_point_abi` | `convert-gpu-to-spirv` + `spirv::serialize` | 当前公开输入不是这种形态 | **到 SPIR-V 为真实**；接 Metal 必须再经 B。不能停在 GPU 方言 |
| D. MLIR GPU → NVVM / ROCDL | 同 C 的 GPU 方言 | `convert-gpu-to-nvvm` / `GPUToROCDL`，序列化为 cubin/AMDGPU | 官方流水线示例即此 | **不是 Metal** |
| E. LLVM IR → NVPTX / AMDGPU / SPIR-V / DXIL | LLVM IR + 对应 triple | `llc` | `llvm/lib/Target` 无 Metal | **不是 Metal** |
| F. SPIR-V 或 WGSL → naga → MSL → A | naga IR（由 SPIR-V/WGSL 前端来） | `msl-out` | 自备 binding map；部分原子能力缺失；AIR 后端栏为空 | **真实转换器**，IR 与 SPIRV-Cross 不同 |
| G. Vulkan + SPIR-V → MoltenVK | Vulkan API 与 SPIR-V 着色器 | 运行时转 MSL 并调 Metal | 应用改走 Vulkan，不产出给 tile-rs host 用的 `.metal` | **真实运行时**，不是 emitter 分层候选 |
| H. 上游 MLIR Metal 方言 | — | — | Dialects 目录无 Metal.md | **未找到**。不能从 GPU 方言存在推出 |

### 3.1 路径 A：MSL 是 Metal 的公共编译输入

Apple 规范：Metal compiler 可在线或离线使用；离线产物以二进制加载。§1.6 列出的是面向 Metal 源的预处理、数学与优化选项（`-std` 语言版本、`-fmetal-math-mode` 等），不是 LLVM triple。SIMD-group matrix 类型在 Metal 2.3+，头文件 `<metal_simdgroup_matrix>`；规范在 4.1 里建议新代码考虑 Tensor API。Spec Version 4.1，版权页日期 2026-06-04。

`MTLLibrary` 是「Metal shader functions 的集合」。[MTLLibrary](https://developer.apple.com/documentation/metal/mtllibrary)。源字符串编译入口见上表。

IREE 把离线命令写成：

```text
xcrun -sdk macosx metal -c <file.metal> -o - | xcrun -sdk macosx metallib - -o <file.metallib>
```

iOS / simulator 换 sdk 名。实现还要求 **编译主机是 macOS**，否则跳过 metallib、只嵌入 MSL。[MSLToMetalLib.cpp](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/plugins/target/MetalSPIRV/MSLToMetalLib.cpp)；[MetalSPIRVTarget.cpp](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/plugins/target/MetalSPIRV/MetalSPIRVTarget.cpp) 中 `hostTriple.isMacOSX()` 判断。

tile-rs 生成器文件头描述同一对工具（`xcrun metal`、运行时 `new_library_with_source`）。领域地图将其标为注释，不是本树调用。关联仓 example 使用 `metal` crate 的 `new_library_with_file` / `new_library_with_source`。**适用性推断**：路径 A 是现状输出已经对准的工具链，不依赖新 IR。

### 3.2 路径 B：SPIR-V → MSL 是核实过的间接通路

IREE 设计文档写明：Metal 没有开放中间语言，因此复用 SPIR-V 代码生成，再用 SPIRV-Cross 交叉编译到 MSL；MoltenVK 也如此，文档称这条路径在图形领域常见。[metal-hal-driver.md § Shader/kernel compilation](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/docs/website/docs/developers/design-docs/metal-hal-driver.md)

实现步骤（同一 SHA 的 `serializeExecutable`）：

1. `spirv::serialize` 得到 SPIR-V 二进制。
2. 对每个 `spirv.EntryPoint` 调 `crossCompileSPIRVToMSL`。内部是 `CompilerMSL`，`ExecutionModelGLCompute`，MSL 3.0，打开 argument buffers，按 descriptor set/binding 填 `MSLResourceBinding`，push constant 固定到 `[[buffer(3)]]`（与 HAL 约定一致）。entry 名可能被改掉以避免 MSL 保留字。workgroup size 必须非零，否则失败。[SPIRVToMSL.cpp](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/compiler/plugins/target/MetalSPIRV/SPIRVToMSL.cpp)
3. 可选 `compileMSLToMetalLib`。
4. 打进 FlatBuffer：MSL 源和/或 metallib、threadgroup size、binding flags。

SPIRV-Cross 源码会 `#include <metal_simdgroup_matrix>`（`spirv_msl.cpp` 约 1953 行），并实现 `OpGroupNonUniform*` 到 SIMD-group/quadgroup。这只证明转换器认识这类 SPIR-V 指令。IREE Apple 目标 `mmaCount = 0`，因此 **未核实** IREE 或 SPIRV-Cross 会为 tile-rs 这类 gemm 发出 `simdgroup_matrix` 乘加。

对 tile-rs 的关键错位：名为 `spirv` 的生成器产出 GLSL `.comp` 文本，声明再经 `glslangValidator`/`glslc` 得 SPIR-V。SPIRV-Cross 的输入是 SPIR-V 二进制。中间还缺一次 glslang，且生成的 GLSL 是否满足 Vulkan 语义、能否通过 SPIRV-Cross 的 compute 路径，**未核实**。不能把「已有 spirv 生成器」读成「已有 Metal 间接通路」。

### 3.3 路径 C–E：GPU/LLVM 家族 lowering 到不了 Metal

`convert-gpu-to-spirv` 是真实的 GPU→SPIR-V。之后只有再接 B 才到 Metal。`convert-gpu-to-nvvm` 与 `GPUToROCDL` 的官方用途是 NVIDIA/AMD。LLVM Target 列表在 2026-09-15 没有 Metal。naga 的 AIR 后端在支持表里为空。

**适用性推断**：把 tile-rs 输入升到 `gpu` 方言，可以得到 CUDA/ROCm/Vulkan 共享层，仍然要为 Metal 外挂 B 或继续手写 MSL。

### 3.4 路径 F–G：库与运行时替代

naga 可作为 SPIRV-Cross 的库替代：同样输出 MSL 源，再走路径 A。它是 Rust 库，接口是自己的 IR 与 binding map，不是 MLIR pass。验证 MSL 的 xtask 需要 Xcode 命令行工具。本次未跑 naga。

MoltenVK 让 **Vulkan 程序** 在 Apple 上运行。若 tile-rs 的 host 改走 Vulkan/wgpu，理论上可用 GLSL→SPIR-V→MoltenVK，不再自己 emit MSL。这改变的是目标平台（Vulkan 在 Metal 上），不是 MSL 生成器分层。关联仓 USAGE 曾写 wgpu/Metal，源码依赖的是 `metal` crate，领域核对已记录该不一致。

IREE 网站的 GPU-Metal 部署指南正文是 “Documentation coming soon!”（[gpu-metal.md](https://github.com/iree-org/iree/blob/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf/docs/website/docs/guides/deployment-configurations/gpu-metal.md)）。Metal 路径的一手证据以 design doc 与 `MetalSPIRV` 插件源码为准，不以该指南为准。

---

## 4. 对照前置职责：机制能接住什么

| #14 里的问题 | 硬件家族 | 编译职责/能力 | 共享 IR / MLIR lowering | 直接 emit + 局部共享 |
|---|---|---|---|---|
| 文本解析已共享 | 不涉及 | 已有 `emit` 缝 | 若改用 MLIR API，parser 会被替换而非扩展 | 现状 |
| SiLU/链分析各自一份 | 家族层不管输入识别 | 可把「识别」标成独立职责，仍要写实现 | 升到 SSA/方言后，use-def 由 IR 提供，融合策略仍可按目标分叉 | 抽公共助手可减复制，不强迫相同融合 |
| Metal 分类与输出联动 | 无 Metal 分叉可卸这层联动 | 可把分类从 emit 正文拆出；调度顺序仍是 Metal 实现细节 | 用 conversion legality 代替 `KernelType` 后，专用正文要么变成 pattern，要么仍留在 target | 拆文件或共享分类注释已经失败过一次 |
| softmax 数学同、实现不同 | GPU 家族可共享「未展开的 reduce」；NKI/TPU/linalg 不在该家族 | 「谁展开」正是职责切分 | linalg.softmax 已演示不展开；Metal 展开仍要写 threadgroup | 各写各的，现状 |
| CUDA matmul 仍是 TODO 注释 | 共享 GPU IR 不会自动补实现 | 能力表能记录缺口，不能生成 cublas 调用 | 降到 `linalg.matmul` 把实现推给后续工具 | 局部共享会连缺口一起复制（MUSA 已继承 CUDA） |
| 入口/buffer/dispatch 契约 | 各家族 ABI 不同 | SPIR-V ABI 属性、Metal buffer index 都是这一层 | 共享 IR 通常停在 device 计算，host 绑定另做 | 直接写在生成文本与 host 里 |
| 非 GPU 目标 | 解释不了 | 可按目标各做 legality | `gpu`/SPIR-V 复用不上 | 现状：各文件一份 |

**适用性推断**：若目标是减少「识别输入约定」的同步修改，共享 IR 或共享分析助手都可能碰到这个问题；前者换来可验证的 use-def，代价是 raising 与新依赖。若目标是减少 Metal 专用正文，SPIR-V 交叉编译能生成一份可编译 MSL，但 IREE/SPIRV-Cross 不会自动复现 tile-rs 当前的单 workgroup 融合循环、partition-cell 或 simdgroup gemm 选择。那些属于实现选择，#14 已说明不全是硬件必需，也没有被外部通路验证为可替换。

---

## 5. 引入成本（外部机制自身写出的）

来自 IREE / SPIRV-Cross / naga / Apple 文档的成本，不是工时估计。

- **依赖**：SPIRV-Cross（C++，IREE 用 `third_party/spirv_cross`，异常改断言）；或 naga（Rust）。MLIR GPU/SPIR-V 还要完整 MLIR 库、pass 与序列化。与当前 std-only 的 `mlir_parse` / `CodegenTarget` 测试模型冲突。
- **主机限制**：IREE 离线 metallib 需要 macOS 上的 `xcrun`。Linux CI 只能停在 MSL 源。Apple 在线 API 同样要在有 Metal 框架的机器上。
- **ABI 胶水**：descriptor set → `[[buffer]]` / argument buffer；push constant 槽位；entry 重命名；threadgroup size 从着色器抽到 host。IREE 为这些写了专门代码，并与 HAL 常量互锁。
- **能力子集**：IREE Apple 目标按 SPIR-V 1.3 Shader capability 配置，MMA 计数为 0。SPIRV-Cross 在 iOS 上可改用 quadgroup 或 `emulate_subgroups`。naga 缺 64 位原子全集。交叉编译结果是「Vulkan 计算模型能表达、且转换器认识」的子集，不是 MSL 语言全集。
- **链接粒度**：IREE 记录 spirv-cross 导致一函数一 metallib，弱于其 Vulkan 目标。
- **数值与数学模式**：MSL 默认 fast math 允许无 NaN/Inf、重关联、收缩。Spec §1.6.3。交叉编译或手写 MSL 若未设 `-fmetal-math-mode`，与 CUDA/CPU 参考的误差模型不必相同。本次未测。
- **非复用**：NKI/TPU/PTO/AIE 的原语、tile-rs 量化/GGUF 专用生成函数、以及发布后端内部 MIR→MLIR，都不在上述 Metal 通路里。

---

## 6. 未核实缺口

1. 未用 `mlir-opt` 解析 tile-rs 测试里的 `llvm.func` + `hacc.entry` + 函数体内 `^bb0` 片段，不知道上游 LLVM 方言解析器是否接受。
2. 未把 `mlir_to_spirv` 的 GLSL 经 glslang 再送入 SPIRV-Cross/naga，不知道能否得到可 `metal` 编译的 compute kernel。
3. 未比较手写 MSL（含 simdgroup / 单 workgroup 循环）与 SPIRV-Cross 输出在正确性或性能上的差异。
4. 未核对 SPIRV-Cross 的 `metal_simdgroup_matrix` 路径需要哪些 SPIR-V 扩展，以及是否覆盖 tile-rs 的 gemm/partition 形状。
5. 发布 `librustc_codegen_tile` 实际产出的 MLIR 文本是否仍是 `__tile_*` 调用，不可见。
6. Apple 开发者网站「Building a Library With Metal's Command-Line Tools」页面需 JS，本次未能抓到正文；离线命令以 IREE 源码与 MSL spec §1.6 为准。
7. SPIRV-Cross GitHub「最新 release」标签仍是 `MoltenVK-1.1.5`（2021-08-30）；开发在 `main`。引用以 `main` SHA 为准，不要用那个 release 当当前功能集。
8. IREE 部署指南 GPU-Metal 页尚未成文。Metal 3 期望（macOS Ventura / iOS 16）来自 2023 年左右的 design doc，是否仍是运行时下限，未对照当前 IREE HAL 源码逐项复核。

这些缺口若要影响 #16 的架构判断，需要专项补查或原型；本票不把它们补成肯定句。

---

## 7. 对后续决策的事实输入

本票不选择是否分层。下面是机制研究能稳定提供给 #16 的句子。

1. Metal 的公共编译入口是 MSL 源码（在线 API 或 `metal`/`metallib`）。LLVM 与上游 MLIR 在 2026-09-15 没有 Metal Target 或 Metal 方言。
2. 已核实的间接 Metal 通路是：合法 SPIR-V 二进制 → SPIRV-Cross 或 naga → MSL → 路径 A。IREE 的 `metal-spirv` 是这条路的完整编译器实例。MLIR `gpu` 方言只保证能降到 NVVM/ROCDL/SPIR-V。
3. 按硬件家族组织能说明 CUDA/ROCm/Vulkan 为何共享 GPU 编程模型，说明不了 Metal ISA，也覆盖不了 NKI/TPU/PTO。tile-rs 的 MUSA 已经是家族内局部共享。
4. 按编译职责组织与现状的 `CodegenTarget::emit` 一致。SPIR-V `target_env` 和 Metal 的 buffer/threadgroup 契约属于「能力与 ABI」，和各文件里的 `KernelType` 模式识别不是同一层。
5. 共享 IR 要先改变或 raising 当前 `__tile_*` 文本输入。linalg 生成器已是「不展开算法」的 egress 样本，不是其它目标的前置阶段。
6. 直接 emit 加局部共享是现状；SPIRV-Cross/naga 是「已有着色器 IR 之后」的局部转换，接不上现在的文本识别职责。
7. 交叉编译接上 Metal 之后，仍要单独维护 binding、workgroup size、entry 名，以及手写 MSL 里那些未被 SPIR-V 模型表达的实现选择。

地图「尚未指定」里关于有区分度的 kernel/对照/原型，仍取决于 #16 是否把某一条路径当成假设。本报告能提供的区分线索（尚未做成原型票）：

- 同一份 `__tile_softmax` 输入：手写 MSL 归约 vs SPIRV-Cross 从 GLSL/SPIR-V 来的 MSL，能否通过 `xcrun metal`，host 绑定是否还成立。
- 同一份 matvec/gemm 链：`simdgroup_matrix` 是否能从 SPIR-V 交叉编译出现，还是只能留在直接 emit。
- 不经 GPU 方言、只把 SiLU 识别抽成助手，能否在不引入 MLIR 的情况下打断 #14 的同步修改，同时允许各目标保留不同融合策略。

---

## 外部版本（2026-09-15 抓取）

| 对象 | 版本 / SHA | 日期 | 链接 |
|------|------------|------|------|
| `llvm/llvm-project` `main` | `f0907c3d807f08a34f45e106c3f28b034b8e8c8e` | 2026-09-15 | [commit](https://github.com/llvm/llvm-project/commit/f0907c3d807f08a34f45e106c3f28b034b8e8c8e) |
| MLIR GPU 方言文档 | 同上树 `mlir/docs/Dialects/GPU.md` | 同上 | [GPU.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/GPU.md) |
| MLIR SPIR-V 方言文档 | 同上树 `mlir/docs/Dialects/SPIR-V.md` | 同上 | [SPIR-V.md](https://github.com/llvm/llvm-project/blob/f0907c3d807f08a34f45e106c3f28b034b8e8c8e/mlir/docs/Dialects/SPIR-V.md) |
| `iree-org/iree` `main` | `5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf` | 2026-09-15 | [commit](https://github.com/iree-org/iree/commit/5bd1d9629ee60284fdc56ae61b0c5bc712c78ccf) |
| `KhronosGroup/SPIRV-Cross` `main` | `be71ee8c12cd7dc5ca8fa9581f708c2e8561fe2a` | 2026-09-07 | [commit](https://github.com/KhronosGroup/SPIRV-Cross/commit/be71ee8c12cd7dc5ca8fa9581f708c2e8561fe2a) |
| `gfx-rs/wgpu` `trunk`（含 naga） | `044e92665050a08110cf1da3809f543719bd1ad0` | 2026-09-15 | [commit](https://github.com/gfx-rs/wgpu/commit/044e92665050a08110cf1da3809f543719bd1ad0) |
| `KhronosGroup/MoltenVK` `main` | `4aaf714aa1b3e78e26ecfcefa9c75e9a576c500b` | 2026-09-05 | [commit](https://github.com/KhronosGroup/MoltenVK/commit/4aaf714aa1b3e78e26ecfcefa9c75e9a576c500b) |
| Metal Shading Language Specification | Version 4.1 | PDF 元数据 2026-06-04 | [PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) |
| Apple `MTLDevice.makeLibrary(source:options:)` | 开发者文档 JSON，2026-09-15 | — | [文档](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)) |
| Khronos SPIR 概述 | 站点页，2026-09-15 | — | [khronos.org/spir](https://www.khronos.org/spir/) |
| `Jopqior/tile-rs` 对照 | `45a3cd148bba0c953a8b03071038ea46a9d08b93` | 职责调查提交 | [commit](https://github.com/Jopqior/tile-rs/commit/45a3cd148bba0c953a8b03071038ea46a9d08b93) |
