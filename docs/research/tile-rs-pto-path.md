# tile-rs 的 PTO / Ascend 实际路径核查

调查日期：2026-09-22。对应 [核实 tile-rs 仓库中 PTO / Ascend 路径的实际调用链](https://github.com/Jopqior/tile-rs/issues/56)，属于 [PTO / Ascend CI 资源依赖调查](https://github.com/Jopqior/tile-rs/issues/49)。

源码固定为 `e8d1acdf58f4e2239bf6be9d59333d7a337282b3`（提交日期 2026-09-18）。下列本仓库引用全部固定到此提交，不用当前 upstream 分支替代。该提交没有 `CONTEXT.md` 或 `docs/adr/`。本次只审阅源码、保留的产物/记录、GitHub 第一方 API 和发布包中的文本；**没有执行安装器、后端动态库、PTOAS、CANN 编译器或 NPU 测试，没有复现任何性能或正确性结果**。下载的发布包只列目录、计算摘要、读取配置与说明。

## 结论

1. 仓库确有从 Rust kernel 选择 `cpp` 或 `pto` 的入口，且留有真实的 `.pto` 文本、调用 PTO Tile Library 的 C++ 和 PTO 标名的测量 CSV。不能把 PTO 起点当成尚不存在的纯假设。[S1][S2][S10][S11][S12]
2. 但**公开树不是完整编译器源码**：MIR→MLIR 驱动、AscendC/PTO emitter 及其编译调度在不透明发布后端内。公开 `tile_codegen` 只注册 debug 和可选 Metal；不能从 registry 接口推演 PTO 的 pass 顺序、自动 PTOAS 调用、fallback 或完整环境变量合同。[S1][S3][S4]
3. 两条主要路径应分别记为：
   - Rust → 后端的直接 AscendC 路径（`cpp`）→ CANN 目标编译/打包 → Rust ACL host；**不要求经过 PTOAS**。
   - Rust → 后端的 PTO 路径（`pto`）→ `.pto` → 外部 `ptoas` → 调用 `pto/pto-inst.hpp` 的 C++ → CANN 目标编译、launch wrapper → ACL host。自动集成中间段不可审计；另有公开手工 C++ host recipe 可核对。[S2][S5][S6][S10][S13]
4. “有预编译后端”“有 C++ 输出”“编译链接成功”“库加载成功”“设备运行正确”是不同证据。默认发布标签当前只有 macOS ARM 包；Linux 包的 workflow 声明不是已发布资产证明。PTOAS 0.55/CANN 9 的 matmul 记录也不能替代更早 softmax 或直接 AscendC 路径的记录。

## 1. 从哪个 Rust 入口开始

`examples/tile_matmul/kernels/src/lib.rs` 是 `no_core` kernel crate，定义五种 f32 matmul：16×16×16、32³、64³、16×32×16、32×64×32。`#[tile_std::tile_kernel]` 接收 `GmView/GmViewMut`，宏把边界改为裸指针、插入 block/view prelude，附加 `#[unsafe(no_mangle)]` 和 `#[tile::kernel]`；kernel 调用 tile load、safe matmul、tile store。softmax 示例同样从 Rust 入口开始，当前源码有 `tile_softmax`、`tile_softmax_safe`、`tile_softmax_view` 三个 1×1024 入口。[S7][S8][S9]

host 示例 `build.rs` 调用 `add_ascend_link_args()`，然后 `KernelBuilder::new("kernels").copy_to(OUT_DIR/"kernel.o").build()`。**`kernel.o` 只是拷贝目的名，不保证内容是裸设备 object，可能是 host shared library**。Builder 输入目录走 Rust，输入文件走 C++；`.codegen_path("pto")` 可只覆盖子进程，但上述两个示例没有显式调用它，而是继承环境。softmax 的“uses PTO”注释本身不选后端。[S2][S5][S14][S36][S37]

### 公开源码能追到的调用边界

```text
Rust kernel + 宏 / tile_std                         公开源码
  ↓ build.rs → KernelBuilder → rs_builder
cargo build --release --target=<davinci-huawei-none.json>
  -Zcodegen-backend=<动态库>                         公开启动合同
  ↓ librustc_codegen_tile
MIR → MLIR → cpp 或 pto emitter + 编译调度            不透明发布实现
  ├─ 直接 AscendC C++                              文档/调用合同
  └─ PTO IR → PTOAS → PTO-library C++               留存产物 + 手工 recipe
  ↓ 后端可能提供 .tile.so / .tile.gen.cpp / .tile.o
rs_builder：优先 .tile.so；否则 C++ 经 CANN CMake；否则 .tile.o
  ↓ 拷贝成 OUT_DIR/kernel.o
Rust host：ACL 初始化 → dlopen + <name>_do 或设备 object 注册 → launch
```

`rs_builder` 在 Cargo JSON 的 `compiler-artifact` 文件所在目录扫描上述后缀，而不是验证产物 provenance，也不把 `.pto` 单独作为 build 返回值；它没有公开调用 `ptoas` 的代码。`.tile.so` 的 dual AIC+AIV 说明是 builder 的消费合同，不证明发布后端在所有 PTO kernel 上必然走该分支。[S5]

## 2. 入口、选项和命令合同

以下命令是**静态还原/原文 recipe**，不是本次执行成功的操作手册。SDK、后端包、PTOAS 路径及目标必须先匹配；公开树和包不足以保证这些命令现在可直接完成全链路。

### 2.1 Rust 构建入口

```sh
# host 示例入口，选择 pto；cpp 是另一条路径
TILERS_CODEGEN_SO=/absolute/path/librustc_codegen_tile.so \
TILERS_CODEGEN_PATH=pto ACLRS_CANN_PATH=/path/to/cann \
ACLRS_SOC_VERSION=<explicit-soc> ACLRS_RUN_MODE=npu \
cargo +nightly-2025-08-04 build --release \
  --manifest-path examples/tile_matmul/Cargo.toml
```

其 kernel 子进程固定使用 `cargo build --release --message-format=json-render-diagnostics --target=<绝对 JSON 路径>`，通过 `CARGO_ENCODED_RUSTFLAGS` 传入 `-Zcodegen-backend=...`、两个 `register_tool(tile)` 相关参数、`-Cpanic=abort -Clto=off`。优先从 kernel 目录向上寻找 `davinci-huawei-none.json`，否则尝试 `RUST_TARGET_PATH`，最后用 target 名。根 `.cargo/config.toml` 只有 `incremental=false`，不自动选择 backend；根 workspace 排除 examples、builder、compiletest，也没有完整后端 crate。不能把根 `cargo build` 等同于 Rust→PTO 检查。[S5][S15][S38]

| 变量/入口 | 可见的精确作用与边界 |
|---|---|
| `TILERS_CODEGEN_SO` | Builder 找动态库的首选，兼容 `ASCEND_RUSTC_CODEGEN_TILE_SO`；未找到有效文件则扫描平台动态库搜索路径。普通 Cargo 并不原生解释此变量，必须由 builder 或显式 `-Zcodegen-backend` 接线。[S5] |
| `TILERS_CODEGEN_PATH=cpp/pto` | README 的后端选择合同；builder 注释写缺省 cpp，override 只影响 kernel cargo 子进程。缺省及完整选择逻辑在后端内，无法源码复核。[S1][S2] |
| `ACLRS_PTOAS_PATH` | benchmark 脚本查 `/data/czq/ptoas-bin/ptoas`、`/data/linyifan/ptoas-bin/ptoas` 后设置，也可继承已有值；未找到则 SKIP。**消费它的后端代码不公开，不能断言完整搜索优先级**。[S16] |
| `ACLRS_CANN_PATH` | `Settings` 的实际 SDK override，然后 `$HOME/Ascend/ascend-toolkit/latest`，再 `/usr/local/Ascend/ascend-toolkit/latest`。报错却提 `ASCEND_HOME_PATH`；脚本 source SDK 不等于 Settings 读取该变量。[S6] |
| `ACLRS_SOC_VERSION` / `ACLRS_RUN_MODE` | Settings 缺省 `Ascend310P1` / `npu`，后者只接受 npu 或 sim。传入 CMake 的 `SOC_VERSION/RUN_MODE`；sim 增加 simulator 库路径及 `runtime_camodel` 链接。不是 PTO CPU-SIM。[S6][S17] |
| `ASCEND_DEVICE_ID` | Rust `Device::new` 的设备号，缺省 0；C++ `pto_run` 固定 0。SDK/目标选择与执行设备选择不同。[S18][S13] |
| `ASCEND_SOFTMAX_KERNEL` | Rust softmax host 选择 `<name>_do` 符号，缺省 `tile_softmax`；不是后端选择器。[S19] |

没有在公开路径找到可核实的 Rust→PTO **emit-only/skip-target-compile** 开关。不能因为 emitter 理论上是字符串转换，就声称完整 `KernelBuilder` 不需要 CANN。切换环境后的缓存也需独立核实：`rerun-if-env-changed` 不能作为后端内部缓存键的证明；扫描输出目录又可能遇到残留产物。[S5]

### 2.2 PTOAS 与手工目标编译旁路

`tools/ascend/README.md` 的 2026-08-05 recipe：[S20]

```sh
ptoas --enable-insert-sync X.pto -o X.cpp
CANN=/usr/local/Ascend/cann-9.0.0/aarch64-linux
$CANN/ccec_compiler/bin/ccec -O2 -std=c++17 -DMEMORY_BASE \
  --cce-aicore-arch=dav-c220-cube -x cce pto_run.cpp -o pto_run \
  -I$CANN/include -I. -L$CANN/lib64 -lascendcl -lruntime -lstdc++
LD_LIBRARY_PATH=$CANN/lib64 ./pto_run 20
```

此处须在包含 `pto_run.cpp` 和生成头的工作目录运行。`PTO_KERNEL_HEADER` 缺省 `"matmul.cpp"`，可 `-DPTO_KERNEL_HEADER=\"X.cpp\"` 覆盖；`PTO_M/K/N` 缺省 16，可用 `-D` 覆盖，但函数调用仍硬编码 `tile_matmul(a,b,c)`，不是任意 kernel 的通用 launcher。当前 Rust 示例导出的是带形状后缀的名字，不能不适配就接上。README 列出的 `matmul.pto / matmul.cpp / attention.pto` **没有被此提交跟踪**；该目录实际只有 README 和两个 launcher。[S7][S13][S20]

recipe 解释 PTOAS 给出的是 `AICORE void name`，需包为 `extern "C" __global__ AICORE void pto_entry`，才能 launch；matmul 使用 cube arch，不能换 vec。C++ host/device 放在同一 `-x cce` translation unit，输出可执行程序，并不经过 Rust `KernelLoader`。但保留的 softmax `.pto.cpp` 已经有 `extern "C" __global__ AICORE`，且有额外 auto-sync tail helper；**它不能被当成未经后处理的同版本 PTOAS 原始输出**，也不能再机械套一层同名 global wrapper。[S10][S13][S20]

外部工具本次只复核 PTOAS 固定快照 `97e9ecb247744c587171c73ab58ac2423bba3197` 的职责/compile-only 文档。它解析验证 `.pto`、做 pass、lower 到 EmitC/Linalg 并输出调用 `pto-isa` 的 C++，不是直接生成可装载 NPU 二进制。该快照基于 LLVM19 VPTO；tile-rs 后端包带 LLVM/MLIR20，这是**两个工具的依赖**，不能合并成一个版本要求。该 PTOAS 快照也不是 recipe 中“0.55”的可证实源码 pin。[E1][E2]

### 2.3 CANN 编译和打包的公开实现

- C++ builder 写临时 CMake 工程，`include(<cann>/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)`、`ascendc_library(kernels SHARED <source>)`，设置 `SOC_VERSION`、`RUN_MODE`、`ASCEND_CANN_PACKAGE_PATH`，返回 `build/lib/libkernels.so`。它只复制指定 C++ 文件，不能自动视为能收集 `.pto.cpp` 的任意外部 shim/headers。[S14]
- 独立 `ascend_compile` 提供 `ascend-compile input.cpp --soc ... -o output.o`，选项有 `--shared`、`--flag-style cce|npu`、`--validate-only`、`--no-auto-sync`、`-I/-D/-l`。CLI 缺省 SoC 是 `Ascend910B3`，不同于 Settings 的 310P1；`--validate-only` 是本地 C++ 文本规则检查，不调用 SDK，不是 PTO IR verifier。[S21]
- 普通 compile 构建 `<cann>/tools/ccec_compiler/bin/bisheng` 命令，Cce 路径带 tikcpp include、`--cce-aicore-only --cce-aicore-lang --cce-disable-kernel-global-attr-check`、默认 `--cce-auto-sync`、`-DTILING_KEY_VAR=0 -O2 -std=c++17`，object 用 `-c`，shared 用 `-fPIC --shared`；Npu 路径用 `--npu-arch=... -xasc`。执行前移除 `LD_PRELOAD`。[S6][S22]
- 另有公开 `compile_cube_kernel` API：AIC `dav-c220-cube` / AIV `dav-c220-vec` 分别编译，objcopy 重命名 `_mix_aic/_mix_aiv`，`ld.lld` 合并 device.o，gcc 编译带 `.ascend.kernel` 的 host stub，`ascendc_pack_kernel` 打包，最后 gcc 链 `libascendc_runtime.a`、ascendcl/runtime 等成为 `.so`；生成 `<kernel>_do` launcher。**该 API 的存在不证明不透明 PTO 后端实际调用它；CLI 的普通 `--shared` 也不等于这个 dual compile API。**[S22][S23]
- `Settings::cann_ascendnpu_ir()` 仍能拼出 `bishengir-compile` 路径，但 `needs_cpp_codegen()` 总为 true，注释说其 DMA/同步路径有问题。这是遗留配置表面，不是另一条已验证的 Rust→PTO→机器码捷径。[S6]

## 3. host-only、SDK、加载与真机的分界

| 阶段 | 输入→输出；依赖 | 本次能确认的边界 |
|---|---|---|
| 公开 Rust 宏/类型、debug/Metal registry 测试 | Rust/MLIR 文本→检查/文本；host Rust | 不需 NPU，但这些测试不覆盖私有 PTO emitter。[S3][S4][S8] |
| 预编译 backend 加载/生成 | Rust→后端输出；精确 rustc ABI、LLVM/MLIR 动态库 | 加载 host 编译插件不等于加载 ACL；完整自动路径是否在 emit 前后探测 SDK不可见，不能保证无 SDK。[S5][S24] |
| 独立 PTOAS | `.pto`→C++；host 工具及其 LLVM | 不运行设备。解析/静态 verifier/同步 pass 通过不等于数值正确。[E1] |
| 目标编译与链接 | C++→设备 object / host `.so` / executable；CANN compiler、匹配 PTO headers、host linker、ACL 库 | 可在无 NPU 的 SDK 环境做 compile-only；外部 PTOAS 文档明确此界限，但这不是本次已验证 tile-rs 全自动构建。[E2][S22] |
| Rust ACL host 构建 | `ascend_sys` bindgen + 链 ascendcl；sim 链 runtime_camodel | 即使尚未运行 host，也需要 SDK headers/libs 和 libclang；“Rust host build”不自动等于无 SDK。[S17] |
| 启动/导入/注册 | 动态链接器装入 ACL；`dlopen(kernel.o/.so)` 可运行 constructor | 不是纯文件读取。cube constructor 调 `AscendCheckSoCVersion/RegisterAscendBinary/AscendProfRegister`；注册成功仍非 kernel 执行正确。[S39][S25] |
| runtime 初始化、选卡 | `Acl::new`→`aclInit`；`Device::new`→`aclrtSetDevice`；context/stream | 已进入 runtime/设备访问合同，不能算 emit-only。真机模式需要匹配驱动、固件、库路径、设备权限；本票未取得完整版本清单。[S18][S19][S26] |
| 执行与检查 | 分配、H2D、launch、同步、D2H、CPU reference | 这是设备执行层。`ACLRS_RUN_MODE=sim` 对应另一套 CANN simulator 库；不是仅切换变量就证明无卡可跑所有 kernel。[S13][S17][S19] |

Rust runtime 有两种不可互换的装载方式：`KernelLoader::from_bin_path` 把设备 ELF 交给懒加载的 `libruntime.so` 做 `rtDevBinaryRegister/rtFunctionRegister/rtKernelLaunch`；`from_shared_lib` 则 dlopen 后读取 `g_kernel_handle`。示例 matmul/softmax 直接 dlopen `kernel.o` 并调用 `<name>_do`，softmax 注释明确原始 stub launch 曾失败。HAL 的 Ascend launcher 又按 **扩展名** 判 `.so`，不是示例中把 `.so` 改名 `kernel.o` 的装载合同；不能把 HAL 成功或其他 loader 的结果套到所有 PTO 产物。[S19][S25][S27]

`tests/npu_correctness` 是另一条测试入口：接收预生成 `*.acl.o/*.acl.so`，选 `.so` 优先，按组初始化 runtime；部分 MatmulF16、融合 matmul、reduce_sum_f16 用 `libopapi.so` 的 aclnn 算子旁路，甚至不需要 tile kernel 二进制。**该套件的 PASS 必须核对具体执行分支，不能统称 Rust→PTO 验证**。[S28]

## 4. 已有证据到底覆盖了什么

| 证据类型 | kernel / 环境 / 结果 | 强度和不足 |
|---|---|---|
| PTO 留存 IR/C++ | softmax 1×1024；benchmark 另有 1×4096、1×8192、4×256、16×256、16×512。IR 有 `pto.tload/trowmax/texp/trowsum/tstore`，C++ 有 PTO Tile、TLOAD/TEXP/TSTORE 和显式同步。[S10][S11] | 证明文本产物存在且属于 PTO library 路径；没有生成日志、后端/PTOAS hash，不能证明当前 Rust 源能重建相同文本。IR 还保留 block index out-of-scope、unhandled alloca 注释；单块证据不覆盖多块地址计算。当前三个 softmax Rust 入口在旧 artifact 中只有一个。[S40] |
| PTO 留存结果 CSV | 文件名 `bench_pto_910b2_2026-04-01.csv`，上述六 shape 各 10 次，标 PASS；max_err 分别 `1.05e-9,1.75e-10,2.62e-10,2.79e-9,3.26e-9,2.79e-9`。[S12] | 是仓库保存的运行结果表，不是本次复现/独立日志。910b2、日期来自文件名；表内无 CANN/PTOAS/驱动/host架构或命令。当前 host CSV 格式是 `BENCH,...` 且不写 correctness/max_err 列，因此缺少从当前 harness 到该表的加工过程。[S29] |
| PTO matmul 作者运行记录 | 2026-08-05 README：16³ f32，910c、CANN9.0.0、SoC 字符串 Ascend910；median 6.14μs、CPU reference max_rel_err `9.78e-08`，讨论 PTOAS0.55。[S20] | 具体 recipe + 汇总记录，比笼统 On-HW 强，但无原始 run 日志/编译器完整版本/hash/生成 IR；设备叫法和目标字符串不能自动换算物理 SKU。配套 launcher 有 CPU reference，但 mismatch 只打印，仍 return 0。[S13] |
| PTO 未跑通/限制记录 | 同 README：910b/CANN8.5.2 缺 `CompactMode`，0.55 输出不能编译；attention 需要 a5 的 Acc→Vec tmov，现有两机 a2a3 不可运行。[S20] | 作者报告的兼容性/结构限制，相关 `mlir_to_pto/module_uses_a5_ops` 不公开，无法独立审查。不能推广为所有 PTO kernel 都需要9.0或 a5；较早 softmax artifact 没有 CompactMode。 |
| 兼容 shim | softmax `compat-a3.hpp` 提到 CANN8.5.0/ccec/dav-c220-vec、910c `Ascend910_9392`；替换属性、提供 parse stub。[S30] | 证明曾保留适配代码，不证明本次使用或支持所有 op。`st_atomic` 是非原子的普通 store，注明 softmax不调用；不能当通用 PTO 兼容层。其“910B2/B4 a5 series”叫法与其他文件/外部目标分类不一致，应按实际 arch/headers核对，不据注释选设备。 |
| 直接 AscendC/低层 Rust相关记录 | async pipeline 表：910C、NPU id2、CANN8.5.0、Driver25.3.rc1；softmax 256/1024/4096/16384，layernorm 256/1024/4096，多种 Rust 与 C++ variant。[S31] | 表中没有 PTOAS pin/选择信息，不转记为 PTO PASS；softmax16384 有 C++ crash、部分 Rust FAIL，文末“all variants/all sizes”数值一致的概括与表自身冲突。id2不代表需要三卡。 |
| 测试代码合同 | Rust softmax 阈值 abs `1e-5` + sum `1e-4`，失败 exit1；matmul 五 shape abs `1e-2`，失败只 warning；softmax benchmark abs `1e-3` + row sum `0.01`，失败也只 warning。[S19][S29][S32] | 有独立 CPU oracle代码，不等于有对应 PTO运行记录；进程 exit0不一定代表正确性通过。matmul C++ launcher对 NaN无显式拒绝；这些测试不是完整数值验证规格。 |

## 5. 预编译发布物：确实检查了什么

固定提交的 installer 从 **`yijunyu/tile-rs`** 下载，不是当前 fork。默认 `v0.0.1+nightly-2025-08-04`，可用 `TILERS_CODEGEN_TAG` 覆盖，安装目录 `TILERS_CODEGEN_HOME`（默认当前目录 `tile-rs-codegen`）。nightly 固定 `2025-08-04`，要求 rustc-dev/llvm-tools/rust-src。installer 没有摘要校验步骤。[S24]

2026-09-22 第一方 Releases API 观察：[R1][R2]

| 标签 / release ID | 实际资产 | 发布信息 |
|---|---|---|
| v0.0.1+nightly-2025-08-04 / 342559398 | 仅 `tile-rs-codegen-aarch64-apple-darwin.tar.gz`，asset ID453850492，101724767 bytes | 发布 2026-06-21，asset updated 17:33:22Z；tag 当前指向 `fed207a63cd5f82df32ebbf632ea28850aae4193` |
| v0.0.2+nightly-2025-08-04 / 383425885 | 仅同名 macOS ARM 包，asset ID546560936，103449511 bytes | 发布 2026-09-06T01:26:13Z；非 installer 默认，仅核对 API 元数据 |

默认包已下载但未运行，SHA256 与 API digest 一致：

```text
231f85f11622ae38eeb724d4804432a7a394f3c2f371076de8bd57292c3d63b4
```

其目录仅含 `USAGE.md`、`.cargo/config.toml` 和 `lib/{librustc_codegen_tile,libtile_std_macros,libMLIR,libLLVM,libzstd.1}.dylib`。USAGE 列 cpp/pto 等选择，写 ABI `rustc 1.91.0-nightly f34ba774c`、LLVM/MLIR20 bundled；配置确有 `-Zcodegen-backend=lib/librustc_codegen_tile.dylib` 和 `register_tool(tile)`。**包不含 PTO emitter 源码、PTOAS executable、CANN/PTO headers 或运行验证日志。** 摘要只锁定所读包的 bytes，不证明其内部 lowering 正确，也无法把内部私有源码提交对应到本票固定源码。[R1]

workflow 确实声明从 secret 指定的私有 repo 的 `main` 构建，然后发布 macOS ARM / Linux x86_64 tarball；installer 还接受 Linux aarch64，但 workflow matrix 没有该项。实际 API 两个 release 都只有 macOS ARM，不能给 Linux Ascend host 宣称已有默认可用下载。workflow 的 private `main` 也是浮动 ref，public tag 不是私有源码 pin。[S33] `install-smoke` 仅在 macos-14 安装、检查文件、otool依赖和配置文字，**没有加载 rustc backend 编译 kernel，也没有 PTOAS或NPU步骤**；本次未采集其 run日志，不把配置存在当执行成功。[S34]

## 6. 冲突、遗留与缺口

- README 将 `cpp,pto` 合在“Ascend On-HW 910B”一行；应拆到具体 kernel/工具组合，不能解释成所有分支已验。`tile.rs` 还称 PTO future path，CHANGELOG又声称公开 emitters 有 `mlir_to_pto/mlir_to_gpu`；当前注册源码不支持这条说法。[S1][S3][S4][S35]
- `bench.sh` 的 PTO选择分支存在，但先尝试 `cargo build -p rustc_codegen_mlir`，公开 workspace没有这个旧名包；`--skip-build` 只是跳过构建，不重新证明 binary 的后端归属。不能把脚本当可重跑的当前端到端证据。[S15][S16]
- `pto_run.cpp` 头注释说910B和`__aicore__`，实际代码使用`AICORE`，README则说910c/CANN9与`__aicore__`失败；以具体代码/recipe限定，不复制旧注释作运行条件。[S13][S20]
- `.acl.pto/.acl.o` 留存命名与 builder 的 `.tile.*` 消费合同不同；部分 host 用裸 object注册、部分用 dlopen `_do`。这些不是可以只改后缀互换的格式。[S5][S19][S25][S28]

仍阻碍后续适用性判断的具体缺口：

1. **前半链 provenance**：可用 Linux后端包/摘要、实际私有源码版本、Rust→MLIR→PTO的 emit-only接口、完整选项、失败/fallback策略、缓存键及留存中间文件合同。
2. **跨工具 pin**：recipe 的 PTOAS0.55到底是哪一构建；PTO dialect版本、PTO ISA headers commit（SDK内置还是外置）、CANN/ccec/bisheng完整版本及目标arch。当前外部PTOAS支持不能追溯证明旧产物兼容。
3. **连通性**：从当前 Rust入口到一个命名一致的 `.pto/.cpp/.so` 的完整命令和摘要；matmul缺失输入；softmax artifact可能的后处理、compat shim接入；`_do`/object/双核wrapper究竟哪个是实际测试产物。
4. **运行 provenance**：原始stdout/stderr、独立oracle、失败退出、非有限值检查、多块/shape/dtype覆盖，CSV加工过程。不能只拿PASS表、checksum或成功退出码判断。
5. **环境分层**：无SDK host emit是否可用；SDK-only链接是否可用；哪套CANN simulator实际验证过；真实SKU、host架构、驱动固件/权限。910b/910c/Ascend910/a3/a5命名需结合实际设备报告，不能用标签推导。

## 7. 前置调查落在哪一段

[PTO ISA / PTOAS 工具链调查](https://github.com/Jopqior/tile-rs/issues/53) 现在可对应到本仓库**已有 `.pto` 之后**的 verifier/pass、PTO-library C++、SDK compile-only、模拟/真机验证层；不能覆盖缺失的 Rust→MLIR/PTO实现或保证旧产物被其2026-09快照接受。本次直接复核该快照第一方文档：`STAGE=build` 可无 `/dev/davinci*` 编译，但不能验证ACL/驱动/权限/数值。[E1][E2]

[triton-ascend](https://github.com/Jopqior/tile-rs/issues/50)、[TileLang Ascend](https://github.com/Jopqior/tile-rs/issues/51)、[torch_npu](https://github.com/Jopqior/tile-rs/issues/52) 和 [其他可比项目](https://github.com/Jopqior/tile-rs/issues/54) 的证据，后续可按“host工具/SDK目标编译/动态加载初始化/设备执行与oracle”这些实际阶段对齐；直接AscendC案例不证明PTOAS段，torch_npu加载或设备测试不证明Rust emitter段，PTOAS测试也不证明本仓库host ABI。这里仅提供定位，不替 [适用边界判断](https://github.com/Jopqior/tile-rs/issues/55) 作选择，不设计CI、不指定硬件或采购规模。

## 来源索引

所有 S 引用均为本票固定提交；R 是调查日第一方API/发布资产快照（资产可替换，因此同时记录ID与摘要）；E 是独立固定的外部源码快照。

[S1]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/README.md#L17-L77
[S2]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_kernel_builder/src/kernel_builder/mod.rs#L34-L93
[S3]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/src/targets.rs#L1-L21
[S4]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_codegen/src/emitters.rs#L1-L25
[S5]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_kernel_builder/src/kernel_builder/rs_builder.rs#L16-L223
[S6]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_kernel_builder_config/src/lib.rs#L5-L266
[S7]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_matmul/kernels/src/lib.rs#L1-L45
[S8]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_std_macros/src/lib.rs#L8-L175
[S9]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_softmax/kernels/src/lib.rs#L1-L51
[S10]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_softmax/artifacts/tile_softmax_kernels.acl.pto.cpp#L1-L83
[S11]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/bench_softmax_tile/artifacts/tile_softmax_bench_kernels.acl.pto#L1-L169
[S12]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/bench_softmax_tile/results/bench_pto_910b2_2026-04-01.csv#L1-L61
[S13]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tools/ascend/pto_run.cpp#L1-L132
[S14]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_kernel_builder/src/kernel_builder/cpp_builder.rs#L8-L65
[S15]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/Cargo.toml#L1-L30
[S16]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/benchmarks/kernel_bench/bench.sh#L60-L183
[S17]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_sys/build.rs#L1-L104
[S18]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_rs/src/device.rs#L16-L39
[S19]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_softmax/src/main.rs#L32-L130
[S20]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tools/ascend/README.md#L1-L72
[S21]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_compile/src/main.rs#L7-L162
[S22]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_compile/src/compiler.rs#L39-L367
[S23]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_compile/src/lib.rs#L93-L139
[S24]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/scripts/install.sh#L19-L83
[S25]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_rs/src/kernel.rs#L14-L287
[S26]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_rs/src/acl.rs#L15-L24
[S27]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/tile_hal/src/ascend.rs#L150-L230
[S28]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/tests/npu_correctness/src/main.rs#L432-L608
[S29]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/bench_softmax_tile/src/main.rs#L20-L239
[S30]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_softmax/artifacts/tile_softmax_kernels.acl.pto.compat-a3.hpp#L1-L66
[S31]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/benchmarks/kernel_bench/results/async_pipeline_results.md#L1-L45
[S32]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_matmul/src/main.rs#L65-L234
[S33]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/.github/workflows/codegen-release.yml#L5-L155
[S34]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/.github/workflows/install-smoke.yml#L20-L57
[S35]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/CHANGELOG.md#L34-L43
[S36]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_matmul/build.rs#L1-L13
[S37]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_softmax/build.rs#L1-L13
[S38]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/.cargo/config.toml#L1-L2
[S39]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/crates/ascend_compile/src/compiler.rs#L592-L677
[S40]: https://github.com/Jopqior/tile-rs/blob/e8d1acdf58f4e2239bf6be9d59333d7a337282b3/examples/tile_softmax/artifacts/tile_softmax_kernels.acl.pto#L1-L31
[R1]: https://api.github.com/repos/yijunyu/tile-rs/releases/342559398
[R2]: https://api.github.com/repos/yijunyu/tile-rs/releases/383425885
[E1]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/README_en.md#L1-L14
[E2]: https://github.com/hw-native-sys/PTOAS/blob/97e9ecb247744c587171c73ab58ac2423bba3197/docs/no_npu_compile_only_guide_zh.md#L1-L163

## Resolution comment 草稿（待逐字批准，未发布）

已按固定提交核实 Rust kernel、后端选择、PTO 中间产物、PTOAS、CANN 编译与 ACL host 的实际边界。仓库保留了 PTO 产物和运行结果，但 Rust→MLIR/PTO 的核心实现依赖不透明预编译后端，不能从公开接口补推完整调用链。报告区分了直接 AscendC 与 PTO、编译链接与加载执行，并列出发布资产、版本对应、host ABI 和运行记录的缺口，供后续适用边界判断使用。未复跑设备测试。

报告：`research/tile-rs-pto-path` 分支的 `docs/research/tile-rs-pto-path.md`。本票不选择 CI 方案或硬件。
