// Host for the 1x256 Rust add case, shared between camodel and NPU.
#include <acl/acl.h>
#include <dlfcn.h>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

extern "C" void launch_add(float*, float*, float*, void*);

#define ACL(call) \
    do { \
        const auto rc = (call); \
        if (rc != ACL_SUCCESS) { \
            std::cerr << "FAIL: " #call " returned " << rc << '\n'; \
            std::exit(3); \
        } \
    } while (0)

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const char* mode = std::getenv("EXECUTION_MODE");
    if (!mode || (std::string(mode) != "sim" && std::string(mode) != "npu")) return 2;
    const bool simulated = std::string(mode) == "sim";
    const char* device_text = std::getenv("ASCEND_DEVICE_ID");
    if (!device_text || !*device_text) return 2;
    char* end{};
    const long parsed_device = std::strtol(device_text, &end, 10);
    if (*end || parsed_device < 0 || parsed_device > 65535) return 2;
    const int device = static_cast<int>(parsed_device);
    // Check the actual loaded runtime, not just environment-variable intent.
    for (const char* symbol : {"rtSetDevice", "rtMalloc", "rtMemcpy", "rtStreamSynchronize", "rtKernelLaunch"}) {
        Dl_info info{};
        void* address = dlsym(RTLD_DEFAULT, symbol);
        if (!address || !dladdr(address, &info) || !info.dli_fname) {
            std::cerr << "FAIL: unresolved runtime symbol " << symbol << '\n';
            return 3;
        }
        const bool camodel = std::string(info.dli_fname).find("camodel") != std::string::npos;
        if (camodel != simulated) {
            std::cerr << "FAIL: runtime mode mismatch for " << symbol << '\n';
            return 3;
        }
        std::cout << "binding " << symbol << '=' << info.dli_fname << '\n';
    }
    constexpr size_t bytes = 256 * sizeof(float);
    std::array<float, 256> a{}, b{}, out;
    out.fill(std::numeric_limits<float>::quiet_NaN());
    std::ifstream input(argv[1]);
    for (size_t i = 0; i < a.size(); ++i)
        if (!(input >> a[i] >> b[i])) return 2;
    ACL(aclInit(nullptr));
    ACL(aclrtSetDevice(device));
    aclrtStream stream{};
    ACL(aclrtCreateStream(&stream));
    void *da{}, *db{}, *dc{};
    ACL(aclrtMalloc(&da, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&db, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMalloc(&dc, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL(aclrtMemcpy(da, bytes, a.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(db, bytes, b.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL(aclrtMemcpy(dc, bytes, out.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    launch_add(static_cast<float*>(da), static_cast<float*>(db), static_cast<float*>(dc), stream);
    ACL(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
    ACL(aclrtSynchronizeStream(stream));
    ACL(aclrtMemcpy(out.data(), bytes, dc, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL(aclrtFree(dc));
    ACL(aclrtFree(db));
    ACL(aclrtFree(da));
    ACL(aclrtDestroyStream(stream));
    ACL(aclrtResetDevice(device));
    ACL(aclFinalize());
    std::ofstream output(argv[2]);
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : out) output << value << '\n';
    return output ? 0 : 2;
}
