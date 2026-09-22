// THROWAWAY: shared ACL host for both device entry points.
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
#define ACL(call) do { const auto rc = (call); if (rc != ACL_SUCCESS) { \
    std::cerr << "FAIL: " #call " returned " << rc << '\n'; std::exit(3); } } while (0)

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    // Check process-global symbol resolution, not just presence of a simulator file.
    for (const char* symbol : {"rtSetDevice", "rtMalloc", "rtMemcpy", "rtStreamSynchronize", "rtKernelLaunch"}) {
        Dl_info info{};
        void* address = dlsym(RTLD_DEFAULT, symbol);
        if (!address || !dladdr(address, &info) || !info.dli_fname ||
            std::string(info.dli_fname).find("libruntime_camodel.so") == std::string::npos) {
            std::cerr << "FAIL: simulator binding for " << symbol << '\n';
            return 3;
        }
        std::cout << "binding " << symbol << "=libruntime_camodel.so\n";
    }
    constexpr size_t bytes = 256 * sizeof(float);
    std::array<float, 256> a{}, b{}, out;
    out.fill(std::numeric_limits<float>::quiet_NaN());
    std::ifstream input(argv[1]);
    for (size_t i = 0; i < a.size(); ++i)
        if (!(input >> a[i] >> b[i])) return 2;
    ACL(aclInit(nullptr));
    ACL(aclrtSetDevice(0));
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
    ACL(aclrtResetDevice(0));
    ACL(aclFinalize());
    std::ofstream output(argv[2]);
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : out) output << value << '\n';
    return output ? 0 : 2;
}
