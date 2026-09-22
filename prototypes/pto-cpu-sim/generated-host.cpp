// Throwaway host I/O only. generated.cpp is emitted by PTOAS, never patched.
#include <cstdint>
#include <cstddef>
#include "generated.cpp"
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::array<float, 256> a{}, b{}, out;
    out.fill(std::numeric_limits<float>::quiet_NaN());
    std::ifstream input(argv[1]);
    for (int i = 0; i < 256; ++i) {
        if (!(input >> a[i] >> b[i])) return 2;
    }
    add_generated(a.data(), b.data(), out.data());
    std::ofstream output(argv[2]);
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : out) output << value << '\n';
    return output ? 0 : 2;
}
