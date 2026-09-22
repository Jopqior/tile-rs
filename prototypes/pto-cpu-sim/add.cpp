// Throwaway PTO CPU-SIM experiment. Not a tile-rs backend or NPU binary.
#include <cstdint>
#include <cstddef>
#include <pto/pto-inst.hpp>
#include <array>
#include <fstream>
#include <iomanip>
#include <limits>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    constexpr int n = 256;
    std::array<float, n> a{}, b{}, out;
    out.fill(std::numeric_limits<float>::quiet_NaN());
    std::ifstream input(argv[1]);
    for (int i = 0; i < n; ++i) {
        if (!(input >> a[i] >> b[i])) return 2;
    }
    using namespace pto;
    using Global = GlobalTensor<float, Shape<1, 1, 1, 1, n>, Stride<n, n, n, n, 1>>;
    using Vector = Tile<TileType::Vec, float, 1, n, BLayout::RowMajor, 1, n>;
    Vector x, y, z;
    TASSIGN(x, 0);
    TASSIGN(y, n * sizeof(float));
    TASSIGN(z, 2 * n * sizeof(float));
    Global ga(a.data()), gb(b.data()), gc(out.data());
    TLOAD(x, ga);
    TLOAD(y, gb);
    TADD(z, x, y);
    TSTORE(gc, z);
    std::ofstream output(argv[2]);
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (float value : out) output << value << '\n';
    return output ? 0 : 2;
}
