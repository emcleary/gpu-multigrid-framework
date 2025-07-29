#include "iterator_naive.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


namespace naive {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2, const int color) {
    int i = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;
    if (0 < i && i < v.size() - 1)
        v[i] = (v[i-1] + v[i+1] + h2 * f[i]) / 2;
}

__global__
void boundaries(ArrayRaw v, const ArrayRaw f, const BoundaryConditions bcs, const double h2) {
    const int nh = v.size() - 1;
    
    if (bcs.is_periodic_x()) {
        v[0] = (v[nh-1] + v[1] + f[0] * h2) / 2;
        v[nh] = v[0];
    } else {
        if (bcs.is_west_dirichlet()) {
            v[0] = f[0];
        } else { // neumann
            v[0] = v[1] + f[0] * h2 / 2;
        }

        if (bcs.is_east_dirichlet()) {
            v[nh] = f[nh];
        } else { // neumann
            v[nh] = v[nh-1] + f[nh] * h2 / 2;
        }
    }
}
} // namespace naive

void IteratorNaive::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const uint n = v.size() / 2;
    const uint threads = std::min(m_max_threads_per_block, n);
    const uint blocks = (n + threads - 1) / threads;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    // Red-Black Gauss Seidel
    naive::kernel<<<blocks, threads>>>(v, f, h2, 1);
    naive::kernel<<<blocks, threads>>>(v, f, h2, 2);
    naive::boundaries<<<1, 1>>>(v, f, bcs, h2);
}
