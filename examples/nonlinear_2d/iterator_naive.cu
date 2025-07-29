#include "iterator_naive.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


namespace iterator_naive {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h, const double gamma, const int color) {
    const int n = v.get_ncols();
    const double h2 = h * h;
    const int k = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;
    const int i = k / n;
    const int j = k % n;

    if (0 < i && i < n-1 && 0 < j && j < n-1) {
        const double denom = 4 + h / 2 * gamma * (v(i+1, j) - v(i-1, j));
        v(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) + f(i, j) * h2) / denom;
    }
}

__global__
void boundaries(ArrayRaw v, ArrayRaw f, const double h, const double gamma,
        const BoundaryConditions bcs, const NonlinearEquation eqn, const ArrayRaw x, const ArrayRaw y,
        const int color) {

    const int nh = v.get_ncols() - 1;
    const double h2 = h * h;
    const int i = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;

    if (bcs.is_west_neumann()) {
        if (0 < i && i < nh) {
            const double un = eqn.neumann_bc_west(x[0], y[i]);
            const double denom = 4 + h2 * gamma * un;
            v(0, i) = (2 * v(1, i) + v(0, i-1) + v(0, i+1) + f(0, i) * h2) / denom;
        }
    }
}
} // namespace iterator_naive

void IteratorNaive::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    // will not work for alternating directions
    assert(v.get_nrows() == v.get_ncols());

    // will not work for anisotropic meshes
    assert(grid.get_cell_width() == grid.get_cell_height());

    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    const uint threads = std::min(m_max_threads_per_block, static_cast<uint>(v.size()) - 1);
    const uint blocks = (v.size() + threads - 1) / threads;

    const uint n_1d = v.get_nrows() / 2;
    const uint threads_1d = std::min(m_max_threads_per_block, n_1d);
    const uint blocks_1d = (n_1d + threads_1d - 1) / threads_1d;

    const Array& x = grid.get_x();
    const Array& y = grid.get_y();

    // odds
    iterator_naive::kernel<<<blocks, threads>>>(v, f, h, gamma, 1);
    iterator_naive::boundaries<<<blocks_1d, threads_1d>>>(v, f, h, gamma, bcs, *m_eqn, x, y, 1);

    // evens
    iterator_naive::kernel<<<blocks, threads>>>(v, f, h, gamma, 0);
    iterator_naive::boundaries<<<blocks_1d, threads_1d>>>(v, f, h, gamma, bcs, *m_eqn, x, y, 0);
}
