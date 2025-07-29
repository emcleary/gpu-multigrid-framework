#include "lhs_naive.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


namespace lhs_naive {
__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const double h, const double gamma, const BoundaryConditions bcs) {
    const int nh = v.get_nrows() - 1;
    const double h2 = h * h;
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (0 < i && i < nh && 0 < j && j < nh)
        lhs(i, j) = (4 * v(i, j) - v(i-1, j) - v(i+1, j) - v(i, j-1) - v(i, j+1)) / h2
            + gamma * v(i, j) * (v(i+1, j) - v(i-1, j)) / 2 / h;
}

__global__
void boundaries(ArrayRaw lhs, ArrayRaw v, const double h, const double gamma,
        const BoundaryConditions bcs, const NonlinearEquation eqn, const ArrayRaw x, const ArrayRaw y) {
    const int nh = v.get_nrows() - 1;
    const double h2 = h * h;
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (bcs.is_west_neumann()) {
        if (0 < i && i < nh) {
            const double un = eqn.neumann_bc_west(x[0], y[i]);
            lhs(0, i) = (4 * v(0, i) - 2 * v(1, i) - v(0, i+1) - v(0, i-1)) / h2
                + gamma * v(0, i) * un;
        }
    }

    // internal AND corners
    if (0 <= i && i <= nh) {
        if (bcs.is_north_dirichlet())
            lhs(i, nh) = v(i, nh);

        if (bcs.is_south_dirichlet())
            lhs(i, 0) = v(i, 0);

        if (bcs.is_east_dirichlet())
            lhs(nh, i) = v(nh, i);

        if (bcs.is_west_dirichlet())
            lhs(0, i) = v(0, i);
    }    
}
} // namespace lhs_naive

void LHSNaive::run_device(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    // will not work for alternating directions
    assert(lhs.get_nrows() == lhs.get_ncols());

    // will not work for anisotropic meshes
    assert(grid.get_cell_width() == grid.get_cell_height());

    const Array& x = grid.get_x();
    const Array& y = grid.get_y();
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    const uint n_2d = lhs.get_nrows();
    const uint threads = std::min(32U, n_2d);
    const uint blocks = (n_2d + threads - 1) / threads;
    dim3 threads_2d(threads, threads);
    dim3 blocks_2d(blocks, blocks);

    const uint n_1d = v.get_nrows();
    const uint threads_1d = std::min(m_max_threads_per_block, n_1d);
    const uint blocks_1d = (n_1d + threads_1d - 1) / threads_1d;

    lhs_naive::kernel<<<blocks_2d, threads_2d>>>(lhs, v, h, gamma, bcs);
    lhs_naive::boundaries<<<blocks_1d, threads_1d>>>(lhs, v, h, gamma, bcs, *m_eqn, x, y);
}
