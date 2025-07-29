#include "lhs_naive.cuh"

#include <iomanip>

#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


namespace lhs_naive {
__global__
void kernel(ArrayRaw lhs, const ArrayRaw v, const double h2) {
    const int nh = v.get_nrows() - 1;
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (0 < i && i < nh && 0 < j && j < nh)
        lhs(i, j) = (4 * v(i, j) - v(i-1, j) - v(i+1, j) - v(i, j-1) - v(i, j+1)) / h2;
}

__global__
void boundaries(ArrayRaw lhs, const ArrayRaw v, const double h2,
        const BoundaryConditions bcs) {
    const int nh = v.get_nrows() - 1;
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (0 < i && i < nh) {
        if (bcs.is_periodic_x()) {
            lhs(0, i) = (4 * v(0, i) - v(0, i-1) - v(0, i+1) - v(1, i) - v(nh-1, i)) / h2;
            lhs(nh, i) = lhs(0, i);
        }

        if (bcs.is_periodic_y()) {
            lhs(i, 0) = (4 * v(i, 0) - v(i-1, 0) - v(i+1, 0) - v(i, 1) - v(i, nh-1)) / h2;
            lhs(i, nh) = lhs(i, 0);
        }

        if (bcs.is_north_neumann())
            lhs(i, nh) = (4 * v(i, nh) - 2 * v(i, nh-1) - v(i+1, nh) - v(i-1, nh)) / h2;

        if (bcs.is_south_neumann())
            lhs(i, 0) = (4 * v(i, 0) - 2 * v(i, 1) - v(i+1, 0) - v(i-1, 0)) / h2;

        if (bcs.is_east_neumann())
            lhs(nh, i) = (4 * v(nh, i) - 2 * v(nh-1, i) - v(nh, i+1) - v(nh, i-1)) / h2;

        if (bcs.is_west_neumann())
            lhs(0, i) = (4 * v(0, i) - 2 * v(1, i) - v(0, i+1) - v(0, i-1)) / h2;

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

__global__
void corners(ArrayRaw lhs, const ArrayRaw v, const double h2,
        const BoundaryConditions bcs) {

    const int nh = v.get_nrows() - 1;

    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        lhs(0, 0) = (4 * v(0, 0) - v(1, 0) - v(nh-1, 0) - v(0, 1) - v(0, nh-1)) / h2;
        lhs(nh, 0) = lhs(0, 0);
        lhs(0, nh) = lhs(0, 0);
        lhs(nh, nh) = lhs(0, 0);
    }

    if (bcs.is_periodic_x()) {
        if (bcs.is_north_neumann()) {
            lhs(0, nh) = (4 * v(0, nh) - 2 * v(0, nh-1) - v(1, nh) - v(nh-1, nh)) / h2;
            lhs(nh, nh) = lhs(0, nh);
        }
        if (bcs.is_south_neumann()) {
            lhs(0, 0) = (4 * v(0, 0) - 2 * v(0, 1) - v(1, 0) - v(nh-1, 0)) / h2;
            lhs(nh, 0) = lhs(0, 0);
        }
    }

    if (bcs.is_periodic_y()) {
        if (bcs.is_east_neumann()) {
            lhs(nh, 0) = (4 * v(nh, 0) - 2 * v(nh-1, 0) - v(nh, 1) - v(nh, nh-1)) / h2;
            lhs(nh, nh) = lhs(nh, 0);
        }
        if (bcs.is_west_neumann()) {
            lhs(0, 0) = (4 * v(0, 0) - 2 * v(1, 0) - v(0, 1) - v(0, nh-1)) / h2;
            lhs(0, nh) = lhs(0, 0);
        }
    }

    if (bcs.is_east_neumann() && bcs.is_north_neumann())
        lhs(nh, nh) = (4 * v(nh, nh) - 2 * v(nh-1, nh) - 2 * v(nh, nh-1)) / h2;

    if (bcs.is_east_neumann() && bcs.is_south_neumann())
        lhs(nh, 0) = (4 * v(nh, 0) - 2 * v(nh-1, 0) - 2 * v(nh, 1)) / h2;

    if (bcs.is_west_neumann() && bcs.is_south_neumann())
        lhs(0, 0) = (4 * v(0, 0) - 2 * v(1, 0) - 2 * v(0, 1)) / h2;

    if (bcs.is_west_neumann() && bcs.is_north_neumann())
        lhs(0, nh) = (4 * v(0, nh) - 2 * v(1, nh) - 2 * v(0, nh-1)) / h2;

    if (bcs.is_north_dirichlet() || bcs.is_east_dirichlet())
        lhs(nh, nh) = v(nh, nh);

    if (bcs.is_south_dirichlet() || bcs.is_east_dirichlet())
        lhs(nh, 0) = v(nh, 0);

    if (bcs.is_south_dirichlet() || bcs.is_west_dirichlet())
        lhs(0, 0) = v(0, 0);

    if (bcs.is_north_dirichlet() || bcs.is_west_dirichlet())
        lhs(0, nh) = v(0, nh);

}
} // namespace lhs_naive

void LHSNaive::run_device(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    // will not work for alternating directions
    assert(lhs.get_nrows() == lhs.get_ncols());

    // will not work for anisotropic meshes
    assert(grid.get_cell_width() == grid.get_cell_height());

    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    const uint n_2d = lhs.get_nrows();
    const uint threads = std::min(32U, n_2d);
    const uint blocks = (n_2d + threads - 1) / threads;
    dim3 threads_2d(threads, threads);
    dim3 blocks_2d(blocks, blocks);

    const uint n_1d = v.get_nrows();
    const uint threads_1d = std::min(m_max_threads_per_block, n_1d);
    const uint blocks_1d = (n_1d + threads_1d - 1) / threads_1d;

    lhs_naive::kernel<<<blocks_2d, threads_2d>>>(lhs, v, h2);
    lhs_naive::boundaries<<<blocks_1d, threads_1d>>>(lhs, v, h2, bcs);
    lhs_naive::corners<<<1, 1>>>(lhs, v, h2, bcs);
}
