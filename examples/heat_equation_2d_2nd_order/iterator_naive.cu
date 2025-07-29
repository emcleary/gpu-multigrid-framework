#include "iterator_naive.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


namespace iterator_naive {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2, const int color) {
    const int n = v.get_ncols();
    const int k = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;
    const int i = k / n;
    const int j = k % n;

    if (0 < i && i < n-1 && 0 < j && j < n-1)
        v(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) + f(i, j) * h2) / 4;
}

__global__
void boundaries(ArrayRaw v, const ArrayRaw f, const double h2, const int color,
        const BoundaryConditions bcs) {
    const int nh = v.get_nrows() - 1;
    const int i = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;

    if (0 < i && i < nh) {
        if (bcs.is_periodic_x()) {
            v(0, i) = (v(nh-1, i) + v(1, i) + v(0, i-1) + v(0, i+1) + f(0, i) * h2) / 4;
            v(nh, i) = v(0, i);
        }

        if (bcs.is_periodic_y()) {
            v(i, 0) = (v(i-1, 0) + v(i+1, 0) + v(i, nh-1) + v(i, 1) + f(i, 0) * h2) / 4;
            v(i, nh) = v(i, 0);
        }

        if (bcs.is_north_neumann())
            v(i, nh) = (2 * v(i, nh-1) + v(i-1, nh) + v(i+1, nh) + f(i, nh) * h2) / 4;

        if (bcs.is_south_neumann())
            v(i, 0) = (2 * v(i, 1) + v(i-1, 0) + v(i+1, 0) + f(i, 0) * h2) / 4;

        if (bcs.is_east_neumann())
            v(nh, i) = (2 * v(nh-1, i) + v(nh, i-1) + v(nh, i+1) + f(nh, i) * h2) / 4;

        if (bcs.is_west_neumann())
            v(0, i) = (2 * v(1, i) + v(0, i-1) + v(0, i+1) + f(0, i) * h2) / 4;
    }

}

__global__
void corners(ArrayRaw v, const ArrayRaw f, const double h2, const BoundaryConditions bcs) {

    const int nh = v.get_nrows() - 1;

    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        v(0, 0) = (v(nh-1, 0) + v(1, 0) + v(0, nh-1) + v(0, 1) + f(0, 0) * h2) / 4;
        v(nh, 0) = v(0, 0);
        v(0, nh) = v(0, 0);
        v(nh, nh) = v(0, 0);
    }

    if (bcs.is_periodic_x()) {
        if (bcs.is_north_neumann()) {
            v(0, nh) = (2 * v(0, nh-1) + v(nh-1, nh) + v(1, nh) + f(0, nh) * h2) / 4;
            v(nh, nh) = v(0, nh);
        }

        if (bcs.is_south_neumann()) {
            v(0, 0) = (2 * v(0, 1) + v(nh-1, 0) + v(1, 0) + f(0, 0) * h2) / 4;
            v(nh, 0) = v(0, 0);
        }
    }

    if (bcs.is_periodic_y()) {
        if (bcs.is_east_neumann()) {
            v(nh, 0) = (2 * v(nh-1, 0) + v(nh, nh-1) + v(nh, 1) + f(nh, 0) * h2) / 4;
            v(nh, nh) = v(nh, 0);
        }

        if (bcs.is_west_neumann()) {
            v(0, 0) = (2 * v(1, 0) + v(0, nh-1) + v(0, 1) + f(0, 0) * h2) / 4;
            v(0, nh) = v(0, 0);
        }
    }

    if (bcs.is_east_neumann() && bcs.is_north_neumann())
        v(nh, nh) = (2 * v(nh-1, nh) + 2 * v(nh, nh-1) + f(nh, nh) * h2) / 4;

    if (bcs.is_east_neumann() && bcs.is_south_neumann())
        v(nh, 0) = (2 * v(nh-1, 0) + 2 * v(nh, 1) + f(nh, 0) * h2) / 4;

    if (bcs.is_west_neumann() && bcs.is_south_neumann())
        v(0, 0) = (2 * v(1, 0) + 2 * v(0, 1) + f(0, 0) * h2) / 4;

    if (bcs.is_west_neumann() && bcs.is_north_neumann())
        v(0, nh) = (2 * v(1, nh) + 2 * v(0, nh-1) + f(0, nh) * h2) / 4;
}
} // namespace naive


void IteratorNaive::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    // will not work for alternating directions
    assert(v.get_nrows() == v.get_ncols());

    // will not work for anisotropic meshes
    assert(grid.get_cell_width() == grid.get_cell_height());

    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    const uint ns = v.size();
    const uint threads = std::min(m_max_threads_per_block, ns);
    const uint blocks = (ns + threads - 1) / threads;

    const uint n_1d = v.get_nrows() / 2;
    const uint threads_1d = std::min(m_max_threads_per_block, n_1d);
    const uint blocks_1d = (n_1d + threads_1d - 1) / threads_1d;

    // odds
    iterator_naive::kernel<<<blocks, threads>>>(v, f, h2, 1);
    iterator_naive::boundaries<<<blocks_1d, threads_1d>>>(v, f, h2, 1, bcs);

    // evens
    iterator_naive::kernel<<<blocks, threads>>>(v, f, h2, 0);
    iterator_naive::boundaries<<<blocks_1d, threads_1d>>>(v, f, h2, 0, bcs);
    iterator_naive::corners<<<1, 1>>>(v, f, h2, bcs);
}
