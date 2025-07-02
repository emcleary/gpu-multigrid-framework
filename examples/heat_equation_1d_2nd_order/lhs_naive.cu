#include "lhs_naive.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;
using gmf::modules::BoundaryConditions;


namespace lhs_naive {
__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const BoundaryConditions bcs, const double h2) {
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int n = v.size() - 1;

    if (0 < idx && idx < n)
        lhs[idx] = eval_lhs(v, h2, idx);

    // set lhs to rhs so residuals are zero
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (bcs.is_periodic()) {
            lhs[0] = (-v[n-1] + 2 * v[0] - v[1]) / h2;
            lhs[n-1] = (-v[n-2] + 2 * v[n-1] - v[0]) / h2;
            lhs[n] = lhs[0];
        } else {
            if (bcs.is_left_dirichlet()) {
                lhs[0] = v[0];
            } else { // neumann
                lhs[0] = 2 * (v[0] - v[1]) / h2;
            }

            if (bcs.is_right_dirichlet()) {
                lhs[n] = v[n];
            } else { // neumann
                lhs[n] = 2 * (v[n] - v[n-1]) / h2;
            }
        }
    }
}
}

void LHSNaive::run_host(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int n = v.size() - 1;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    for (int i = 1; i < n; i += 1)
        lhs[i] = eval_lhs(v, h2, i);

    if (bcs.is_periodic()) {
        lhs[0] = (-v[n-1] + 2 * v[0] - v[1]) / h2;
        lhs[n-1] = (-v[n-2] + 2 * v[n-1] - v[0]) / h2;
        lhs[n] = lhs[0];
    } else {
        if (bcs.is_left_dirichlet()) {
            lhs[0] = v[0];
        } else { // neumann
            lhs[0] = 2 * (v[0] - v[1]) / h2;
        }

        if (bcs.is_right_dirichlet()) {
            lhs[n] = v[n];
        } else { // neumann
            lhs[n] = 2 * (v[n] - v[n-1]) / h2;
        }
    }
}

void LHSNaive::run_device(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size());
    const int blocksPerGrid = (v.size() + threadsPerBlock - 1) / threadsPerBlock;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    lhs_naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs, v, bcs, h2);
}
