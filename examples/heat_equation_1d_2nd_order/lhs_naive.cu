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
void kernel(ArrayRaw lhs, ArrayRaw v, const BoundaryConditions bcs, const double h2) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int nh = v.size() - 1;

    if (0 < i && i < nh)
        lhs[i] = - (v[i-1] - 2 * v[i] +  v[i+1]) / h2;

    // set lhs to rhs so residuals are zero
    if (i == 0) {
        if (bcs.is_periodic_x()) {
            lhs[0] = (-v[nh-1] + 2 * v[0] - v[1]) / h2;
            lhs[nh-1] = (-v[nh-2] + 2 * v[nh-1] - v[0]) / h2;
            lhs[nh] = lhs[0];
        } else {
            if (bcs.is_west_dirichlet()) {
                lhs[0] = v[0];
            } else { // neumann
                lhs[0] = 2 * (v[0] - v[1]) / h2;
            }

            if (bcs.is_east_dirichlet()) {
                lhs[nh] = v[nh];
            } else { // neumann
                lhs[nh] = 2 * (v[nh] - v[nh-1]) / h2;
            }
        }
    }
}
} // namespace lhs_naive

void LHSNaive::run_device(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const uint threads = std::min(m_max_threads_per_block, static_cast<uint>(v.size()));
    const uint blocks = (v.size() + threads - 1) / threads;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    lhs_naive::kernel<<<blocks, threads>>>(lhs, v, bcs, h2);
}
