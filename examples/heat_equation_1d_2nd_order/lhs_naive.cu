#include "lhs_naive.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace lhs_naive {
__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < v.size() - 1)
        lhs[idx] = eval_lhs(v, h2, idx);

    // set lhs to rhs so residuals are zero
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}
}

void LHSNaive::run_host(Array& lhs, Array& v, const Array& f, const Grid& grid) {
    const int n_pts = v.size();
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    for (int i = 1; i < n_pts - 1; i += 1)
        lhs[i] = eval_lhs(v, h2, i);
    // dirichlet BCs
    lhs.front() = f.front();
    lhs.back() = f.back();
}

void LHSNaive::run_device(Array& lhs, Array& v, const Array& f, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size());
    const int blocksPerGrid = (v.size() + threadsPerBlock - 1) / threadsPerBlock;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    lhs_naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs, v, f, h2);
}
