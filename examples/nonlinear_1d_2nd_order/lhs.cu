#include "lhs.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


__host__ __device__
inline double eval(ArrayRaw& v, const double h, const double gamma, const int i) {
    double lhs = (-v[i-1] + 2 * v[i] - v[i+1]) / h;
    lhs += gamma * v[i] * (v[i+1] - v[i-1]) / 2;
    return lhs / h;
}

namespace example_lhs {
__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h, const double gamma) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < v.size() - 1)
        lhs[idx] = eval(v, h, gamma, idx);

    // set lhs to rhs so residuals are zero
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}
}

void LHSNonlinear::run_host(Array& lhs, Array& v, const Array& f, const Grid& grid) {
    const int n_pts = v.size();
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();
    
    for (int i = 1; i < n_pts - 1; i += 1)
        lhs[i] = eval(v, h, gamma, i);

    // dirichlet BCs
    lhs.front() = f.front();
    lhs.back() = f.back();
}

void LHSNonlinear::run_device(Array& lhs, Array& v, const Array& f, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size());
    const int blocksPerGrid = (v.size() + threadsPerBlock - 1) / threadsPerBlock;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();
    example_lhs::kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs, v, f, h, gamma);
}
