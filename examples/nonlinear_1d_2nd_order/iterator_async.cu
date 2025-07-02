#include "iterator_async.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;
using gmf::modules::BoundaryConditions;

__host__ __device__
inline double eval(const ArrayRaw& v, const ArrayRaw& f, const double h, const double gamma, const int i) {
    double num = 2 * (h * h * f[i] + v[i+1] + v[i-1]);
    double denom = 4 + h * gamma * (v[i+1] - v[i-1]);
    return num / denom;
}

namespace async {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h, const double gamma) {
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x);

    if (idx + 1 < v.size() - 1)
        v[idx+1] = eval(v, f, h, gamma, idx+1);

    if (idx > 0)
        v[idx] = eval(v, f, h, gamma, idx);
}

__global__
void boundaries(ArrayRaw v, const ArrayRaw f, const double gamma, const BoundaryConditions bcs, const double h) {
    const int n = v.size() - 1;
    
    if (bcs.is_left_dirichlet()) {
        v[0] = f[0];
    } else { // neumann
        double num = h * h * f[0] + 2 * v[1];
        double denom = 2 + gamma * bcs.get_left() * h * h;
        v[0] = num / denom;
    }

    if (bcs.is_right_dirichlet()) {
        v[n] = f[n];
    } else { // neumann
        double num = h * h * f[n] + 2 * v[n-1];
        double denom = 2 + gamma * bcs.get_right() * h * h;
        v[n] = num / denom;
    }
}
} // namespace async


void IteratorAsync::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() / 2);
    const int blocksPerGrid = (v.size() / 2 + threadsPerBlock - 1) / threadsPerBlock;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    // Red-Black Gauss Seidel
    async::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h, gamma);
    async::boundaries<<<1, 1>>>(v, f, gamma, bcs, h);
}
