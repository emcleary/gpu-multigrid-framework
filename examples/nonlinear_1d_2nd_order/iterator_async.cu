#include "iterator_async.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;

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
void boundaries(ArrayRaw v, const ArrayRaw f, const double gamma,
        const BoundaryConditions bcs, const double h, const NonlinearEquation eqn,
        const ArrayRaw x, const ArrayRaw y) {

    const int n = v.size() - 1;
    
    if (bcs.is_west_dirichlet()) {
        v[0] = f[0];
    } else { // neumann
        const double un = eqn.neumann_bc_west(x.front());
        double num = h * h * f[0] + 2 * v[1];
        double denom = 2 + gamma * un * h * h;
        v[0] = num / denom;
    }

    if (bcs.is_east_dirichlet()) {
        v[n] = f[n];
    } else { // neumann
        const double un = eqn.neumann_bc_east(x.back());
        double num = h * h * f[n] + 2 * v[n-1];
        double denom = 2 + gamma * un * h * h;
        v[n] = num / denom;
    }
}
} // namespace async


void IteratorAsync::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const uint n = v.size() / 2;
    const uint threads = std::min(m_max_threads_per_block, n);
    const uint blocks = (n + threads - 1) / threads;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();
    const Array& x = grid.get_x();
    const Array& y = grid.get_y();

    // Red-Black Gauss Seidel
    async::kernel<<<blocks, threads>>>(v, f, h, gamma);
    async::boundaries<<<1, 1>>>(v, f, gamma, bcs, h, *m_eqn, x, y);
}
