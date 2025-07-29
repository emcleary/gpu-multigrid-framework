#include "lhs_naive.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


__host__ __device__
inline double eval(ArrayRaw& v, const double h, const double gamma, const int i) {
    double lhs = (-v[i-1] + 2 * v[i] - v[i+1]) / h;
    lhs += gamma * v[i] * (v[i+1] - v[i-1]) / 2;
    return lhs / h;
}

namespace example_lhs {
__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const double h, const double gamma) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int n = v.size() - 1;

    // must skip boundaries for Dirichlet BCs
    if (0 < i && i < n) {
        double tmp = (-v[i-1] + 2 * v[i] - v[i+1]) / h;
        tmp += gamma * v[i] * (v[i+1] - v[i-1]) / 2;
        lhs[i] = tmp / h;
    }
}


__global__
void boundaries(ArrayRaw lhs, ArrayRaw v, const double h, const double gamma,
        const BoundaryConditions bcs, const NonlinearEquation eqn, const ArrayRaw x) {
    const int n = v.size() - 1;

    if (bcs.is_west_dirichlet()) {
        lhs[0] = v[0];
    } else { // neumann
        const double un = eqn.neumann_bc_west(x.front());
        lhs[0] = (2/h/h + gamma * un) * v[0] - 2/h/h * v[1];
    }

    if (bcs.is_east_dirichlet()) {
        lhs[n] = v[n];
    } else { // neumann
        const double un = eqn.neumann_bc_east(x.back());
        lhs[n] = (2/h/h + gamma * un) * v[n] - 2/h/h * v[n-1];
    }
}
}


void NonlinearLHS::run_device(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const uint threads = std::min(m_max_threads_per_block, static_cast<uint>(v.size()));
    const uint blocks = (v.size() + threads - 1) / threads;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();
    const Array& x = grid.get_x();

    example_lhs::kernel<<<blocks, threads>>>(lhs, v, h, gamma);
    example_lhs::boundaries<<<1, 1>>>(lhs, v, h, gamma, bcs, *m_eqn, x);
}
