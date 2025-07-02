#include "lhs.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;
using gmf::modules::BoundaryConditions;


__host__ __device__
inline double eval(ArrayRaw& v, const double h, const double gamma, const int i) {
    double lhs = (-v[i-1] + 2 * v[i] - v[i+1]) / h;
    lhs += gamma * v[i] * (v[i+1] - v[i-1]) / 2;
    return lhs / h;
}

namespace example_lhs {
__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const double h, const double gamma, const BoundaryConditions bcs) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int n = v.size() - 1;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < n)
        lhs[idx] = eval(v, h, gamma, idx);

    if (idx == 0 && blockIdx.x == 0) {
        if (bcs.is_left_dirichlet()) {
            lhs[0] = v[0];
        } else { // neumann
            lhs[0] = (2/h/h + gamma * bcs.get_left()) * v[0] - 2/h/h * v[1];
        }

        if (bcs.is_right_dirichlet()) {
            lhs[n] = v[n];
        } else { // neumann
            lhs[n] = (2/h/h + gamma * bcs.get_right()) * v[n] - 2/h/h * v[n-1];
        }
    }
}
}

void NonlinearLHS::run_host(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int n = v.size() - 1;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    for (int i = 1; i < n; i += 1)
        lhs[i] = eval(v, h, gamma, i);

    if (bcs.is_left_dirichlet()) {
        lhs[0] = v[0];
    } else { // neumann
        lhs[0] = (2/h/h + gamma * bcs.get_left()) * v[0] - 2/h/h * v[1];
    }

    if (bcs.is_right_dirichlet()) {
        lhs[n] = v[n];
    } else { // neumann
        lhs[n] = (2/h/h + gamma * bcs.get_right()) * v[n] - 2/h/h * v[n-1];
    }
}

void NonlinearLHS::run_device(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size());
    const int blocksPerGrid = (v.size() + threadsPerBlock - 1) / threadsPerBlock;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();
    example_lhs::kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs, v, h, gamma, bcs);
}
