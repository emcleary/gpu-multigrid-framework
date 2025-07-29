#include "iterator.cuh"

#include <cuda_runtime.h>
#include <omp.h>

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

namespace iterator {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h, const double gamma, const int color) {
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < v.size() - 1)
        v[idx] = eval(v, f, h, gamma, idx);
}

__global__
void boundaries(ArrayRaw v, const ArrayRaw f, const double h, const double gamma,
        const BoundaryConditions bcs, const NonlinearEquation eqn, const ArrayRaw x) {
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
} // namespace iterator


void IteratorNonlinear::run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const int n = v.size() - 1;
    const double h = grid.get_cell_width();
    const Array& x = grid.get_x();
    const double gamma = m_eqn->get_gamma();

    omp_set_num_threads(m_omp_threads);

    // Red-Black Gauss Seidel
#pragma omp parallel for
    for (int i = 1; i < n; i += 2)
        v[i] = eval(v, f, h, gamma, i);

#pragma omp parallel for
    for (int i = 2; i < n; i += 2)
        v[i] = eval(v, f, h, gamma, i);

    if (bcs.is_west_dirichlet()) {
        v[0] = f[0];
    } else { // neumann
        double num = h * h * f[0] + 2 * v[1];
        double denom = 2 + m_eqn->get_gamma() * m_eqn->neumann_bc_west(x.front()) * h * h;
        v[0] = num / denom;
    }

    if (bcs.is_east_dirichlet()) {
        v[n] = f[n];
    } else { // neumann
        double num = h * h * f[n] + 2 * v[n-1];
        double denom = 2 + m_eqn->get_gamma() * m_eqn->neumann_bc_east(x.back()) * h * h;
        v[n] = num / denom;
    }
}

void IteratorNonlinear::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const uint n = v.size() / 2;
    const uint threads = std::min(m_max_threads_per_block, n);
    const uint blocks = (n + threads - 1) / threads;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();
    const Array& x = grid.get_x();

    // Red-Black Gauss Seidel
    iterator::kernel<<<blocks, threads>>>(v, f, h, gamma, 1);
    iterator::kernel<<<blocks, threads>>>(v, f, h, gamma, 2);
    iterator::boundaries<<<1, 1>>>(v, f, h, gamma, bcs, *m_eqn, x);
}
