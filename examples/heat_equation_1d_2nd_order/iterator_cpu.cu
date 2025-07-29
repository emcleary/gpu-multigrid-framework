#include "iterator_cpu.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


void IteratorCPU::run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const int n = v.size() - 1;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    omp_set_num_threads(m_omp_threads);

    // Red-Black Gauss Seidel
#pragma omp parallel for
    for (int i = 1; i < n; i += 2)
        v[i] = (v[i-1] + v[i+1] + h2 * f[i]) / 2;

#pragma omp parallel for
    for (int i = 2; i < n; i += 2)
        v[i] = (v[i-1] + v[i+1] + h2 * f[i]) / 2;

    if (bcs.is_periodic_x()) {
        v[0] = (v[n-1] + v[1] + f[0] * h2) / 2;
        v[n] = v[0];
    } else {
        if (bcs.is_west_dirichlet()) {
            v.front() = f.front();
        } else { // neumann
            v[0] = v[1] + f[0] * h2 / 2;
        }

        if (bcs.is_east_dirichlet()) {
            v.back() = f.back();
        } else { // neumann
            v[n] = v[n-1] + f[n] * h2 / 2;
        }
    }
}
