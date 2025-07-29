#include "lhs_cpu.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


void LHSCPU::run_host(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int nh = v.size() - 1;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    omp_set_num_threads(m_omp_threads);

#pragma omp parallel for
    for (int i = 1; i < nh; i += 1)
        lhs[i] = - (v[i-1] - 2 * v[i] +  v[i+1]) / h2;

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
