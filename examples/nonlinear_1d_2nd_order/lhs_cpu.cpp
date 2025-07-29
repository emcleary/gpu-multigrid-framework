#include "lhs_cpu.hpp"

#include <omp.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


void NonlinearLHSCPU::run_host(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int n = v.size() - 1;
    const double h = grid.get_cell_width();
    const Array& x = grid.get_x();
    const double gamma = m_eqn->get_gamma();

    omp_set_num_threads(m_omp_threads);

#pragma omp parallel for
    for (int i = 1; i < n; i += 1) {
        double tmp =  (-v[i-1] + 2 * v[i] - v[i+1]) / h;
        tmp += gamma * v[i] * (v[i+1] - v[i-1]) / 2;
        lhs[i] = tmp / h;
    }

    if (bcs.is_west_dirichlet()) {
        lhs[0] = v[0];
    } else { // neumann
        lhs[0] = (2/h/h + gamma * m_eqn->neumann_bc_west(x.front())) * v[0] - 2/h/h * v[1];
    }

    if (bcs.is_east_dirichlet()) {
        lhs[n] = v[n];
    } else { // neumann
        lhs[n] = (2/h/h + gamma * m_eqn->neumann_bc_east(x.back())) * v[n] - 2/h/h * v[n-1];
    }
}
