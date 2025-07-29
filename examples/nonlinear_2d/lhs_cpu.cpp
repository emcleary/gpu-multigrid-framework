#include "lhs_cpu.hpp"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


void LHSCPU::run_host(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const double h = grid.get_cell_width();
    assert(h == grid.get_cell_height());
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    const pmf::Array& x = grid.get_x();
    const pmf::Array& y = grid.get_y();
    const int nh = v.get_nrows() - 1;

    omp_set_num_threads(m_omp_threads);

#pragma omp parallel for
    for (int i = 1; i < nh; ++i)
        for (int j = 1; j < nh; ++j) {
            lhs(i, j) = (4 * v(i, j) - v(i-1, j) - v(i+1, j) - v(i, j-1) - v(i, j+1)) / h2
                + m_eqn->get_gamma() * v(i, j) * (v(i+1, j) - v(i-1, j)) / 2 / h;
        }

    if (bcs.is_west_neumann()) {
#pragma omp parallel for
        for (int j = 1; j < nh; ++j) {
            double un = m_eqn->neumann_bc_west(x[0], y[j]);
            lhs(0, j) = (4 * v(0, j) - 2 * v(1, j) - v(0, j+1) - v(0, j-1)) / h2
                + m_eqn->get_gamma() * v(0, j) * un;
        }
    }

    if (bcs.is_north_dirichlet())
#pragma omp parallel for
        for (int i = 0; i <= nh; ++i)
            lhs(i, nh) = v(i, nh);

    if (bcs.is_south_dirichlet())
#pragma omp parallel for
        for (int i = 0; i <= nh; ++i)
            lhs(i, 0) = v(i, 0);

    if (bcs.is_east_dirichlet())
#pragma omp parallel for
        for (int j = 0; j <= nh; ++j)
            lhs(nh, j) = v(nh, j);

    if (bcs.is_west_dirichlet())
#pragma omp parallel for
        for (int j = 0; j <= nh; ++j)
            lhs(0, j) = v(0, j);
}
