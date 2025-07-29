#include "lhs_cpu.hpp"

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
    const int nh = x.size() - 1;
    assert(nh == grid.get_y().size() - 1);

    // TODO: set as class argument
    const int n_threads = 4;
    omp_set_num_threads(n_threads);

    // NB: slower with collapse(2)
#pragma omp parallel for
    for (int i = 1; i < nh; ++i)
        for (int j = 1; j < nh; ++j)
            lhs(i, j) = (4 * v(i, j) - v(i-1, j) - v(i+1, j) - v(i, j-1) - v(i, j+1)) / h2;

    if (bcs.is_periodic_y()) {
#pragma omp parallel for
        for (int i = 1; i < nh; ++i) {
            lhs(i, 0) = (4 * v(i, 0) - v(i-1, 0) - v(i+1, 0) - v(i, 1) - v(i, nh-1)) / h2;
            lhs(i, nh) = lhs(i, 0);
        }
    }

    if (bcs.is_periodic_x()) {
#pragma omp parallel for
        for (int j = 1; j < nh; ++j) {
            lhs(0, j) = (4 * v(0, j) - v(0, j-1) - v(0, j+1) - v(1, j) - v(nh-1, j)) / h2;
            lhs(nh, j) = lhs(0, j);
        }
    }

    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        lhs(0, 0) = (4 * v(0, 0) - v(1, 0) - v(nh-1, 0) - v(0, 1) - v(0, nh-1)) / h2;
        lhs(nh, 0) = lhs(0, 0);
        lhs(0, nh) = lhs(0, 0);
        lhs(nh, nh) = lhs(0, 0);
    }

    if (bcs.is_north_neumann()) {
        if (bcs.is_periodic_x()) {
            lhs(0, nh) = (4 * v(0, nh) - 2 * v(0, nh-1) - v(1, nh) - v(nh-1, nh)) / h2;
            lhs(nh, nh) = lhs(0, nh);
        }

#pragma omp parallel for
        for (int i = 1; i < nh; ++i)
            lhs(i, nh) = (4 * v(i, nh) - 2 * v(i, nh-1) - v(i+1, nh) - v(i-1, nh)) / h2;
    }

    if (bcs.is_south_neumann()) {
        if (bcs.is_periodic_x()) {
            lhs(0, 0) = (4 * v(0, 0) - 2 * v(0, 1) - v(1, 0) - v(nh-1, 0)) / h2;
            lhs(nh, 0) = lhs(0, 0);
        }

#pragma omp parallel for
        for (int i = 1; i < nh; ++i)
            lhs(i, 0) = (4 * v(i, 0) - 2 * v(i, 1) - v(i+1, 0) - v(i-1, 0)) / h2;
    }

    if (bcs.is_east_neumann()) {
        if (bcs.is_periodic_y()) {
            lhs(nh, 0) = (4 * v(nh, 0) - 2 * v(nh-1, 0) - v(nh, 1) - v(nh, nh-1)) / h2;
            lhs(nh, nh) = lhs(nh, 0);
        }

#pragma omp parallel for
        for (int j = 1; j < nh; ++j)
            lhs(nh, j) = (4 * v(nh, j) - 2 * v(nh-1, j) - v(nh, j+1) - v(nh, j-1)) / h2;
    }

    if (bcs.is_west_neumann()) {
        if (bcs.is_periodic_y()) {
            lhs(0, 0) = (4 * v(0, 0) - 2 * v(1, 0) - v(0, 1) - v(0, nh-1)) / h2;
            lhs(0, nh) = lhs(0, 0);
        }

#pragma omp parallel for
        for (int j = 1; j < nh; ++j)
            lhs(0, j) = (4 * v(0, j) - 2 * v(1, j) - v(0, j+1) - v(0, j-1)) / h2;
    }

    if (bcs.is_east_neumann() && bcs.is_north_neumann())
        lhs(nh, nh) = (4 * v(nh, nh) - 2 * v(nh-1, nh) - 2 * v(nh, nh-1)) / h2;

    if (bcs.is_east_neumann() && bcs.is_south_neumann())
        lhs(nh, 0) = (4 * v(nh, 0) - 2 * v(nh-1, 0) - 2 * v(nh, 1)) / h2;

    if (bcs.is_west_neumann() && bcs.is_south_neumann())
        lhs(0, 0) = (4 * v(0, 0) - 2 * v(1, 0) - 2 * v(0, 1)) / h2;

    if (bcs.is_west_neumann() && bcs.is_north_neumann())
        lhs(0, nh) = (4 * v(0, nh) - 2 * v(1, nh) - 2 * v(0, nh-1)) / h2;

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
