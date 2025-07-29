#include "iterator_cpu.hpp"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


void IteratorCPU::run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const double h = grid.get_cell_width();
    assert(h == grid.get_cell_height());
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    const pmf::Array& x = grid.get_x();
    const pmf::Array& y = grid.get_y();
    const int nx = x.size() - 1;
    const int ny = y.size() - 1;

    omp_set_num_threads(m_omp_threads);

    // ODD colors (internal only)
#pragma omp parallel for
    for (int i = 1; i < nx; ++i) {
        // i, j must be even-odd or odd-even pair
        int j0 = i % 2 == 0 ? 1 : 2;
        for (int j = j0; j < ny; j += 2) {
            double denom = 4 + h / 2 * m_eqn->get_gamma() * (v(i+1, j) - v(i-1, j));
            v(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) + f(i, j) * h2) / denom;
        }
    }

    // ODD BCs
    if (bcs.is_west_neumann())
#pragma omp parallel for
        for (int j = 1; j < ny; j += 2) {
            double un = m_eqn->neumann_bc_west(x[0], y[j]);
            double denom = 4 + h2 * m_eqn->get_gamma() * un;
            v(0, j) = (2 * v(1, j) + v(0, j-1) + v(0, j+1) + f(0, j) * h2) / denom;
        }

    // EVEN colors (internal only)
#pragma omp parallel for
    for (int i = 1; i < nx; ++i) {
        // i, j must be even-even or odd-odd pair
        int j0 = i % 2 == 0 ? 2 : 1;
        for (int j = j0; j < ny; j += 2) {
            double denom = 4 + h / 2 * m_eqn->get_gamma() * (v(i+1, j) - v(i-1, j));
            v(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) + f(i, j) * h2) / denom;
        }
    }

    // EVEN BCs
    if (bcs.is_west_neumann()) {
#pragma omp parallel for
        for (int j = 2; j < ny; j += 2) {
            double un = m_eqn->neumann_bc_west(x[0], y[j]);
            double denom = 4 + h2 * m_eqn->get_gamma() * un;
            v(0, j) = (2 * v(1, j) + v(0, j-1) + v(0, j+1) + f(0, j) * h2) / denom;
        }
    }
}
