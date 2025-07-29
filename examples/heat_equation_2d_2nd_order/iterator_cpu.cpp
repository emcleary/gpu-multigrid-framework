#include "iterator_cpu.hpp"

#include <omp.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


void IteratorCPU::run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    // omp_set_num_threads(8);

    const double h = grid.get_cell_width();
    assert(h == grid.get_cell_height());
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    const pmf::Array& x = grid.get_x();
    const pmf::Array& y = grid.get_y();
    const int nx = x.size() - 1;
    const int ny = y.size() - 1;

    // TODO: set as class argument
    const int n_threads = 4;
    omp_set_num_threads(n_threads);

    // ODD colors (internal only)
#pragma omp parallel for
    for (int i = 1; i < nx; ++i) {
        // i, j must be even-odd or odd-even pair
        int j0 = i % 2 == 0 ? 1 : 2;
        for (int j = j0; j < ny; j += 2)
            v(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) + f(i, j) * h2) / 4;
    }

    // ODD BCs
    if (bcs.is_periodic_y()) {
#pragma omp parallel for
        for (int i = 1; i < nx; i += 2) {
            v(i, 0) = (v(i-1, 0) + v(i+1, 0) + v(i, ny-1) + v(i, 1) + f(i, 0) * h2) / 4;
            v(i, ny) = v(i, 0);
        }
    }

    if (bcs.is_periodic_x()) {
#pragma omp parallel for
        for (int j = 1; j < ny; j += 2) {
            v(0, j) = (v(nx-1, j) + v(1, j) + v(0, j-1) + v(0, j+1) + f(0, j) * h2) / 4;
            v(nx, j) = v(0, j);
        }
    }

    if (bcs.is_north_neumann())
#pragma omp parallel for
        for (int i = 1; i < nx; i += 2)
            v(i, ny) = (2 * v(i, ny-1) + v(i-1, ny) + v(i+1, ny) + f(i, ny) * h2) / 4;

    if (bcs.is_south_neumann())
#pragma omp parallel for
        for (int i = 1; i < nx; i += 2)
            v(i, 0) = (2 * v(i, 1) + v(i-1, 0) + v(i+1, 0) + f(i, 0) * h2) / 4;

    if (bcs.is_east_neumann())
#pragma omp parallel for
        for (int j = 1; j < ny; j += 2)
            v(nx, j) = (2 * v(nx-1, j) + v(nx, j-1) + v(nx, j+1) + f(nx, j) * h2) / 4;

    if (bcs.is_west_neumann())
#pragma omp parallel for
        for (int j = 1; j < ny; j += 2)
            v(0, j) = (2 * v(1, j) + v(0, j-1) + v(0, j+1) + f(0, j) * h2) / 4;

    // EVEN colors (internal only)
#pragma omp parallel for
    for (int i = 1; i < nx; ++i) {
        // i, j must be even-even or odd-odd pair
        int j0 = i % 2 == 0 ? 2 : 1;
        for (int j = j0; j < ny; j += 2)
            v(i, j) = (v(i-1, j) + v(i+1, j) + v(i, j-1) + v(i, j+1) + f(i, j) * h2) / 4;
    }

    // EVEN BCs
    if (bcs.is_periodic_y()) {
#pragma omp parallel for
        for (int i = 2; i < nx; i += 2) {
            v(i, 0) = (v(i-1, 0) + v(i+1, 0) + v(i, ny-1) + v(i, 1) + f(i, 0) * h2) / 4;
            v(i, ny) = v(i, 0);
        }
    }

    if (bcs.is_periodic_x()) {
#pragma omp parallel for
        for (int j = 2; j < ny; j += 2) {
            v(0, j) = (v(nx-1, j) + v(1, j) + v(0, j-1) + v(0, j+1) + f(0, j) * h2) / 4;
            v(nx, j) = v(0, j);
        }
    }

    if (bcs.is_north_neumann()) {
        if (bcs.is_periodic_x()) {
            v(0, ny) = (2 * v(0, ny-1) + v(nx-1, ny) + v(1, ny) + f(0, ny) * h2) / 4;
            v(nx, ny) = v(0, ny);
        }

#pragma omp parallel for
        for (int i = 2; i < nx; i += 2)
            v(i, ny) = (2 * v(i, ny-1) + v(i-1, ny) + v(i+1, ny) + f(i, ny) * h2) / 4;
    }

    if (bcs.is_south_neumann()) {
        if (bcs.is_periodic_x()) {
            v(0, 0) = (2 * v(0, 1) + v(nx-1, 0) + v(1, 0) + f(0, 0) * h2) / 4;
            v(nx, 0) = v(0, 0);
        }

#pragma omp parallel for
        for (int i = 2; i < nx; i += 2)
            v(i, 0) = (2 * v(i, 1) + v(i-1, 0) + v(i+1, 0) + f(i, 0) * h2) / 4;
    }

    if (bcs.is_east_neumann()) {
        if (bcs.is_periodic_y()) {
            v(nx, 0) = (2 * v(nx-1, 0) + v(nx, ny-1) + v(nx, 1) + f(nx, 0) * h2) / 4;
            v(nx, ny) = v(nx, 0);
        }

#pragma omp parallel for
        for (int j = 2; j < ny; j += 2)
            v(nx, j) = (2 * v(nx-1, j) + v(nx, j-1) + v(nx, j+1) + f(nx, j) * h2) / 4;
    }

    if (bcs.is_west_neumann()) {
        if (bcs.is_periodic_y()) {
            v(0, 0) = (2 * v(1, 0) + v(0, ny-1) + v(0, 1) + f(0, 0) * h2) / 4;
            v(0, ny) = v(0, 0);
        }

#pragma omp parallel for
        for (int j = 2; j < ny; j += 2)
            v(0, j) = (2 * v(1, j) + v(0, j-1) + v(0, j+1) + f(0, j) * h2) / 4;
    }

    // CORNERS (always EVEN)
    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        v(0, 0) = (v(nx-1, 0) + v(1, 0) + v(0, ny-1) + v(0, 1) + f(0, 0) * h2) / 4;
        v(nx, 0) = v(0, 0);
        v(0, ny) = v(0, 0);
        v(nx, ny) = v(0, 0);
    }

    if (bcs.is_east_neumann() && bcs.is_north_neumann())
        v(nx, ny) = (2 * v(nx-1, ny) + 2 * v(nx, ny-1) + f(nx, ny) * h2) / 4;

    if (bcs.is_east_neumann() && bcs.is_south_neumann())
        v(nx, 0) = (2 * v(nx-1, 0) + 2 * v(nx, 1) + f(nx, 0) * h2) / 4;

    if (bcs.is_west_neumann() && bcs.is_south_neumann())
        v(0, 0) = (2 * v(1, 0) + 2 * v(0, 1) + f(0, 0) * h2) / 4;

    if (bcs.is_west_neumann() && bcs.is_north_neumann())
        v(0, ny) = (2 * v(1, ny) + 2 * v(0, ny-1) + f(0, ny) * h2) / 4;
}
