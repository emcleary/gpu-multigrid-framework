#include "iterator_cpu.cuh"

#include <cuda_runtime.h>

#include "iterator_schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::Grid;
using gmf::modules::BoundaryConditions;


void IteratorCPU::run_host(Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    const int n_pts = v.size();
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    if (v.size() > 5) {
        v[1] = eval4left(v, f, h2, 1);
        for (int i = 4; i < n_pts - 2; i += 3)
            v[i] = eval4(v, f, h2, i);
        for (int i = 2; i < n_pts - 2; i += 3)
            v[i] = eval4(v, f, h2, i);
        for (int i = 3; i < n_pts - 2; i += 3)
            v[i] = eval4(v, f, h2, i);
        v[n_pts-2] = eval4right(v, f, h2, n_pts-2);
    } else if (v.size() == 5) {
        v[1] = eval2(v, f, h2, 1);
        v[2] = eval4(v, f, h2, 2);
        v[3] = eval2(v, f, h2, 3);
    } else {
        assert(v.size() == 3);
        v[1] = eval2(v, f, h2, 1);
    }
}

void IteratorCPU::run_error_host(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_host(e, r, grid);
}
