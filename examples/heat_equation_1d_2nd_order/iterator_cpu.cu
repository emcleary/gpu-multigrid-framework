#include "iterator_cpu.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::Grid;


void IteratorCPU::run_host(Array& v, const Array& f, const Grid& grid) {
    const int n_pts = v.size();
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    // Red-Black Gauss Seidel -- parallel and converges faster
    for (int i = 1; i < n_pts - 1; i += 2)
        v[i] = eval_gs(v, f, h2, i);
    for (int i = 2; i < n_pts - 1; i += 2)
        v[i] = eval_gs(v, f, h2, i);
}

// For linear problems, solving for solution and solving for error are the exact same.
void IteratorCPU::run_error_host(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_host(e, r, grid);
}
