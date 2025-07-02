#include "iterator_cpu.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::Grid;
using gmf::modules::BoundaryConditions;


void IteratorCPU::run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const int n = v.size() - 1;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    // Red-Black Gauss Seidel -- parallel and converges faster
    for (int i = 1; i < n; i += 2)
        v[i] = eval_gs(v, f, h2, i);
    for (int i = 2; i < n; i += 2)
        v[i] = eval_gs(v, f, h2, i);

    if (bcs.is_periodic()) {
        v[0] = (v[n-1] + v[1] + f[0] * h2) / 2;
        v[n] = v[0];
    } else {
        if (bcs.is_left_dirichlet()) {
            v.front() = f.front();
        } else { // neumann
            v[0] = v[1] + f[0] * h2 / 2;
        }

        if (bcs.is_right_dirichlet()) {
            v.back() = f.back();
        } else { // neumann
            v[n] = v[n-1] + f[n] * h2 / 2;
        }
    }
}
