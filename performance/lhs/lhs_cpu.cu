#include "lhs_cpu.cuh"

#include <cuda_runtime.h>

#include "lhs_schemes.cuh"
#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::Grid;
using gmf::modules::BoundaryConditions;


void LHSCPU::run_host(Array& lhs, Array& v, const BoundaryConditions& bcs, const Grid& grid) {
    // const int n_pts = v.size();
    // const double h2 = grid.get_cell_width() * grid.get_cell_width();

    // if (v.size() > 5) {
    //     lhs[1] = eval4left(v, h2, 1);
    //     for (int i = 2; i < n_pts - 2; i += 1)
    //         lhs[i] = eval4(v, h2, i);
    //     lhs[n_pts-2] = eval4right(v, h2, n_pts-2);
    // } else if (v.size() == 5) {
    //     lhs[1] = eval2(v, h2, 1);
    //     lhs[2] = eval4(v, h2, 2);
    //     lhs[3] = eval2(v, h2, 3);
    // } else {
    //     assert(v.size() == 3);
    //     lhs[1] = eval2(v, h2, 1);
    // }

    // // dirichlet BCs
    // lhs.front() = f.front();
    // lhs.back() = f.back();
}
