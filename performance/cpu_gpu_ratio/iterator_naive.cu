#include "iterator_naive.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;
using gmf::modules::BoundaryConditions;


namespace naive {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2, const int color) {
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;
    if (0 < idx && idx < v.size() - 1)
        v[idx] = eval_gs(v, f, h2, idx);
}

__global__
void boundaries(ArrayRaw v, const ArrayRaw f, const BoundaryConditions bcs, const double h2) {
    const int n = v.size() - 1;
    
    if (bcs.is_periodic()) {
        v[0] = (v[n-1] + v[1] + f[0] * h2) / 2;
        v[n] = v[0];
    } else {
        if (bcs.is_left_dirichlet()) {
            v[0] = f[0];
        } else { // neumann
            v[0] = v[1] + f[0] * h2 / 2;
        }

        if (bcs.is_right_dirichlet()) {
            v[n] = f[n];
        } else { // neumann
            v[n] = v[n-1] + f[n] * h2 / 2;
        }
    }
}
} // namespace naive

void IteratorNaive::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() / 2);
    const int blocksPerGrid = (v.size() / 2 + threadsPerBlock - 1) / threadsPerBlock;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    // Red-Black Gauss Seidel -- parallel and converges faster
    naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h2, 1);
    naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h2, 2);
    naive::boundaries<<<1, 1>>>(v, f, bcs, h2);
}
