#include "iterator_naive.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace naive {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2, const int color) {
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < v.size() - 1)
        v[idx] = eval_gs(v, f, h2, idx);
}
}

void IteratorNaive::run_device(Array& v, const Array& f, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() / 2);
    const int blocksPerGrid = (v.size() / 2 + threadsPerBlock - 1) / threadsPerBlock;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();

    // Red-Black Gauss Seidel -- parallel and converges faster
    naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h2, 1);
    naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h2, 2);
}

// For linear problems, solving for solution and solving for error are the exact same.
void IteratorNaive::run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_device(e, r, grid);
}
