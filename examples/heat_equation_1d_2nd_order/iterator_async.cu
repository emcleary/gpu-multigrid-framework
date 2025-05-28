#include "iterator_async.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace async {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2) {
    static const int n_colors = 2;
    int index = n_colors * (threadIdx.x + blockDim.x * blockIdx.x);
    const int stride = n_colors * blockDim.x * gridDim.x;

    for (int color = 1; color <= n_colors; ++color) {
        int idx = index + color;
        while (idx < v.size()) {
            if (0 < idx && idx < v.size() - 1)
                v[idx] = eval_gs(v, f, h2, idx);
            idx += stride;
        }
        __syncthreads();
    }
}
} // namespace async


void IteratorAsync::run_device(Array& v, const Array& f, const Grid& grid) {
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    size_t i = 1 << (int)std::ceil(std::log2(v.size() / 2));
    const int threadsPerBlock = std::min(m_max_threads_per_block, i);
    const int blocksPerGrid = (i + threadsPerBlock - 1) / threadsPerBlock;
    async::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h2);
}

// For linear problems, solving for solution and solving for error are the exact same.
void IteratorAsync::run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_device(e, r, grid);
}
