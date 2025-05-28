#include "iterator_async_smem.cuh"

#include <cuda_runtime.h>

#include "schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace async_smem {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2) {
    extern __shared__ double cache[];

    int gi = threadIdx.x + 2 * blockDim.x * blockIdx.x;
    int si = threadIdx.x + 1;
    if (threadIdx.x == 0 && gi > 0)
        cache[0] = v[gi-1];
    if (0 <= gi && gi <= v.size() - 1)
        cache[si] = v[gi];
    gi += blockDim.x;
    si += blockDim.x;
    if (0 <= gi && gi <= v.size() - 1)
        cache[si] = v[gi];
    if (threadIdx.x == blockDim.x - 1 && gi < v.size() - 1)
        cache[2 + 2 * blockDim.x - 1] = v[gi + 1];
    __syncthreads();
    
    // "color #1; all even indices"
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x);
    int i = 2 * threadIdx.x + 1;
    if (0 < idx && idx < v.size() - 1)
        cache[i] =  eval_gs(cache, f[idx], h2, i);
    __syncthreads();

    // "color #2; all odd indices"
    ++i;
    ++idx;
    if (0 < idx && idx < v.size() - 1)
        cache[i] = eval_gs(cache, f[idx], h2, i);
    __syncthreads();

    if (0 < gi && gi < v.size() - 1)
        v[gi] = cache[si];
    gi -= blockDim.x;
    si -= blockDim.x;
    if (0 < gi && gi < v.size() - 1)
        v[gi] = cache[si];
}
}

void IteratorAsyncSMEM::run_device(Array& v, const Array& f, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() / 2);
    const int blocksPerGrid = (v.size() / 2 + threadsPerBlock - 1) / threadsPerBlock;
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    const int smem_size = (2 * threadsPerBlock + 2) * sizeof(double);

    // Red-Black Gauss Seidel -- parallel and converges faster
    async_smem::kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(v, f, h2);
}

// For linear problems, solving for solution and solving for error are the exact same.
void IteratorAsyncSMEM::run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_device(e, r, grid);
}
