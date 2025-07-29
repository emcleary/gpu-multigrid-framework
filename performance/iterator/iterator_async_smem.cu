#include "iterator_async_smem.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using pmf::Array;
using pmf::ArrayRaw;
using pmf::Grid;
using pmf::modules::BoundaryConditions;


namespace async_smem {

__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2) {
    extern __shared__ double cache[];

    int offset = 2 * blockDim.x * blockIdx.x;
    int gi = threadIdx.x + offset;
    const int stride = 2 * blockDim.x * gridDim.x;
    
    while (gi < v.size()) {
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
            cache[2 * blockDim.x + 2 - 1] = v[gi + 1];
        __syncthreads();
    
        int idx = 2 * threadIdx.x + offset;
        int i = 2 * threadIdx.x + 1;
        if (1 < idx && idx < v.size() - 1) {
            // cache[i] = eval_gs(cache, f[idx], h2, i);
            cache[i] = (cache[i-1] + cache[i+1] + h2 * f[idx]) / 2;
        } // do nothing at bounds (idx=0, idx=v.size()-1)
        __syncthreads();

        ++i;
        ++idx;
        if (1 < idx && idx < v.size() - 1) {
            // cache[i] = eval_gs(cache, f[idx], h2, i);
            cache[i] = (cache[i-1] + cache[i+1] + h2 * f[idx]) / 2;
        } // do nothing at bounds (idx=0, idx=v.size()-1)
        __syncthreads();
    
        if (0 < gi && gi < v.size() - 1)
            v[gi] = cache[si];
        gi -= blockDim.x;
        si -= blockDim.x;
        if (0 < gi && gi < v.size() - 1)
            v[gi] = cache[si];

        gi += stride;
        offset += stride;
    }
}


__global__
void boundaries(ArrayRaw v, const ArrayRaw f, const BoundaryConditions bcs, const double h2) {
    const int n = v.size() - 1;
    
    if (bcs.is_periodic_x()) {
        v[0] = (v[n-1] + v[1] + f[0] * h2) / 2;
        v[n] = v[0];
    } else {
        if (bcs.is_west_dirichlet()) {
            v[0] = f[0];
        } else { // neumann
            v[0] = v[1] + f[0] * h2 / 2;
        }

        if (bcs.is_east_dirichlet()) {
            v[n] = f[n];
        } else { // neumann
            v[n] = v[n-1] + f[n] * h2 / 2;
        }
    }
}
}

void IteratorAsyncSMEM::run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) {
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    uint i = 1 << (int)std::ceil(std::log2(v.size() / 2));
    const int threads = std::min(m_max_threads_per_block, i);
    const int blocks = (i + threads - 1) / threads;
    const int smem_size = (2 * threads + 2) * sizeof(double);
    async_smem::kernel<<<blocks, threads, smem_size>>>(v, f, h2);
    async_smem::boundaries<<<1, 1>>>(v, f, bcs, h2);
}
