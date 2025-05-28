#include "iterator_async_smem.cuh"

#include <cuda_runtime.h>

#include "iterator_schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace async_smem {
__global__
void kernel3(ArrayRaw v, const ArrayRaw f, const double h2) {
    if (threadIdx.x == 1)
        v[threadIdx.x] = eval2(v.data(), f[threadIdx.x], h2, threadIdx.x);
}

__global__
void kernel5(ArrayRaw v, const ArrayRaw f, const double h2) {
    if (threadIdx.x == 1)
        v[threadIdx.x] = eval2(v.data(), f[threadIdx.x], h2, threadIdx.x);
    else if (threadIdx.x == 2)
        v[threadIdx.x] = eval4(v.data(), f[threadIdx.x], h2, threadIdx.x);
    else if (threadIdx.x == 3)
        v[threadIdx.x] = eval2(v.data(), f[threadIdx.x], h2, threadIdx.x);
}

__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2) {
    extern __shared__ double cache[];

    int offset = 3 * blockDim.x * blockIdx.x;
    int gi = threadIdx.x + offset;
    const int stride = 3 * blockDim.x * gridDim.x;
    
    while (gi < v.size()) {
        int si = threadIdx.x + 2;
        if (threadIdx.x == 0 && gi > 1)
            cache[0] = v[gi-2];
        if (threadIdx.x == 0 && gi > 0)
            cache[1] = v[gi-1];
        if (0 <= gi && gi <= v.size() - 1)
            cache[si] = v[gi];
        gi += blockDim.x;
        si += blockDim.x;
        if (0 <= gi && gi <= v.size() - 1)
            cache[si] = v[gi];
        gi += blockDim.x;
        si += blockDim.x;
        if (0 <= gi && gi <= v.size() - 1)
            cache[si] = v[gi];
        if (threadIdx.x == blockDim.x - 1 && gi < v.size() - 1)
            cache[3 * blockDim.x + 4 - 2] = v[gi + 1];
        if (threadIdx.x == blockDim.x - 1 && gi < v.size() - 2)
            cache[3 * blockDim.x + 4 - 1] = v[gi + 2];
        __syncthreads();
    
        int idx = 3 * threadIdx.x + offset;
        int i = 3 * threadIdx.x + 2;
        if (1 < idx && idx < v.size() - 2) {
            cache[i] = eval4(cache, f[idx], h2, i);
        } else if (idx == 1) {
            cache[i] = eval4left(cache, f[idx], h2, i);
        } else if (idx == v.size() - 2) {
            cache[i] = eval4right(cache, f[idx], h2, i);
        } // do nothing at bounds (idx=0, idx=v.size()-1)
        __syncthreads();

        ++i;
        ++idx;
        if (1 < idx && idx < v.size() - 2) {
            cache[i] = eval4(cache, f[idx], h2, i);
        } else if (idx == 1) {
            cache[i] = eval4left(cache, f[idx], h2, i);
        } else if (idx == v.size() - 2) {
            cache[i] = eval4right(cache, f[idx], h2, i);
        }
        __syncthreads();
    
        ++i;
        ++idx;
        if (1 < idx && idx < v.size() - 2) {
            cache[i] = eval4(cache, f[idx], h2, i);
        } else if (idx == 1) {
            cache[i] = eval4left(cache, f[idx], h2, i);
        } else if (idx == v.size() - 2) {
            cache[i] = eval4right(cache, f[idx], h2, i);
        }
        __syncthreads();
    
        if (0 < gi && gi < v.size() - 1)
            v[gi] = cache[si];
        gi -= blockDim.x;
        si -= blockDim.x;
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
}

void IteratorAsyncSMEM::run_device(Array& v, const Array& f, const Grid& grid) {
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    if (v.size() == 3) {
        async_smem::kernel3<<<1, 3>>>(v, f, h2);
    } else if (v.size() == 5) {
        async_smem::kernel5<<<1, 5>>>(v, f, h2);
    } else {
        size_t i = 1 << (int)std::ceil(std::log2(v.size() / 3));
        const int threadsPerBlock = std::min(m_max_threads_per_block, i);
        const int blocksPerGrid = (i + threadsPerBlock - 1) / threadsPerBlock;
        const int smem_size = (3 * threadsPerBlock + 4) * sizeof(double);
        async_smem::kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(v, f, h2);
    }
}

void IteratorAsyncSMEM::run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_device(e, r, grid);
}
