#include "lhs_smem.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "lhs_schemes.cuh"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace lhs_smem {
__global__
void kernel3(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {
    if (threadIdx.x == 1) {
        lhs[threadIdx.x] = eval2(v.data(), h2, threadIdx.x);
        
        // set lhs to rhs so residuals are zero (dirichlet BCs)
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}

__global__
void kernel5(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {

    if (threadIdx.x == 1)
        lhs[threadIdx.x] = eval2(v.data(), h2, threadIdx.x);
    else if (threadIdx.x == 2)
        lhs[threadIdx.x] = eval4(v.data(), h2, threadIdx.x);
    else if (threadIdx.x == 3)
        lhs[threadIdx.x] = eval2(v.data(), h2, threadIdx.x);
    else if (threadIdx.x == 0) {
        // set lhs to rhs so residuals are zero (dirichlet BCs)
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}

__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {
    extern __shared__ double cache[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int si = threadIdx.x + 2;
    if (idx < v.size())
        cache[si] = v[idx];
    if (threadIdx.x == 0) {
        if (idx > 1) cache[0] = v[idx - 2];
        if (idx > 0) cache[1] = v[idx - 1];
    }
    if (threadIdx.x == blockDim.x - 1) {
        if (idx < v.size() - 1) cache[blockDim.x + 2] = v[idx + 1];
        if (idx < v.size() - 2) cache[blockDim.x + 3] = v[idx + 2];
    }
    __syncthreads();

    if (1 < idx && idx < v.size() - 2)
        lhs[idx] = eval4(cache, h2, si);
    else if (idx == 1)
        lhs[idx] = eval4left(cache, h2, si);
    else if (idx == v.size() - 2)
        lhs[idx] = eval4right(cache, h2, si);

    // set lhs to rhs so residuals are zero
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}
}

void LHSSMEM::run_device(Array& lhs, Array& v, const Array& f, const Grid& grid) {
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    if (v.size() == 3) {
        lhs_smem::kernel3<<<1, 3>>>(lhs, v, f, h2);
    } else if (v.size() == 5) {
        lhs_smem::kernel5<<<1, 5>>>(lhs, v, f, h2);
    } else {
        const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() - 1);
        const int blocksPerGrid = (v.size() - 1 + threadsPerBlock - 1) / threadsPerBlock;
        const int smem_size = (4 + threadsPerBlock) * sizeof(double);
        lhs_smem::kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(lhs, v, f, h2);
    }
}
