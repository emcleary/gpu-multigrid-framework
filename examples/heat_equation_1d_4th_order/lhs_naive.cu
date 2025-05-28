#include "lhs_naive.cuh"

#include <cuda_runtime.h>

#include "lhs_schemes.cuh"
#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace lhs_naive {
__global__
void kernel3(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {
    if (threadIdx.x == 1) {
        lhs[threadIdx.x] = eval2(v, h2, threadIdx.x);
        
        // set lhs to rhs so residuals are zero (dirichlet BCs)
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}

__global__
void kernel5(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {

    if (threadIdx.x == 1)
        lhs[threadIdx.x] = eval2(v, h2, threadIdx.x);
    else if (threadIdx.x == 2)
        lhs[threadIdx.x] = eval4(v, h2, threadIdx.x);
    else if (threadIdx.x == 3)
        lhs[threadIdx.x] = eval2(v, h2, threadIdx.x);
    else if (threadIdx.x == 0) {
        // set lhs to rhs so residuals are zero (dirichlet BCs)
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}

__global__
void kernel(ArrayRaw lhs, ArrayRaw v, const ArrayRaw f, const double h2) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (1 < idx && idx < v.size() - 2)
        lhs[idx] = eval4(v, h2, idx);
    else if (idx == 1)
        lhs[idx] = eval4left(v, h2, idx);
    else if (idx == v.size() - 2)
        lhs[idx] = eval4right(v, h2, idx);

    // set lhs to rhs so residuals are zero
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        lhs.front() = f.front();
        lhs.back() = f.back();
    }
}
}

void LHSNaive::run_device(Array& lhs, Array& v, const Array& f, const Grid& grid) {
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    if (v.size() == 3) {
        lhs_naive::kernel3<<<1, 3>>>(lhs, v, f, h2);
    } else if (v.size() == 5) {
        lhs_naive::kernel5<<<1, 5>>>(lhs, v, f, h2);
    } else {
        const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() - 1);
        const int blocksPerGrid = (v.size() - 1 + threadsPerBlock - 1) / threadsPerBlock;
        lhs_naive::kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs, v, f, h2);
    }
}
