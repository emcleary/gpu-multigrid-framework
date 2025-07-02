#include "interpolator_linear.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

namespace gmf {
namespace modules {

__host__ __device__
inline double eval_even(const ArrayRaw& coarse, const int i) {
    return coarse[i];
}

__host__ __device__
inline double eval_odd(const ArrayRaw& coarse, const int i) {
    return (coarse[i] + coarse[i + 1]) / 2;
}

namespace interpolator_linear {
__global__
void kernel(ArrayRaw coarse, ArrayRaw fine, BoundaryConditions bcs) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < coarse.size() - 1) {
        fine[2*idx] = eval_even(coarse, idx);
        fine[2*idx + 1] = eval_odd(coarse, idx);
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (bcs.is_periodic()) {
            const int nc = coarse.size() - 1;
            const int nf = fine.size() - 1;
            fine[nf-1] = (coarse[nc-1] + coarse[0]) / 2;
        }

        fine.front() = coarse.front();
        fine.back() = coarse.back();
    }
}
}


void InterpolatorLinear::run_host(Array& coarse, Array& fine, BoundaryConditions& bcs) {
    for (int i = 0; i < coarse.size() - 1; ++i) {
        fine[2*i] = eval_even(coarse, i);
        fine[2*i+1] = eval_odd(coarse, i);
    }

    if (bcs.is_periodic()) {
        const int nc = coarse.size() - 1;
        const int nf = fine.size() - 1;
        fine[nf-1] = (coarse[nc-1] + coarse[0]) / 2;
    }

    fine.front() = coarse.front();
    fine.back() = coarse.back();
}

void InterpolatorLinear::run_device(Array& coarse, Array& fine, BoundaryConditions& bcs) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, coarse.size() - 1);
    const int blocksPerGrid = (coarse.size() + threadsPerBlock - 1) / threadsPerBlock;
    interpolator_linear::kernel<<<blocksPerGrid, threadsPerBlock>>>(coarse, fine, bcs);
}

} // namespace modules
} // namespace gmf
