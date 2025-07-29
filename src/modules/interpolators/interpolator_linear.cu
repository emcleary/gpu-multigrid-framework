#include "interpolator_linear.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"

namespace pmf {
namespace modules {

namespace interpolator_linear {
__global__
void kernel(ArrayRaw coarse, ArrayRaw fine, BoundaryConditions bcs) {
    const int ci = threadIdx.x + blockDim.x * blockIdx.x;
    const int fi = ci * 2;

    if (ci < coarse.size() - 1) {
        fine[fi] = coarse[ci];
        fine[fi + 1] =  (coarse[ci] + coarse[ci + 1]) / 2;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (bcs.is_periodic_x()) {
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
    
    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for
    for (int ci = 0; ci < coarse.size() - 1; ++ci) {
        const int fi = ci * 2;
        fine[fi] = coarse[ci];
        fine[fi + 1] =  (coarse[ci] + coarse[ci + 1]) / 2;
    }

    if (bcs.is_periodic_x()) {
        const int nc = coarse.size() - 1;
        const int nf = fine.size() - 1;
        fine[nf-1] = (coarse[nc-1] + coarse[0]) / 2;
    }

    fine.front() = coarse.front();
    fine.back() = coarse.back();
}

void InterpolatorLinear::run_device(Array& coarse, Array& fine, BoundaryConditions& bcs) {
    const uint threads = std::min(m_max_threads_per_block, static_cast<uint>(coarse.size()) - 1);
    const uint blocks = (coarse.size() + threads - 1) / threads;
    interpolator_linear::kernel<<<blocks, threads>>>(coarse, fine, bcs);
}

} // namespace modules
} // namespace pmf
