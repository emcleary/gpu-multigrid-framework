#include "restrictor_full_weighting.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"


namespace pmf {
namespace modules {

namespace restrictor_full_weighting {

__global__
void kernel(ArrayRaw fine, ArrayRaw coarse, BoundaryConditions bcs) {
    const int ci = threadIdx.x + blockDim.x * blockIdx.x;
    const int fi = ci * 2;

    if (0 < ci && ci < coarse.size() - 1)
        coarse[ci] = (fine[fi-1] + 2 * fine[fi] + fine[fi+1]) / 4;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (bcs.is_periodic_x()) {
            const int n = fine.size() - 1;
            coarse.front() = (fine[n-1] + 2 * fine[0] + fine[1]) / 4;
            coarse.back() = coarse.front();
        } else {
            if (bcs.is_west_dirichlet()) {
                coarse.front() = fine[0];
            } else { // neumann
                coarse.front() = (2 * fine[0] + fine[1]) / 4;
            }

            if (bcs.is_east_dirichlet()) {
                coarse.back() = fine.back();
            } else { // neumann
                const int n = fine.size() - 1;
                coarse.back() = (2 * fine[n] + fine[n-1]) / 4;
            }
        }
    }
}

} // namespace restrictor_full_weighting


void RestrictorFullWeighting::run_host(Array& fine, Array& coarse, BoundaryConditions& bcs) {

    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for
    for (int ci = 1; ci < coarse.size() - 1; ++ci) {
        const int fi = ci * 2;
        coarse[ci] = (fine[fi-1] + 2 * fine[fi] + fine[fi+1]) / 4;
    }

    if (bcs.is_periodic_x()) {
        const int n = fine.size() - 1;
        coarse.front() = (fine[n-1] + 2 * fine[0] + fine[1]) / 4;
        coarse.back() = coarse.front();
    } else {
        if (bcs.is_west_dirichlet()) {
            coarse.front() = fine[0];
        } else { // neumann
            coarse.front() = (2 * fine[0] + fine[1]) / 4;
        }

        if (bcs.is_east_dirichlet()) {
            coarse.back() = fine.back();
        } else { // neumann
            const int n = fine.size() - 1;
            coarse.back() = (2 * fine[n] + fine[n-1]) / 4;
        }
    }
}

void RestrictorFullWeighting::run_device(Array& fine, Array& coarse, BoundaryConditions& bcs) {
    const uint threads = std::min(m_max_threads_per_block, static_cast<uint>(coarse.size()) - 1);
    const uint blocks = (coarse.size() + threads - 1) / threads;
    restrictor_full_weighting::kernel<<<blocks, threads>>>(fine, coarse, bcs);
}

} // namespace modules
} // namespace pmf
