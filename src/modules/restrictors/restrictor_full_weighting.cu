#include "restrictor_full_weighting.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"


namespace gmf {
namespace modules {

namespace restrictor_full_weighting {

__host__ __device__
double eval(const ArrayRaw& fine, const int i) {
    const int j = i + i;
    return (fine[j-1] + 2 * fine[j] + fine[j+1]) / 4;
}

__global__
void kernel(ArrayRaw fine, ArrayRaw coarse, BoundaryConditions bcs) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (0 < idx && idx < coarse.size() - 1)
        coarse[idx] = eval(fine, idx);

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (bcs.is_periodic()) {
            const int n = fine.size() - 1;
            coarse.front() = (fine[n-1] + 2 * fine[0] + fine[1]) / 4;
            coarse.back() = coarse.front();
        } else {
            if (bcs.is_left_dirichlet()) {
                coarse.front() = fine[0];
            } else { // neumann
                coarse.front() = (2 * fine[0] + fine[1]) / 4;
            }

            if (bcs.is_right_dirichlet()) {
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
    for (int i = 1; i < coarse.size() - 1; ++i)
        coarse[i] = restrictor_full_weighting::eval(fine, i);

    if (bcs.is_periodic()) {
        const int n = fine.size() - 1;
        coarse.front() = (fine[n-1] + 2 * fine[0] + fine[1]) / 4;
        coarse.back() = coarse.front();
    } else {
        if (bcs.is_left_dirichlet()) {
            coarse.front() = fine[0];
        } else { // neumann
            coarse.front() = (2 * fine[0] + fine[1]) / 4;
        }

        if (bcs.is_right_dirichlet()) {
            coarse.back() = fine.back();
        } else { // neumann
            const int n = fine.size() - 1;
            coarse.back() = (2 * fine[n] + fine[n-1]) / 4;
        }
    }
}

void RestrictorFullWeighting::run_device(Array& fine, Array& coarse, BoundaryConditions& bcs) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, coarse.size() - 1);
    const int blocksPerGrid = (coarse.size() + threadsPerBlock - 1) / threadsPerBlock;
    restrictor_full_weighting::kernel<<<blocksPerGrid, threadsPerBlock>>>(fine, coarse, bcs);
}

} // namespace modules
} // namespace gmf
