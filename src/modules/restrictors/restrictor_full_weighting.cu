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
void kernel(ArrayRaw fine, ArrayRaw coarse) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int n_pts = coarse.size();

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < n_pts - 1)
        coarse[idx] = eval(fine, idx);

    // boundary conditions
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        coarse.front() = fine.front();
        coarse.back() = fine.back();
    }
}

} // namespace restrictor_full_weighting


void RestrictorFullWeighting::run_host(Array& fine, Array& coarse) {
    for (int i = 1; i < coarse.size() - 1; ++i)
        coarse[i] = restrictor_full_weighting::eval(fine, i);
    coarse.front() = fine.front();
    coarse.back() = fine.back();
}

void RestrictorFullWeighting::run_device(Array& fine, Array& coarse) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, coarse.size() - 1);
    const int blocksPerGrid = (coarse.size() + threadsPerBlock - 1) / threadsPerBlock;
    restrictor_full_weighting::kernel<<<blocksPerGrid, threadsPerBlock>>>(fine, coarse);
}

} // namespace modules
} // namespace gmf
