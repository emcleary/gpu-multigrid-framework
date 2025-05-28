#include "restrictor_injection.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "src/array.hpp"


namespace gmf {
namespace modules {

namespace restrictor_injection {

__host__ __device__
double eval(const ArrayRaw& fine, const int i) {
    const int j = i + i;
    return fine[j];
}

__global__
void kernel(ArrayRaw fine, ArrayRaw coarse) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int n_pts = coarse.size();

    if (idx < coarse.size())
        coarse[idx] = eval(fine, idx);
}

} // namespace restrictor_injection


void RestrictorInjection::run_host(Array& fine, Array& coarse) {
    for (int i = 0; i < coarse.size(); ++i)
        coarse[i] = restrictor_injection::eval(fine, i);
}

void RestrictorInjection::run_device(Array& fine, Array& coarse) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, coarse.size() - 1);
    const int blocksPerGrid = (coarse.size() + threadsPerBlock - 1) / threadsPerBlock;
    restrictor_injection::kernel<<<blocksPerGrid, threadsPerBlock>>>(fine, coarse);
}

} // namespace modules
} // namespace gmf
