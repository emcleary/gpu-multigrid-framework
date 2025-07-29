#include "restrictor_injection.cuh"

#include <iostream>

#include <cuda_runtime.h>

#include "src/array.hpp"


namespace pmf {
namespace modules {

namespace restrictor_injection {

__global__
void kernel(ArrayRaw fine, ArrayRaw coarse) {
    const int ci = threadIdx.x + blockDim.x * blockIdx.x;
    const int fi = ci * 2;
    
    if (ci < coarse.size())
        coarse[ci] = fine[fi];
}

} // namespace restrictor_injection


void RestrictorInjection::run_host(Array& fine, Array& coarse) {
    for (int ci = 0; ci < coarse.size(); ++ci) {
        const int fi = ci * 2;
        coarse[ci] = fine[fi];
    }
}

void RestrictorInjection::run_device(Array& fine, Array& coarse) {
    const int threads = std::min(m_max_threads_per_block, static_cast<uint>(coarse.size()) - 1);
    const int blocks = (coarse.size() + threads - 1) / threads;
    restrictor_injection::kernel<<<blocks, threads>>>(fine, coarse);
}

} // namespace modules
} // namespace pmf
