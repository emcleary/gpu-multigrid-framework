#include "add.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"

namespace pmf {
namespace modules {

namespace add {
__global__
void kernel(ArrayRaw a, ArrayRaw b, ArrayRaw c) {
    const int n = c.size();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}
} // namespace add


void Add::run_host(Array& a, Array& b, Array& c) {
    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for
    for (int i = 0; i < c.size(); ++i)
        c[i] = a[i] + b[i];
}

void Add::run_device(Array& a, Array& b, Array& c) {
    const uint threads = std::min(m_max_threads_per_block, static_cast<uint>(c.size()) - 1);
    const uint blocks = (c.size() + threads - 1) / threads;
    add::kernel<<<blocks, threads>>>(a, b, c);
}

} // namespace modules
} // namespace pmf
