#include "copy.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"

namespace pmf {
namespace modules {

namespace copy {
__global__
void kernel(ArrayRaw a, ArrayRaw b) {
    const int n = a.size();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        b[i] = a[i];
}
} // namespace copy


void Copy::run_host(Array& a, Array& b) {
    const int n = a.size();
    assert(n <= b.size());
    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        b[i] = a[i];
}

void Copy::run_device(Array& a, Array& b) {
    const uint n = a.size();
    assert(n <= b.size());
    const uint threads = std::min(m_max_threads_per_block, n - 1);
    const uint blocks = (n + threads - 1) / threads;
    copy::kernel<<<blocks, threads>>>(a, b);
}

} // namespace modules
} // namespace pmf
