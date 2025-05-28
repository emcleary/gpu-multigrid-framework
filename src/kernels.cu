#include "kernels.cuh"

#include <cuda_runtime.h>

#include "array.hpp"


namespace gmf {

__global__ void kernel_add(ArrayRaw a, ArrayRaw b, ArrayRaw c) {
    const int n = c.size();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        c[i] = a[i] + b[i];
}


__global__ void kernel_sub(ArrayRaw a, ArrayRaw b, ArrayRaw c) {
    const int n = c.size();
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        c[i] = a[i] - b[i];
}

__global__ void kernel_copy(ArrayRaw a, ArrayRaw b, const int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        b[i] = a[i];
}

} // namespace gmf
