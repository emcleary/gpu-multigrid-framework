#pragma once

#include <cuda_runtime.h>

#include "array.hpp"

namespace pmf {

// __global__ void kernel_add(ArrayRaw a, ArrayRaw b, ArrayRaw c);
// __global__ void kernel_sub(ArrayRaw a, ArrayRaw b, ArrayRaw c);
__global__ void kernel_copy(ArrayRaw a, ArrayRaw b, const int n);

} // namespace pmf
