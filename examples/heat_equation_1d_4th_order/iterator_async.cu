#include "iterator_async.cuh"

#include <cuda_runtime.h>

#include "iterator_schemes.cuh"
#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


namespace async {
__global__
void kernel3(ArrayRaw v, const ArrayRaw f, const double h2) {
    if (threadIdx.x == 1)
        v[threadIdx.x] = eval2(v, f, h2, threadIdx.x);
}

__global__
void kernel5(ArrayRaw v, const ArrayRaw f, const double h2) {
    if (threadIdx.x == 1)
        v[threadIdx.x] = eval2(v, f, h2, threadIdx.x);
    else if (threadIdx.x == 2)
        v[threadIdx.x] = eval4(v, f, h2, threadIdx.x);
    else if (threadIdx.x == 3)
        v[threadIdx.x] = eval2(v, f, h2, threadIdx.x);
}

__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h2) {
    int index = 3 * (threadIdx.x + blockDim.x * blockIdx.x);
    const int stride = 3 * blockDim.x * gridDim.x;

    while (index < v.size()) {
        int idx = index + 1;
        if (1 < idx && idx < v.size() - 2) {
            v[idx] = eval4(v, f, h2, idx);
        } else if (idx == 1) {
            v[idx] = eval4left(v, f, h2, idx);
        } else if (idx == v.size() - 2) {
            v[idx] = eval4right(v, f, h2, idx);
        }

        ++idx;
        if (1 < idx && idx < v.size() - 2) {
            v[idx] = eval4(v, f, h2, idx);
        } else if (idx == 1) {
            v[idx] = eval4left(v, f, h2, idx);
        } else if (idx == v.size() - 2) {
            v[idx] = eval4right(v, f, h2, idx);
        }

        ++idx;
        if (1 < idx && idx < v.size() - 2) {
            v[idx] = eval4(v, f, h2, idx);
        } else if (idx == 1) {
            v[idx] = eval4left(v, f, h2, idx);
        } else if (idx == v.size() - 2) {
            v[idx] = eval4right(v, f, h2, idx);
        }

        index += stride;
    }
}
}

void IteratorAsync::run_device(Array& v, const Array& f, const Grid& grid) {
    const double h2 = grid.get_cell_width() * grid.get_cell_width();
    if (v.size() == 3) {
        async::kernel3<<<1, 3>>>(v, f, h2);
    } else if (v.size() == 5) {
        async::kernel5<<<1, 5>>>(v, f, h2);
    } else {
        size_t i = 1 << (int)std::ceil(std::log2(v.size() / 3));
        const int threadsPerBlock = std::min(m_max_threads_per_block, i);
        const int blocksPerGrid = (i + threadsPerBlock - 1) / threadsPerBlock;
        async::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h2);
    }
}

void IteratorAsync::run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
    run_device(e, r, grid);
}
