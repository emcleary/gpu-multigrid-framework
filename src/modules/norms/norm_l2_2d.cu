#include "norm_l2_2d.cuh"

#include <iostream>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/utilities.hpp"


namespace pmf {
namespace modules {

L2Norm2D::L2Norm2D() {
    cublasCheck(cublasCreate(&m_handle));
}

L2Norm2D::L2Norm2D(uint gpu_threads, uint cpu_threads) : Norm(gpu_threads, cpu_threads) {
    cublasCheck(cublasCreate(&m_handle));
}

L2Norm2D::~L2Norm2D() {
    cublasCheck(cublasDestroy(m_handle));
}

double L2Norm2D::run_host(const Array& array, const Grid& grid) {
    double norm2 = 0;
    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for reduction(+:norm2)
    for (int i = 0; i < array.size(); ++i)
        norm2 += array[i] * array[i];
    return std::sqrt(norm2 * grid.get_cell_width() * grid.get_cell_height());
}

double L2Norm2D::run_device(const Array& array, const Grid& grid) {
    double norm_gpu;
    cublasCheck(cublasDnrm2(m_handle, array.size(), array.data(), 1, &norm_gpu));
    cudaCheck(cudaDeviceSynchronize());
    return norm_gpu * std::sqrt(grid.get_cell_width() * grid.get_cell_height());
}

} // namespace modules
} // namespace pmf
