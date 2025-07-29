#include "norm_l2.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/utilities.hpp"


namespace pmf {
namespace modules {

NormL2::NormL2() {
    cublasCheck(cublasCreate(&m_handle));
}

NormL2::NormL2(uint gpu_threads, uint cpu_threads) : Norm(gpu_threads, cpu_threads) {
    cublasCheck(cublasCreate(&m_handle));
}

NormL2::~NormL2() {
    cublasCheck(cublasDestroy(m_handle));
}

double NormL2::run_host(const Array& array, const Grid& grid) {
    double norm2 = 0;
    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for reduction(+:norm2)
    for (int i = 0; i < array.size() - 1; ++i)
        norm2 += array[i] * array[i];
    return std::sqrt(norm2 * grid.get_cell_width());
}

double NormL2::run_device(const Array& array, const Grid& grid) {
    double norm;
    cublasCheck(cublasDnrm2(m_handle, array.size(), array.data(), 1, &norm));
    return norm * std::sqrt(grid.get_cell_width());
}

} // namespace modules
} // namespace pmf
