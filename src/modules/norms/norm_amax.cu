#include "norm_amax.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/utilities.hpp"

namespace pmf {
namespace modules {

NormAmax::NormAmax() {
    cublasCheck(cublasCreate(&m_handle));
}

NormAmax::NormAmax(uint gpu_threads, uint cpu_threads) : Norm(gpu_threads, cpu_threads) {
    cublasCheck(cublasCreate(&m_handle));
}

NormAmax::~NormAmax() {
    cublasCheck(cublasDestroy(m_handle));
}

double NormAmax::run_host(const Array& array, const Grid& grid) {
    double norm = 0;
    omp_set_num_threads(m_omp_threads);
#pragma omp parallel for reduction(max:norm)
    for (int i = 0; i < array.size(); ++i)
        norm = std::max(norm, std::abs(array[i]));
    return norm;
}

double NormAmax::run_device(const Array& array, const Grid& grid) {
    int index;
    cublasCheck(cublasIdamax(m_handle, array.size(), array.data(), 1, &index));
    return std::abs(array[index - 1]);
}

} // namespace modules
} // namespace pmf
