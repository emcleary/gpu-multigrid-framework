#pragma once

#include <cublas_v2.h>

#include "src/array.hpp"
#include "src/grid.hpp"

#include "src/modules/interfaces/norm.cuh"

namespace pmf {
namespace modules {

class L2Norm2D : public Norm {
public:
    L2Norm2D();
    L2Norm2D(uint gpu_threads, uint cpu_threads);

    ~L2Norm2D();

    virtual double run_host(const Array& array, const Grid& grid) override;
    virtual double run_device(const Array& array, const Grid& grid) override;

private:
    cublasHandle_t m_handle;
};

} // namespace modules
} // namespace pmf {
