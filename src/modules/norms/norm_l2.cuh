#pragma once

#include <cublas_v2.h>

#include "src/array.hpp"
#include "src/grid.hpp"

#include "src/modules/interfaces/norm.cuh"

namespace gmf {
namespace modules {

class NormL2 : public Norm {
public:
    NormL2();
    ~NormL2();

    virtual double run_host(const Array& array, const Grid& grid) override;
    virtual double run_device(const Array& array, const Grid& grid) override;

private:
    cublasHandle_t m_handle;
};

} // namespace modules
} // namespace gmf {
