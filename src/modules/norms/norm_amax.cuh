#pragma once

#include <cublas_v2.h>

#include "src/array.hpp"
#include "src/grid.hpp"

// #include "src/modules/interfaces.hpp"
#include "src/modules/interfaces/norm.cuh"

namespace gmf {
namespace modules {

class NormAmax : public Norm {
public:
    NormAmax();
    ~NormAmax();

    virtual double run_host(const Array& array, const Grid& grid) override;
    virtual double run_device(const Array& array, const Grid& grid) override;

private:
    cublasHandle_t m_handle;
    
};

} // namespace modules
} // namespace gmf {
