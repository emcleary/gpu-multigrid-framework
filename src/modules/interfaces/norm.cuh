#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces/parallel.cuh"


namespace pmf {
namespace modules {

class Norm : public Parallel {
public:
    Norm() {}
    Norm(uint gpu_threads, uint cpu_threads) : Parallel(gpu_threads, cpu_threads) {}

    virtual double run_host(const Array& resid, const Grid& grid) = 0;
    virtual double run_device(const Array& resid, const Grid& grid) = 0;
};

} // namespace modules
} // namespace pmf
