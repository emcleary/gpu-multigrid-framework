#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"
#include "src/modules/interfaces/parallel.cuh"


namespace pmf {
namespace modules {

class Interpolator : public Parallel {
public:
    Interpolator() {}
    Interpolator(uint gpu_threads, uint cpu_threads) : Parallel(gpu_threads, cpu_threads) {}

    virtual void run_host(Array& coarse, Array& fine, BoundaryConditions& bcs) = 0;
    virtual void run_device(Array& coarse, Array& fine, BoundaryConditions& bcs) = 0;
};

} // namespace modules
} // namespace pmf
