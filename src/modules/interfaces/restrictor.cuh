#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"
#include "src/modules/interfaces/parallel.cuh"


namespace pmf {
namespace modules {

class Restrictor : public Parallel {
public:
    Restrictor() {}
    Restrictor(uint gpu_threads, uint cpu_threads) : Parallel(gpu_threads, cpu_threads) {}

    virtual void run_host(Array& fine, Array& coarse, BoundaryConditions& bcs) = 0;
    virtual void run_device(Array& fine, Array& coarse, BoundaryConditions& bcs) = 0;
};

} // namespace modules
} // namespace pmf
