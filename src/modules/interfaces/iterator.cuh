#pragma once

#include <iostream>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"
#include "src/modules/interfaces/parallel.cuh"

namespace pmf {
namespace modules {

class Iterator : public Parallel {
public:
    Iterator() {}
    Iterator(uint gpu_threads) : Parallel(gpu_threads) {}
    Iterator(uint gpu_threads, uint cpu_threads) : Parallel(gpu_threads, cpu_threads) {}

    virtual void run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) = 0;
    virtual void run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) = 0;
};

} // namespace modules
} // namespace pmf
