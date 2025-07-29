#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/interpolator.cuh"


namespace pmf {
namespace modules {

class InterpolatorLinear : public Interpolator {
public:
    InterpolatorLinear() {}
    InterpolatorLinear(uint gpu_threads, uint cpu_threads) : Interpolator(gpu_threads, cpu_threads) {}

    virtual void run_host(Array& coarse, Array& fine, BoundaryConditions& bcs);
    virtual void run_device(Array& coarse, Array& fine, BoundaryConditions& bcs);
};

} // namespace modules
} // namespace pmf
