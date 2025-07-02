#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/interpolator.cuh"


namespace gmf {
namespace modules {

class InterpolatorLinear : public Interpolator {
public:
    InterpolatorLinear(const int max_threads_per_block) : Interpolator(max_threads_per_block) {}

    virtual void run_host(Array& coarse, Array& fine, BoundaryConditions& bcs);

    virtual void run_device(Array& coarse, Array& fine, BoundaryConditions& bcs);
};

} // namespace modules
} // namespace gmf
