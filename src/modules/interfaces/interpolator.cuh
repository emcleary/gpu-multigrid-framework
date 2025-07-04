#pragma once

#include "src/array.hpp"
#include "src/modules/boundary_conditions.hpp"


namespace gmf {
namespace modules {

class Interpolator {
public:
    Interpolator(const size_t max_threads_per_block) : m_max_threads_per_block(max_threads_per_block) {}

    virtual void run_host(Array& coarse, Array& fine, BoundaryConditions& bcs) = 0;
    virtual void run_device(Array& coarse, Array& fine, BoundaryConditions& bcs) = 0;

protected:
    const size_t m_max_threads_per_block;
};

} // namespace modules
} // namespace gmf
