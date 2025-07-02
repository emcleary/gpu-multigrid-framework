#pragma once

#include <iostream>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/boundary_conditions.hpp"

namespace gmf {
namespace modules {

class Iterator {
public:
    Iterator(const size_t max_threads_per_block) : m_max_threads_per_block(max_threads_per_block) {}

    virtual void run_host(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) = 0;

    virtual void run_device(Array& v, const Array& f, const BoundaryConditions& bcs, const Grid& grid) = 0;

protected:
    const size_t m_max_threads_per_block;
};

} // namespace modules
} // namespace gmf
