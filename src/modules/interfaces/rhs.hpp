#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"


namespace pmf {
namespace modules {

class RHS {
public:
    RHS() {}

    virtual void run(Array& rhs, const Grid& grid, const BoundaryConditions& bcs) = 0;
};

} // namespace modules
} // namespace pmf
