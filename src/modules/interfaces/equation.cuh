#pragma once

#include <cmath>

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"

namespace pmf {
namespace modules {

class Equation {
public:
    Equation() {}

    virtual void fill_initial_condition(Array& v, const Grid& grid, const BoundaryConditions& bcs) const = 0;

    virtual void fill_rhs(Array& f, const Grid& grid, const BoundaryConditions& bcs) const = 0;

    // included here as virtual method to facilitate output tools
    virtual double analytical_solution(const double x) const { return std::nan(""); };
    virtual double analytical_solution(const double x, const double y) const { return std::nan(""); };
};

} // namespace modules
} // namespace pmf
