#pragma once

#include <cmath>

#include "src/array.hpp"
#include "src/grid.hpp"

namespace gmf {
namespace modules {

class Equation {
public:
    Equation() {}

    virtual double rhs(const double x, const double v) const = 0;

    virtual  double analytical_solution(const double x) const { return std::nan(""); };

    virtual double initial_condition(const double x) const = 0;
};

} // namespace modules
} // namespace gmf
