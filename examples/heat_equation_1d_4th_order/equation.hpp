#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"
#include "src/grid.hpp"

// TODO: make linear and nonlinear equations (linear doesn't require error calculations)

// TODO: refactor somehow so that more optimized iterative solvers, which are equation-specifc,
// can be implemented
// e.g. some equations might be more efficient if they made use of shared memory, which
// is not possible as currently implemented

// TODO: split numerical and analytical stuff?
// strange to have rhs, analytical_solution, initial solution take double x
// while lhs, gs_element, gs_error take in arrays
class HeatEquation : public gmf::modules::Equation {
public:
    HeatEquation() {}

    virtual double rhs(const double x) const override;

    virtual double analytical_solution(const double x) const override;

    virtual double initial_condition(const double x) const override;
};


