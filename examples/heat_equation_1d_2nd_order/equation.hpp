#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"
#include "src/grid.hpp"


class HeatEquation : public pmf::modules::Equation {
public:
    HeatEquation() {}

    virtual void fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
            const pmf::modules::BoundaryConditions& bcs) const;

    virtual void fill_rhs(pmf::Array& f, const pmf::Grid& grid,
            const pmf::modules::BoundaryConditions& bcs) const;

    virtual double analytical_solution(const double x) const override;

    double analytical_derivative(const double x) const;

    double initial_condition(const double x) const;

    double rhs(const double x) const;
};


