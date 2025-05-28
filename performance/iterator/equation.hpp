#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"
#include "src/grid.hpp"


class HeatEquation : public gmf::modules::Equation {
public:
    HeatEquation() {}

    virtual double rhs(const double x, const double v) const override;

    virtual double initial_condition(const double x) const override;
};


