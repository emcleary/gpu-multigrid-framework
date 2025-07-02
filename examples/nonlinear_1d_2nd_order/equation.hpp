#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"
#include "src/grid.hpp"


class NonlinearEquation : public gmf::modules::Equation {
public:
    NonlinearEquation(const double gamma) : m_gamma(gamma) {}

    virtual double rhs(const double x) const override;

    virtual double analytical_solution(const double x) const override;

    double analytical_derivative(const double x) const;

    virtual double initial_condition(const double x) const override;

    double get_gamma() const { return m_gamma; }

private:
    const double m_gamma;
};


