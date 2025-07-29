#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"
#include "src/grid.hpp"


class NonlinearEquation : public pmf::modules::Equation {
public:
    NonlinearEquation(double gamma) : m_gamma(gamma) {}

    virtual void fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
            const pmf::modules::BoundaryConditions& bcs) const;

    virtual void fill_rhs(pmf::Array& f, const pmf::Grid& grid,
            const pmf::modules::BoundaryConditions& bcs) const;

    virtual double analytical_solution(const double x, const double y) const override;

    __host__ __device__
    double analytical_derivative_x(const double x, const double y) const;

    __host__ __device__
    double analytical_derivative_y(const double x, const double y) const;

    __host__ __device__
    double neumann_bc_west(const double x, const double y) const;

    double get_gamma() const { return m_gamma; };

private:
    double initial_condition(const double x, const double y) const;
    double rhs(const double x, const double y) const;
    const double m_gamma;
};


