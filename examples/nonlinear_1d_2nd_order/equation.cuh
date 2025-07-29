#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"
#include "src/grid.hpp"


class NonlinearEquation : public pmf::modules::Equation {
public:
    NonlinearEquation(const double gamma) : m_gamma(gamma) {}

    virtual void fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
            const pmf::modules::BoundaryConditions& bcs) const;

    virtual void fill_rhs(pmf::Array& f, const pmf::Grid& grid,
            const pmf::modules::BoundaryConditions& bcs) const;

    virtual double analytical_solution(const double x) const override;

    __host__ __device__
    double analytical_derivative(const double x) const;

    __host__ __device__
    double get_gamma() const { return m_gamma; }

    __host__ __device__
    double neumann_bc_west(const double x) const;

    __host__ __device__
    double neumann_bc_east(const double x) const;

private:
    double initial_condition(const double x) const;
    double rhs(const double x) const;

    const double m_gamma;
};


