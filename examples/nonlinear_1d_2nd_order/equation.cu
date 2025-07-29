#include "equation.cuh"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;

double NonlinearEquation::analytical_solution(const double x) const {
    return std::exp(x) * (x - x * x);
}

double NonlinearEquation::analytical_derivative(const double x) const {
    return std::exp(x) * (1 - x - x * x);
}

double NonlinearEquation::initial_condition(const double x) const {
    return 0.0;
}

double NonlinearEquation::rhs(const double x) const {
    return (x*x + 3*x) * std::exp(x) + m_gamma * std::exp(2 * x) * (x*x*x*x - 2*x*x + x);
}

double NonlinearEquation::neumann_bc_west(const double x) const {
    return analytical_derivative(x);
}

double NonlinearEquation::neumann_bc_east(const double x) const {
    return analytical_derivative(x);
}

void NonlinearEquation::fill_rhs(pmf::Array& f, const pmf::Grid& grid,
        const pmf::modules::BoundaryConditions& bcs) const {

    const pmf::Array& x = grid.get_x();
    const int n = x.size() - 1;

    for (int i = 0; i <= n; ++i) {
        f[i] = rhs(x[i]);
    }

    if (bcs.is_west_dirichlet()) {
        f[0] = analytical_solution(x[0]);
    } else if (bcs.is_west_neumann()) {
        const double h = grid.get_cell_width();
        f[0] -= neumann_bc_west(x[0]) * 2 / h;
    }

    if (bcs.is_east_dirichlet()) {
        f[n] = analytical_solution(x[n]);
    } else if (bcs.is_east_neumann()) {
        const double h = grid.get_cell_width();
        f[n] += neumann_bc_east(x[n]) * 2 / h;
    }
}

void NonlinearEquation::fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
        const pmf::modules::BoundaryConditions& bcs) const {

    const pmf::Array& x = grid.get_x();
    const int n = x.size() - 1;

    for (int i = 0; i <= n; ++i)
        v[i] = initial_condition(x[i]);

    if (bcs.is_west_dirichlet())
        v[0] = analytical_solution(x[0]);
    
    if (bcs.is_east_dirichlet())
        v[n] = analytical_solution(x[n]);
}
