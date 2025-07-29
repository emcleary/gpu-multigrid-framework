#include "equation.hpp"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;


double HeatEquation::analytical_solution(const double x) const {
    return  std::cos(2 * PI * x);
}

double HeatEquation::analytical_derivative(const double x) const {
    return  -2 * PI * std::sin(2 * PI * x);
}

double HeatEquation::initial_condition(const double x) const {
    return 0;
}

double HeatEquation::rhs(const double x) const {
    return 4 * PI * PI * std::cos(2 * PI * x);
}

void HeatEquation::fill_rhs(pmf::Array& f, const pmf::Grid& grid,
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
        f[0] -= analytical_derivative(x[0]) * 2 / h;
    }

    if (bcs.is_east_dirichlet()) {
        f[n] = analytical_solution(x[n]);
    } else if (bcs.is_east_neumann()) {
        const double h = grid.get_cell_width();
        f[n] += analytical_derivative(x[n]) * 2 / h;
    }
}

void HeatEquation::fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
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
