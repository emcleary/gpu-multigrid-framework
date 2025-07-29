#include "equation.hpp"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;


double HeatEquation::analytical_solution(const double x, const double y) const {
    // return std::cos(2 * PI * x);
    // return std::cos(2 * PI * y);
    return std::cos(2 * PI * (x - 0.123)) * std::sin(2 * PI * (y + 0.123));
}

double HeatEquation::analytical_derivative_x(const double x, const double y) const {
    // return -2 * PI * std::sin(2 * PI * x);
    // return 0;
    return -2 * PI * std::sin(2 * PI * (x - 0.123)) * std::sin(2 * PI * (y + 0.123));
}

double HeatEquation::analytical_derivative_y(const double x, const double y) const {
    // return 0;
    // return -2 * PI * std::sin(2 * PI * y);
    return 2 * PI * std::cos(2 * PI * (x - 0.123)) * std::cos(2 * PI * (y + 0.123));
}

double HeatEquation::initial_condition(const double x, const double y) const {
    return 0;
}

double HeatEquation::rhs(const double x, const double y) const {
    // return 4 * PI * PI * std::cos(2 * PI * x);
    // return 4 * PI * PI * std::cos(2 * PI * y);
    return 8 * PI * PI * std::cos(2 * PI * (x - 0.123)) * std::sin(2 * PI * (y + 0.123));
}

void HeatEquation::fill_rhs(pmf::Array& f, const pmf::Grid& grid,
        const pmf::modules::BoundaryConditions& bcs) const {

    const double h = grid.get_cell_width();
    assert(h == grid.get_cell_height());

    const pmf::Array& x = grid.get_x();
    const pmf::Array& y = grid.get_y();
    const int nx = x.size() - 1;
    const int ny = y.size() - 1;

    for (int i = 0; i <= nx; ++i)
        for (int j = 0; j <= ny; ++j)
            f(i, j) = rhs(x[i], y[j]);

    if (bcs.is_north_neumann())
        for (int i = 0; i <= nx; ++i)
            f(i, ny) += analytical_derivative_y(x[i], y[ny]) * 2 / h;

    if (bcs.is_south_neumann())
        for (int i = 0; i <= nx; ++i)
            f(i, 0) -= analytical_derivative_y(x[i], y[0]) * 2 / h;

    if (bcs.is_east_neumann())
        for (int j = 0; j <= ny; ++j)
            f(nx, j) += analytical_derivative_x(x[nx], y[j]) * 2 / h;

    if (bcs.is_west_neumann())
        for (int j = 0; j <= ny; ++j)
            f(0, j) -= analytical_derivative_x(x[0], y[j]) * 2 / h;

    if (bcs.is_north_dirichlet())
        for (int i = 0; i <= nx; ++i)
            f(i, ny) = analytical_solution(x[i], y[ny]);

    if (bcs.is_south_dirichlet())
        for (int i = 0; i <= nx; ++i)
            f(i, 0) = analytical_solution(x[i], y[0]);

    if (bcs.is_east_dirichlet())
        for (int j = 0; j <= ny; ++j)
            f(nx, j) = analytical_solution(x[nx], y[j]);

    if (bcs.is_west_dirichlet())
        for (int j = 0; j <= ny; ++j)
            f(0, j) = analytical_solution(x[0], y[j]);
}

void HeatEquation::fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
        const pmf::modules::BoundaryConditions& bcs) const {

    const double h = grid.get_cell_width();
    assert(h == grid.get_cell_height());

    const pmf::Array& x = grid.get_x();
    const pmf::Array& y = grid.get_y();
    const int nx = x.size() - 1;
    const int ny = y.size() - 1;

    for (int i = 0; i <= nx; ++i)
        for (int j = 0; j <= ny; ++j)
            v(i, j) = initial_condition(x[i], y[j]);

    if (bcs.is_north_dirichlet())
        for (int i = 0; i <= nx; ++i)
            v(i, ny) = analytical_solution(x[i], y[ny]);

    if (bcs.is_south_dirichlet())
        for (int i = 0; i <= nx; ++i)
            v(i, 0) = analytical_solution(x[i], y[0]);

    if (bcs.is_east_dirichlet())
        for (int j = 0; j <= ny; ++j)
            v(nx, j) = analytical_solution(x[nx], y[j]);

    if (bcs.is_west_dirichlet())
        for (int j = 0; j <= ny; ++j)
            v(0, j) = analytical_solution(x[0], y[j]);
}
