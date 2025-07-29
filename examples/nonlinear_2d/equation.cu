#include "equation.cuh"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;


double NonlinearEquation::analytical_solution(const double x, const double y) const {
    return std::exp(x) * std::sin(2 * PI * y);
}

double NonlinearEquation::analytical_derivative_x(const double x, const double y) const {
    return std::exp(x) * std::sin(2 * PI * y);
}

double NonlinearEquation::analytical_derivative_y(const double x, const double y) const {
    return 2 * PI * std::exp(x) * std::cos(2 * PI * y);
}

double NonlinearEquation::initial_condition(const double x, const double y) const {
    return 0;
}

double NonlinearEquation::rhs(const double x, const double y) const {
    const double u = analytical_solution(x, y);
    return u * (4 * PI * PI - 1 + m_gamma * u);
}

double NonlinearEquation::neumann_bc_west(const double x, const double y) const {
    return analytical_derivative_x(x, y);
}

void NonlinearEquation::fill_rhs(pmf::Array& f, const pmf::Grid& grid,
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

    if (bcs.is_west_neumann())
        for (int j = 0; j <= ny; ++j)
            f(0, j) -= neumann_bc_west(x[0], y[j]) * 2 / h;

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

void NonlinearEquation::fill_initial_condition(pmf::Array& v, const pmf::Grid& grid,
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
