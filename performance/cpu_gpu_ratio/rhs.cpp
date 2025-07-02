#include "rhs.hpp"

#include "equation.hpp"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;

void HeatEquationRHS::run(gmf::Array& rhs, const gmf::Grid& grid,
        const gmf::modules::BoundaryConditions& bcs) {

    const gmf::Array& x = grid.get_x();
    const int n = x.size() - 1;

    for (int i = 0; i <= n; ++i) {
        rhs[i] = 4 * PI * PI * std::sin(2 * PI * x[i]);
    }

    if (bcs.is_left_dirichlet()) {
        rhs[0] = bcs.get_left();
    } else if (bcs.is_left_neumann()) {
        const double h = grid.get_cell_width();
        rhs[0] -= bcs.get_left() * 2 / h;
    }

    if (bcs.is_right_dirichlet()) {
        rhs[n] = bcs.get_right();
    } else if (bcs.is_right_neumann()) {
        const double h = grid.get_cell_width();
        rhs[n] += bcs.get_right() * 2 / h;
    }
}
