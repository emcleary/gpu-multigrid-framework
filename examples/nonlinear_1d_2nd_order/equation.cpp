#include "equation.hpp"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;

double NonlinearEquation::rhs(const double x, const double v) const {
    return (x*x + 3*x) * std::exp(x) + m_gamma * std::exp(2 * x) * (x*x*x*x - 2*x*x + x);
}

double NonlinearEquation::analytical_solution(const double x) const {
    return std::exp(x) * (x - x * x);
}

double NonlinearEquation::initial_condition(const double x) const {
    return 0.0;
}
