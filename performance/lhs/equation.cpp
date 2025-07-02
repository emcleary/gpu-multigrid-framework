#include "equation.hpp"

#include <cmath>
#include <numbers>

static const double PI = std::numbers::pi;

double HeatEquation::rhs(const double x) const {
    return 4 * PI * PI * std::sin(2 * PI * x);
}

double HeatEquation::analytical_solution(const double x) const {
    return  std::sin(2 * PI * x);
}

double HeatEquation::initial_condition(const double x) const {
    return 0.0;
}
