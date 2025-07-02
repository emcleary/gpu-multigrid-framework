#pragma once

#include "grid.hpp"

namespace gmf {

struct Level {
public:
    Level(const int n, const bool on_gpu, const Grid& g)
            : on_gpu(on_gpu), solution(n), grid(g.get_xmin(), g.get_xmax(), n),
              rhs(n), temporary(n) {}
    const bool on_gpu;
    Array solution;
    Grid grid;
    Array rhs;
    Array temporary;
};

} // namespace gmf
