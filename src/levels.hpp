#pragma once

#include "grid.hpp"

namespace gmf {

struct Level {
public:
    Level(const int n, const bool on_gpu, const Grid& g)
            : on_gpu(on_gpu), solution(n), grid(g.get_xmin(), g.get_xmax(), n) {}
    const bool on_gpu;
    Array solution;
    Grid grid;
};


struct LevelLinear : public Level {
    LevelLinear(const int n, const bool on_gpu, const Grid& g)
            : Level(n, on_gpu, g), rhs(n), temporary(n) {}

    Array rhs;
    Array temporary;
};


struct LevelNonlinearFull : public Level {
    LevelNonlinearFull(const int n, const bool on_gpu, const Grid& g)
            : Level(n, on_gpu, g), rhs(n), temporary(n) {}

    Array rhs;
    Array temporary;
};


struct LevelNonlinearError : public Level {
    LevelNonlinearError(const int n, const bool on_gpu, const Grid& g)
            : Level(n, on_gpu, g), residual(n), error(n) {}

    Array residual;
    Array error;
};

} // namespace gmf
