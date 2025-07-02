#pragma once

#include "solver.cuh"

#include <vector>

#include "array.hpp"
#include "grid.hpp"
#include "kernels.cuh"
#include "levels.hpp"

#include "modules/interfaces/equation.hpp"
#include "modules/interfaces/interpolator.cuh"
#include "modules/interfaces/iterator.cuh"
#include "modules/interfaces/lhs.cuh"
#include "modules/interfaces/norm.cuh"
#include "modules/interfaces/restrictor.cuh"
#include "modules/interfaces/rhs.hpp"


namespace gmf {

class SolverLinear : public Solver {
public:
    SolverLinear() {}

    virtual void restrict(const int lvl) override;
    virtual void correct(const int lvl) override;
};

} // namespace gmf
