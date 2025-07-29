#pragma once

#include "solver.cuh"

#include <vector>

#include "array.hpp"
#include "grid.hpp"
#include "kernels.cuh"
#include "levels.hpp"

#include "modules/interfaces/equation.cuh"
#include "modules/interfaces/interpolator.cuh"
#include "modules/interfaces/iterator.cuh"
#include "modules/interfaces/lhs.cuh"
#include "modules/interfaces/norm.cuh"
#include "modules/interfaces/restrictor.cuh"


namespace pmf {

class SolverNonlinearFull : public Solver {
public:
    SolverNonlinearFull() {}

    virtual double restrict(const int lvl) override;
    virtual void correct(const int lvl) override;
};

} // namespace pmf
