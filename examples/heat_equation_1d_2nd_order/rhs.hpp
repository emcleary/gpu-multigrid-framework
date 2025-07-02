#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class HeatEquationRHS : public gmf::modules::RHS {
public:
    HeatEquationRHS() {}

    virtual void run(gmf::Array& rhs, const gmf::Grid& grid,
            const gmf::modules::BoundaryConditions& bcs) override;
};
