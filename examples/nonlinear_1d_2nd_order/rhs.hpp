#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"

#include "equation.hpp"


class NonlinearRHS : public gmf::modules::RHS {
public:
    NonlinearRHS(std::shared_ptr<NonlinearEquation> eqn) : m_eqn(eqn) {}

    virtual void run(gmf::Array& rhs, const gmf::Grid& grid,
            const gmf::modules::BoundaryConditions& bcs) override;

private:
    std::shared_ptr<NonlinearEquation> m_eqn;
};
