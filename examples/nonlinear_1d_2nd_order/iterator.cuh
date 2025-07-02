#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/boundary_conditions.hpp"
#include "equation.hpp"


class IteratorNonlinear : public gmf::modules::Iterator {
public:
    IteratorNonlinear(const size_t max_threads_per_block, std::shared_ptr<NonlinearEquation> eqn)
            : gmf::modules::Iterator(max_threads_per_block), m_eqn(eqn) {}

    virtual void run_host(gmf::Array& v, const gmf::Array& f, const gmf::modules::BoundaryConditions& bcs,
            const gmf::Grid& grid) override;

    virtual void run_device(gmf::Array& v, const gmf::Array& f, const gmf::modules::BoundaryConditions& bcs,
            const gmf::Grid& grid) override;

protected:
    std::shared_ptr<NonlinearEquation> m_eqn;
};
