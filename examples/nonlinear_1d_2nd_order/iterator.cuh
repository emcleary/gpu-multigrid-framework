#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"
#include "equation.cuh"


class IteratorNonlinear : public pmf::modules::Iterator {
public:
    IteratorNonlinear(std::shared_ptr<NonlinearEquation> eqn) : m_eqn(eqn) {}

    IteratorNonlinear(uint gpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : pmf::modules::Iterator(gpu_threads, 1), m_eqn(eqn) {}

    IteratorNonlinear(uint gpu_threads, uint cpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : pmf::modules::Iterator(gpu_threads, cpu_threads), m_eqn(eqn) {}

    virtual void run_host(pmf::Array& v, const pmf::Array& f, const pmf::modules::BoundaryConditions& bcs,
            const pmf::Grid& grid) override;

    virtual void run_device(pmf::Array& v, const pmf::Array& f, const pmf::modules::BoundaryConditions& bcs,
            const pmf::Grid& grid) override;

protected:
    std::shared_ptr<NonlinearEquation> m_eqn;
};
