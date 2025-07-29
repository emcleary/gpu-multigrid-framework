#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"

#include "equation.cuh"


class IteratorCPU : public pmf::modules::Iterator {
public:
    IteratorCPU(std::shared_ptr<NonlinearEquation> eqn) : m_eqn(eqn) {}

    IteratorCPU(uint gpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : pmf::modules::Iterator(gpu_threads), m_eqn(eqn) {}

    IteratorCPU(uint gpu_threads, uint cpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : pmf::modules::Iterator(gpu_threads, cpu_threads), m_eqn(eqn) {}

    virtual void run_host(pmf::Array& v, const pmf::Array& f,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;

protected:
    std::shared_ptr<NonlinearEquation> m_eqn;
};
