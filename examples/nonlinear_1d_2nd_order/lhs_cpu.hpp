#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"
#include "equation.cuh"

class NonlinearLHSCPU : public pmf::modules::LHS {
public:
    NonlinearLHSCPU(std::shared_ptr<NonlinearEquation> eqn) : m_eqn(eqn) {}

    NonlinearLHSCPU(uint gpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : pmf::modules::LHS(gpu_threads), m_eqn(eqn) {}

    NonlinearLHSCPU(uint gpu_threads, uint cpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : pmf::modules::LHS(gpu_threads, cpu_threads), m_eqn(eqn) {}

    virtual void run_host(pmf::Array& lhs, pmf::Array& v,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;

protected:
    std::shared_ptr<NonlinearEquation> m_eqn;
};
