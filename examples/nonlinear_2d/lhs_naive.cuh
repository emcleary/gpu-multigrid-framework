#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"

#include "equation.cuh"
#include "lhs_cpu.hpp"


class LHSNaive : public LHSCPU {
public:
    LHSNaive(std::shared_ptr<NonlinearEquation> eqn) : LHSCPU(eqn) {}

    LHSNaive(uint gpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : LHSCPU(gpu_threads, 1, eqn) {}

    LHSNaive(uint gpu_threads, uint cpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : LHSCPU(gpu_threads, cpu_threads, eqn) {}

    virtual void run_device(pmf::Array& lhs, pmf::Array& v,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
