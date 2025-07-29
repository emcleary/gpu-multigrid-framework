#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"


class LHSCPU : public pmf::modules::LHS {
public:
    LHSCPU() {}
    LHSCPU(uint gpu_threads, uint cpu_threads = 1)
            : pmf::modules::LHS(gpu_threads, cpu_threads) {}

    virtual void run_host(pmf::Array& lhs, pmf::Array& v,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
