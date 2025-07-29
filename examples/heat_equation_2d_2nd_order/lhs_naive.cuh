#pragma once

#include "lhs_cpu.hpp"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/interfaces/boundary_conditions.cuh"


class LHSNaive : public LHSCPU {
public:
    LHSNaive() {}

    LHSNaive(uint gpu_threads, uint cpu_threads = 1) : LHSCPU(gpu_threads, cpu_threads) {}

    virtual void run_device(pmf::Array& lhs, pmf::Array& v,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
