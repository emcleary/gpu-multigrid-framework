#pragma once

#include "lhs_cpu.cuh"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class LHSNaive : public LHSCPU {
public:
    LHSNaive(const size_t max_threads_per_block) : LHSCPU(max_threads_per_block) {}

    virtual void run_device(gmf::Array& lhs, gmf::Array& v, const gmf::Array& f, const gmf::Grid& grid) override;
};
