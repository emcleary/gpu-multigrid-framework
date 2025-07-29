#pragma once

#include "examples/heat_equation_1d_2nd_order/iterator_cpu.cuh"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class IteratorAsyncSMEM : public IteratorCPU {
public:
    IteratorAsyncSMEM(uint gpu_threads) : IteratorCPU(gpu_threads) {}

    virtual void run_device(pmf::Array& v, const pmf::Array& f,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
