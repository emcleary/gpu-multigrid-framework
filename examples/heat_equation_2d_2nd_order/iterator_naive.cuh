#pragma once

#include "iterator_cpu.hpp"

#include "src/array.hpp"
#include "src/grid.hpp"


class IteratorNaive : public IteratorCPU {
public:
    IteratorNaive() {}
    IteratorNaive(uint gpu_threads, uint cpu_threads = 1) : IteratorCPU(gpu_threads, cpu_threads) {}

    virtual void run_device(pmf::Array& v, const pmf::Array& f,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
