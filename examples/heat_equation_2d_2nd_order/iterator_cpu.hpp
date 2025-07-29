#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class IteratorCPU : public pmf::modules::Iterator {
public:
    IteratorCPU() {}
    IteratorCPU(uint gpu_threads, uint cpu_threads = 1)
            : pmf::modules::Iterator(gpu_threads, cpu_threads) {}

    virtual void run_host(pmf::Array& v, const pmf::Array& f,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
