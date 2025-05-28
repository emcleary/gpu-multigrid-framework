#pragma once

#include "iterator_cpu.cuh"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class IteratorAsync : public IteratorCPU {
public:
    IteratorAsync(const size_t max_threads_per_block) : IteratorCPU(max_threads_per_block) {}

    virtual void run_device(gmf::Array& v, const gmf::Array& f, const gmf::Grid& grid) override;
    virtual void run_error_device(gmf::Array& e, const gmf::Array& v, const gmf::Array& r, const gmf::Grid& grid) override;
};
