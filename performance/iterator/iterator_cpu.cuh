#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class IteratorCPU : public gmf::modules::Iterator {
public:
    IteratorCPU(const size_t max_threads_per_block) : gmf::modules::Iterator(max_threads_per_block) {}

    virtual void run_host(gmf::Array& v, const gmf::Array& f, const gmf::Grid& grid) override;
    virtual void run_error_host(gmf::Array& e, const gmf::Array& v, const gmf::Array& r, const gmf::Grid& grid) override;
};
