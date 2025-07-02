#pragma once

#include "iterator.cuh"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class IteratorAsync : public IteratorNonlinear {
public:
    IteratorAsync(const size_t max_threads_per_block, std::shared_ptr<NonlinearEquation> eqn)
            : IteratorNonlinear(max_threads_per_block, eqn) {}

    virtual void run_device(gmf::Array& v, const gmf::Array& f, const gmf::modules::BoundaryConditions& bcs, const gmf::Grid& grid) override;
};
