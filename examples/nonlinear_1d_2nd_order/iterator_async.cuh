#pragma once

#include "iterator.cuh"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"


class IteratorAsync : public IteratorNonlinear {
public:
    IteratorAsync(std::shared_ptr<NonlinearEquation> eqn) : IteratorNonlinear(eqn) {}

    IteratorAsync(uint gpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : IteratorNonlinear(gpu_threads, eqn) {}

    IteratorAsync(uint gpu_threads, uint cpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : IteratorNonlinear(gpu_threads, cpu_threads, eqn) {}

    virtual void run_device(pmf::Array& v, const pmf::Array& f,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
