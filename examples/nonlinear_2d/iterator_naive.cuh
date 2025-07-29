#pragma once

#include "iterator_cpu.hpp"

#include "src/array.hpp"
#include "src/grid.hpp"

#include "equation.cuh"


class IteratorNaive : public IteratorCPU {
public:
    IteratorNaive(std::shared_ptr<NonlinearEquation> eqn)
            : IteratorCPU(eqn) {}

    IteratorNaive(uint gpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : IteratorCPU(gpu_threads, eqn) {}

    IteratorNaive(uint gpu_threads, uint cpu_threads, std::shared_ptr<NonlinearEquation> eqn)
            : IteratorCPU(gpu_threads, cpu_threads, eqn) {}

    virtual void run_device(pmf::Array& v, const pmf::Array& f,
            const pmf::modules::BoundaryConditions& bcs, const pmf::Grid& grid) override;
};
