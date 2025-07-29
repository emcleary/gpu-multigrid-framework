#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/restrictor.cuh"


namespace pmf {
namespace modules {

class RestrictorFullWeighting2D : public Restrictor {
public:
    RestrictorFullWeighting2D() {}
    RestrictorFullWeighting2D(uint gpu_threads, uint cpu_threads) : Restrictor(gpu_threads, cpu_threads) {}

    void run_host(Array& fine, Array& coarse, BoundaryConditions& bcs);
    void run_device(Array& fine, Array& coarse, BoundaryConditions& bcs);
};

} // namespace modules
} // namespace pmf
