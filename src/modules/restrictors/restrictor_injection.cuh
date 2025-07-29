#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/restrictor.cuh"


namespace pmf {
namespace modules {

class RestrictorInjection : public Restrictor {
public:
    RestrictorInjection() {}
    RestrictorInjection(uint gpu_threads, uint cpu_threads) : Restrictor(gpu_threads, cpu_threads) {}

    void run_host(Array& fine, Array& coarse);
    void run_device(Array& fine, Array& coarse);
};

} // namespace modules
} // namespace pmf
