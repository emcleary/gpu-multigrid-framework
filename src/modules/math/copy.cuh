#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/parallel.cuh"


namespace pmf {
namespace modules {

class Copy : public Parallel {
public:
    Copy() {}
    Copy(uint gpu_threads, uint cpu_threads) : Parallel(gpu_threads, cpu_threads) {}

    virtual void run_host(Array& a, Array& b);
    virtual void run_device(Array& a, Array& b);
};

} // namespace modules
} // namespace pmf
