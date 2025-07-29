#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/parallel.cuh"


namespace pmf {
namespace modules {

class Sub : public Parallel {
public:
    Sub() {}
    Sub(uint gpu_threads, uint cpu_threads) : Parallel(gpu_threads, cpu_threads) {}

    virtual void run_host(Array& a, Array& b, Array& c);
    virtual void run_device(Array& a, Array& b, Array& c);
};

} // namespace modules
} // namespace pmf
