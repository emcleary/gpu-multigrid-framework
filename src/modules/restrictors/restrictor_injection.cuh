#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/restrictor.cuh"


namespace gmf {
namespace modules {

class RestrictorInjection : public Restrictor {
public:
    RestrictorInjection(const int max_threads_per_block) : Restrictor(max_threads_per_block) {}

    void run_host(Array& fine, Array& coarse);
    void run_device(Array& fine, Array& coarse);
};

} // namespace modules
} // namespace gmf
