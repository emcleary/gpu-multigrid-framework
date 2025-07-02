#pragma once

#include "src/array.hpp"
#include "src/modules/interfaces/restrictor.cuh"


namespace gmf {
namespace modules {

class RestrictorFullWeighting : public Restrictor {
public:
    RestrictorFullWeighting(const int max_threads_per_block) : Restrictor(max_threads_per_block) {}

    void run_host(Array& fine, Array& coarse, BoundaryConditions& bcs);
    void run_device(Array& fine, Array& coarse, BoundaryConditions& bcs);
};

} // namespace modules
} // namespace gmf
