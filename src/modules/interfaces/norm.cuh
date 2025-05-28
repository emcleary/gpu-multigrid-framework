#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"


namespace gmf {
namespace modules {

class Norm {
public:
    Norm() {}

    virtual double run_host(const Array& resid, const Grid& grid) = 0;
    virtual double run_device(const Array& resid, const Grid& grid) = 0;
};

} // namespace modules
} // namespace gmf
