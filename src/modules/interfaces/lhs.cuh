#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"


namespace gmf {
namespace modules {

class LHS {
public:
    LHS(const size_t max_threads_per_block) : m_max_threads_per_block(max_threads_per_block) {}

    virtual void run_host(Array& lhs, Array& v, const Array& f, const Grid& grid) = 0;
    virtual void run_device(Array& lhs, Array& v, const Array& f, const Grid& grid) = 0;

protected:
    const size_t m_max_threads_per_block;
};

} // namespace modules
} // namespace gmf
