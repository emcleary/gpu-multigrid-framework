#pragma once

#include <iostream>

#include "src/array.hpp"
#include "src/grid.hpp"

namespace gmf {
namespace modules {

class Iterator {
public:
    Iterator(const size_t max_threads_per_block) : m_max_threads_per_block(max_threads_per_block) {}

    virtual void run_host(Array& v, const Array& f, const Grid& grid) = 0;

    virtual void run_device(Array& v, const Array& f, const Grid& grid) = 0;

    virtual void run_error_host(Array& e, const Array& v, const Array& r, const Grid& grid) {
        std::cerr << "Error iterator equations not implemented.\n";
        exit(EXIT_FAILURE);
    };

    virtual void run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
        std::cerr << "Error iterator equations not implemented.\n";
        exit(EXIT_FAILURE);
    };

protected:
    const size_t m_max_threads_per_block;
};

} // namespace modules
} // namespace gmf
