#pragma once

#include <cstddef> // size_t

namespace pmf {

double* managed_allocator(const size_t N);
double* device_allocator(const size_t N);
double* host_allocator(const size_t N);


struct managed_deleter {
    void operator()(double* ptr);
};

struct device_deleter {
    void operator()(double* ptr);
};

struct host_deleter {
    void operator()(double* ptr);
};

} // namespace pmf
