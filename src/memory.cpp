#include "memory.hpp"

#include <cuda_runtime.h>

#include "utilities.hpp"

namespace gmf {

double* managed_allocator(const size_t N) {
    double* ptr = nullptr;
    cudaCheck(cudaMallocManaged((void**)&ptr, N * sizeof(double)));
    return ptr;
}

double* device_allocator(const size_t N) {
    double* ptr = nullptr;
    cudaCheck(cudaMalloc((void**)&ptr, N * sizeof(double)));
    return ptr;
}

double* host_allocator(const size_t N) {
    double* ptr = nullptr;
    cudaMallocHost((void**)&ptr, N * sizeof(double));
    return ptr;
}

void managed_deleter::operator()(double* ptr) {
    cudaCheck(cudaFree(ptr));
}

void device_deleter::operator()(double* ptr) {
    cudaCheck(cudaFree(ptr));
}

void host_deleter::operator()(double* ptr) {
    cudaCheck(cudaFreeHost(ptr));
}

} // namespace gmf
