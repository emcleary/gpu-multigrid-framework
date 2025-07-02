#pragma once

#include <cstdio>
#include <functional>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "timer.hpp"

#define cudaCheck(err) (cudaCheckDisplay(err, __FILE__, __LINE__))
#define cublasCheck(stat) (cublasCheckDisplay(stat, __FILE__, __LINE__))

void cudaCheckDisplay(cudaError_t err, const char *file, int line);
void cublasCheckDisplay(cublasStatus_t stat, const char *file, int line);

namespace gmf {

template <class T>
float measure_performance(std::function<T()> bound_function,
        int n_warmups, int n_repeats) {

    for (int i = 0; i < n_warmups; ++i)
        bound_function();

    TimerGPU timer;
    timer.start();
    for (int i = 0; i < n_repeats; ++i)
        bound_function();
    timer.stop();

    return timer.duration() / n_repeats;
}

} // namespace gmf
