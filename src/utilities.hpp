#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudaCheck(err) (cudaCheckDisplay(err, __FILE__, __LINE__))
#define cublasCheck(stat) (cublasCheckDisplay(stat, __FILE__, __LINE__))

void cudaCheckDisplay(cudaError_t err, const char *file, int line);
void cublasCheckDisplay(cublasStatus_t stat, const char *file, int line);
