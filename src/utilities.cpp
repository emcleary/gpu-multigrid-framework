#include "utilities.hpp"


void cudaCheckDisplay(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d\n%s\n", file, line,
               cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void cublasCheckDisplay(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("[CUBLAS STATUS] at file %s:%d\n%s: %s\n", file, line,
               cublasGetStatusName(stat), cublasGetStatusString(stat));
        exit(EXIT_FAILURE);
    }
}

