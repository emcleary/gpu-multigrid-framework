#include <cuda_runtime.h>

#include "src/array.hpp"

using gmf::ArrayRaw;


__host__ __device__
inline double eval_gs(ArrayRaw& v, const ArrayRaw& f, const double h2, const int i) {
    return (v[i-1] + v[i+1] + h2 * f[i]) / 2;
}

__device__
inline double eval_gs(const double* const v, const double f, const double h2, const int i) {
    return (v[i-1] + v[i+1] + h2 * f) / 2;
}

__host__ __device__
inline double eval_lhs(ArrayRaw& v, const double h2, const int i) {
    return - (v[i-1] - 2 * v[i] +  v[i+1]) / h2;
}
