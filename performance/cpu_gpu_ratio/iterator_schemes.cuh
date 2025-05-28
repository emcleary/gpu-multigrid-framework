#include <cuda_runtime.h>

#include "src/array.hpp"
#include "src/grid.hpp"


using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


__host__ __device__
inline double eval2(const ArrayRaw& v, const ArrayRaw& f, const double h2, const int i) {
    return (v[i-1] + v[i+1] + h2 * f[i]) / 2;
}

__host__ __device__
inline double eval2(const double* const v, const double f, const double h2, const int i) {
    return (v[i-1] + v[i+1] + h2 * f) / 2;
}

__host__ __device__
inline double eval4(const ArrayRaw& v, const ArrayRaw& f, const double h2, const int i) {
    const double c0 = -1./12;
    const double c1 = 4./3;
    const double c2 = 2./5;
    const double c3 = 4./3;
    const double c4 = -1./12;
    return c2 * (c0 * v[i-2] + c1 * v[i-1] + c3 * v[i+1] + c4 * v[i+2] + h2 * f[i]);
}

__host__ __device__
inline double eval4(const double* const v, const double f, const double h2, const int i) {
    const double c0 = -1./12;
    const double c1 = 4./3;
    const double c2 = 2./5;
    const double c3 = 4./3;
    const double c4 = -1./12;
    return c2 * (c0 * v[i-2] + c1 * v[i-1] + c3 * v[i+1] + c4 * v[i+2] + h2 * f);
}

__host__ __device__
inline double eval4left(const ArrayRaw& v, const ArrayRaw& f, const double h2, const int i) {
    const double c0 =  5./6;
    const double c1 = 4./5;
    const double c2 = -1./3;
    const double c3 = 7./6;
    const double c4 = -1./2;
    const double c5 = 1./12;
    return c1 * (c0 * v[i-1] + c2 * v[i+1] + c3 * v[i+2] + c4 * v[i+3] + c5 * v[i+4] + h2 * f[i]);
}

__host__ __device__
inline double eval4left(const double* const v, const double f, const double h2, const int i) {
    const double c0 =  5./6;
    const double c1 = 4./5;
    const double c2 = -1./3;
    const double c3 = 7./6;
    const double c4 = -1./2;
    const double c5 = 1./12;
    return c1 * (c0 * v[i-1] + c2 * v[i+1] + c3 * v[i+2] + c4 * v[i+3] + c5 * v[i+4] + h2 * f);
}

__host__ __device__
inline double eval4right(const ArrayRaw& v, const ArrayRaw& f, const double h2, const int i) {
    const double c0 =  5./6;
    const double c1 = 4./5;
    const double c2 = -1./3;
    const double c3 = 7./6;
    const double c4 = -1./2;
    const double c5 = 1./12;
    return c1 * (c0 * v[i+1] + c2 * v[i-1] + c3 * v[i-2] + c4 * v[i-3] + c5 * v[i-4] + h2 * f[i]);
}

__host__ __device__
inline double eval4right(const double* const v, const double f, const double h2, const int i) {
    const double c0 =  5./6;
    const double c1 = 4./5;
    const double c2 = -1./3;
    const double c3 = 7./6;
    const double c4 = -1./2;
    const double c5 = 1./12;
    return c1 * (c0 * v[i+1] + c2 * v[i-1] + c3 * v[i-2] + c4 * v[i-3] + c5 * v[i-4] + h2 * f);
}
