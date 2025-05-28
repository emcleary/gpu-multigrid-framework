#include "iterator.cuh"

#include <cuda_runtime.h>

#include "src/array.hpp"

using gmf::Array;
using gmf::ArrayRaw;
using gmf::Grid;


__host__ __device__
inline double eval(const ArrayRaw& v, const ArrayRaw& f, const double h, const double gamma, const int i) {
    double num = 2 * (h * h * f[i] + v[i+1] + v[i-1]);
    double denom = 4 + h * gamma * (v[i+1] - v[i-1]);
    return num / denom;
}

__host__ __device__
inline double eval_error(const ArrayRaw& e, const ArrayRaw& v, const ArrayRaw& r, const double h, const double gamma, const int i) {
    double num = 2 * (h * h * r[i] + e[i+1] + e[i-1]) - h * gamma * v[i] * (e[i+1] - e[i-1]);
    double denom = 4 + h * gamma * (v[i+1] - v[i-1] + e[i+1] - e[i-1]);
    return num / denom;
}

namespace iterator {
__global__
void kernel(ArrayRaw v, const ArrayRaw f, const double h, const double gamma, const int color) {
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < v.size() - 1)
        v[idx] = eval(v, f, h, gamma, idx);
}

__global__
void kernel_error(ArrayRaw e, const ArrayRaw v, const ArrayRaw r, const double h, const double gamma, const int color) {
    int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x) + color;

    // must skip boundaries for Dirichlet BCs
    if (0 < idx && idx < v.size() - 1)
        e[idx] = eval_error(e, v, r, h, gamma, idx);
}
}


void IteratorNonlinear::run_host(Array& v, const Array& f, const Grid& grid) {
    const int n_pts = v.size();
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    // Red-Black Gauss Seidel -- parallel and converges faster
    for (int i = 1; i < n_pts - 1; i += 2)
        v[i] = eval(v, f, h, gamma, i);
    for (int i = 2; i < n_pts - 1; i += 2)
        v[i] = eval(v, f, h, gamma, i);
}

void IteratorNonlinear::run_device(Array& v, const Array& f, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, v.size() / 2);
    const int blocksPerGrid = (v.size() / 2 + threadsPerBlock - 1) / threadsPerBlock;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    // Red-Black Gauss Seidel -- parallel and converges faster
    iterator::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h, gamma, 1);
    iterator::kernel<<<blocksPerGrid, threadsPerBlock>>>(v, f, h, gamma, 2);
}


void IteratorNonlinear::run_error_host(Array& e, const Array& v, const Array& r, const Grid& grid) {
    const int n_pts = e.size();
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    // Red-Black Gauss Seidel -- parallel and converges faster
    for (int i = 1; i < n_pts - 1; i += 2)
        e[i] = eval_error(e, v, r, h, gamma, i);
    for (int i = 2; i < n_pts - 1; i += 2)
        e[i] = eval_error(e, v, r, h, gamma, i);
}

void IteratorNonlinear::run_error_device(Array& e, const Array& v, const Array& r, const Grid& grid) {
    const int threadsPerBlock = std::min(m_max_threads_per_block, e.size() / 2);
    const int blocksPerGrid = (e.size() / 2 + threadsPerBlock - 1) / threadsPerBlock;
    const double h = grid.get_cell_width();
    const double gamma = m_eqn->get_gamma();

    // Red-Black Gauss Seidel -- parallel and converges faster
    iterator::kernel_error<<<blocksPerGrid, threadsPerBlock>>>(e, v, r, h, gamma, 1);
    iterator::kernel_error<<<blocksPerGrid, threadsPerBlock>>>(e, v, r, h, gamma, 2);
}
