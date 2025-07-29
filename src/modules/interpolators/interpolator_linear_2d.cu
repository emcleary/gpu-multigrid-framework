#include "interpolator_linear_2d.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"

namespace pmf {
namespace modules {

namespace interpolator_linear_2d {
__global__
void kernel(ArrayRaw coarse, ArrayRaw fine, BoundaryConditions bcs) {
    const int nh = fine.get_nrows() - 1;
    const int n2h = coarse.get_nrows() - 1;

    int ci = threadIdx.x + blockDim.x * blockIdx.x;
    int cj = threadIdx.y + blockDim.y * blockIdx.y;

    if (ci < n2h && cj < n2h) {
        int fi = 2 * ci;
        int fj = 2 * cj;
        fine(fi, fj) = coarse(ci, cj);
        fine(fi+1, fj) = (coarse(ci, cj) + coarse(ci+1, cj)) / 2;
        fine(fi, fj+1) = (coarse(ci, cj) + coarse(ci, cj+1)) / 2;
        fine(fi+1, fj+1) = (coarse(ci, cj) + coarse(ci+1, cj)
                + coarse(ci, cj+1) + coarse(ci+1, cj+1)) / 4;
    }
}

__global__
void boundaries(ArrayRaw coarse, ArrayRaw fine, BoundaryConditions bcs) {
    const int nh = fine.get_nrows() - 1;
    const int n2h = coarse.get_nrows() - 1;

    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int f = 2 * c;

    if (0 <= c && c < n2h) {
        if (bcs.is_periodic_x())
            fine(nh, f) = fine(0, f);
            
        if (bcs.is_periodic_y())
            fine(f, nh) = fine(f, 0);

        if (bcs.is_north_neumann()) {
            fine(f, nh) = coarse(c, n2h);
            fine(f+1, nh) = (coarse(c, n2h) + coarse(c+1, n2h)) / 2;
        }

        if (bcs.is_south_neumann()) {
            fine(f, 0) = coarse(c, 0);
            fine(f+1, 0) = (coarse(c, 0) + coarse(c+1, 0)) / 2;
        }

        if (bcs.is_east_neumann()) {
            fine(nh, f) = coarse(n2h, c);
            fine(nh, f+1) = (coarse(n2h, c) + coarse(n2h, c+1)) / 2;
        }

        if (bcs.is_west_neumann()) {
            fine(0, f) = coarse(0, c);
            fine(0, f+1) = (coarse(0, c) + coarse(0, c+1)) / 2;
        }

        if (bcs.is_north_dirichlet()) {
            fine(f, nh) = 0;
            fine(f+1, nh) = 0;
        }

        if (bcs.is_south_dirichlet()) {
            fine(f, 0) = 0;
            fine(f+1, 0) = 0;
        }

        if (bcs.is_east_dirichlet()) {
            fine(nh, f) = 0;
            fine(nh, f+1) = 0;
        }

        if (bcs.is_west_dirichlet()) {
            fine(0, f) = 0;
            fine(0, f+1) = 0;
        }

    }
    
}

__global__
void boundaries_dirichlet(ArrayRaw fine, BoundaryConditions bcs) {
    const int nh = fine.get_nrows() - 1;
    const int f = threadIdx.x + blockDim.x * blockIdx.x;

    if (0 <= f && f <= nh) {
        if (bcs.is_north_dirichlet())
            fine(f, nh) = 0;

        if (bcs.is_south_dirichlet())
            fine(f, 0) = 0;

        if (bcs.is_east_dirichlet())
            fine(nh, f) = 0;

        if (bcs.is_west_dirichlet())
            fine(0, f) = 0;
    }
}

__global__
void corners(ArrayRaw coarse, ArrayRaw fine, BoundaryConditions bcs) {
    const int nh = fine.get_ncols() - 1;
    const int n2h = coarse.get_ncols() - 1;

    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        fine(0, nh) = fine(0, 0);
        fine(nh, 0) = fine(0, 0);
        fine(nh, nh) = fine(0, 0);
    }

    if (bcs.is_east_neumann() || bcs.is_north_neumann())
        fine(nh, nh) = coarse(n2h, n2h);

    if (bcs.is_south_neumann())
        fine(nh, 0) = coarse(n2h, 0);

    if (bcs.is_west_neumann())
        fine(nh, nh) = coarse(n2h, n2h);

    if (bcs.is_east_dirichlet() || bcs.is_north_dirichlet())
        fine(nh, nh) = 0;

    if (bcs.is_east_dirichlet() || bcs.is_south_dirichlet())
        fine(nh, 0) = 0;

    if (bcs.is_west_dirichlet() || bcs.is_south_dirichlet())
        fine(0, 0) = 0;

    if (bcs.is_west_dirichlet() || bcs.is_north_dirichlet())
        fine(0, nh) = 0;
}
}


void InterpolatorLinear2D::run_host(Array& coarse, Array& fine, BoundaryConditions& bcs) {
    const int nh = fine.get_nrows() - 1;
    assert(nh == fine.get_ncols() - 1);

    const int n2h = coarse.get_nrows() - 1;
    assert(n2h == coarse.get_ncols() - 1);

    // TODO: set number of threads as class member
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int ci = 0; ci < n2h; ++ci) {
        const int fi = ci * 2;
        for (int cj = 0; cj < n2h; ++cj) {
            const int fj = cj * 2;
            fine(fi, fj) = coarse(ci, cj);
            fine(fi+1, fj) = (coarse(ci, cj) + coarse(ci+1, cj)) / 2;
            fine(fi, fj+1) = (coarse(ci, cj) + coarse(ci, cj+1)) / 2;
            fine(fi+1, fj+1) = (coarse(ci, cj) + coarse(ci+1, cj)
                    + coarse(ci, cj+1) + coarse(ci+1, cj+1)) / 4;
        }        
    }

    if (bcs.is_periodic_x()) {
#pragma omp parallel for
        for (int fj = 1; fj < nh; ++fj)
            fine(nh, fj) = fine(0, fj);
    }

    if (bcs.is_periodic_y()) {
#pragma omp parallel for
        for (int fi = 1; fi < nh; ++fi)
            fine(fi, nh) = fine(fi, 0);
    }

    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        fine(nh, 0) = fine(0, 0);
        fine(0, nh) = fine(0, 0);
        fine(nh, nh) = fine(0, 0);
    }

    if (bcs.is_north_neumann()) {
#pragma omp parallel for
        for (int ci = 0; ci < n2h; ++ci) {
            const int fi = ci * 2;
            fine(fi, nh) = coarse(ci, n2h);
            fine(fi+1, nh) = (coarse(ci, n2h) + coarse(ci+1, n2h)) / 2;
        }
        fine(nh, nh) = coarse(n2h, n2h);
    }

    if (bcs.is_south_neumann()) {
#pragma omp parallel for
        for (int ci = 0; ci < n2h; ++ci) {
            const int fi = ci * 2;
            fine(fi, 0) = coarse(ci, 0);
            fine(fi+1, 0) = (coarse(ci, 0) + coarse(ci+1, 0)) / 2;
        }
        fine(nh, 0) = coarse(n2h, 0);
    }

    if (bcs.is_east_neumann()) {
#pragma omp parallel for
        for (int cj = 0; cj < n2h; ++cj) {
            const int fj = cj * 2;
            fine(nh, fj) = coarse(n2h, cj);
            fine(nh, fj+1) = (coarse(n2h, cj) + coarse(n2h, cj+1)) / 2;
        }
        fine(nh, nh) = coarse(n2h, n2h);
    }

    if (bcs.is_west_neumann()) {
#pragma omp parallel for
        for (int cj = 0; cj < n2h; ++cj) {
            const int fj = cj * 2;
            fine(0, fj) = coarse(0, cj);
            fine(0, fj+1) = (coarse(0, cj) + coarse(0, cj+1)) / 2;
        }
        fine(0, nh) = coarse(0, n2h);
    }

    // Only interpolate errors, which should be 0 for any Dirichlet BC
    // Important to overwrite any corner set by a Neumann BC
    if (bcs.is_north_dirichlet())
#pragma omp parallel for
        for (int i = 0; i <= nh; ++i)
            fine(i, nh) = 0;

    if (bcs.is_south_dirichlet())
#pragma omp parallel for
        for (int i = 0; i <= nh; ++i)
            fine(i, 0) = 0;

    if (bcs.is_east_dirichlet())
#pragma omp parallel for
        for (int j = 0; j <= nh; ++j)
            fine(nh, j) = 0;

    if (bcs.is_west_dirichlet())
#pragma omp parallel for
        for (int j = 0; j <= nh; ++j)
            fine(0, j) = 0;
}

void InterpolatorLinear2D::run_device(Array& coarse, Array& fine, BoundaryConditions& bcs) {
    // i.e. will not work for alternating directions
    assert(coarse.get_nrows() == coarse.get_ncols());
    assert(fine.get_nrows() == fine.get_ncols());

    // TODO: constrain threads better (for my gpu, 1024 threads/block max --> 32x32)
    uint threads = std::min(32U, static_cast<uint>(coarse.get_ncols()) - 1);
    uint blocks = (coarse.get_nrows() + threads - 1) / threads;
    dim3 threads_2d(threads, threads);
    dim3 blocks_2d(blocks, blocks);

    const uint n_fine = fine.get_ncols();
    const uint threads_1d_fine = std::min(m_max_threads_per_block, n_fine);
    const uint blocks_1d_fine = (n_fine + threads_1d_fine - 1) / threads_1d_fine;

    const uint n_coarse = coarse.get_ncols();
    const uint threads_1d_coarse = std::min(m_max_threads_per_block, n_coarse);
    const uint blocks_1d_coarse = (n_coarse + threads_1d_coarse - 1) / threads_1d_coarse;

    interpolator_linear_2d::kernel<<<blocks_2d, threads_2d>>>(coarse, fine, bcs);
    interpolator_linear_2d::boundaries<<<blocks_1d_coarse, threads_1d_coarse>>>(coarse, fine, bcs);
    interpolator_linear_2d::boundaries_dirichlet<<<blocks_1d_fine, threads_1d_fine>>>(fine, bcs);
    interpolator_linear_2d::corners<<<1, 1>>>(coarse, fine, bcs);
}

} // namespace modules
} // namespace pmf
