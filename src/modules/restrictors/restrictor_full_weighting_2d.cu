#include "restrictor_full_weighting_2d.cuh"

#include <cuda_runtime.h>
#include <omp.h>

#include "src/array.hpp"


namespace pmf {
namespace modules {

namespace restrictor_full_weighting_2d {

__global__
void kernel(ArrayRaw fine, ArrayRaw coarse, BoundaryConditions bcs) {
    // NB: only works when the number of rows and columns are the same
    const int nh = fine.get_nrows() - 1;
    const int n2h = coarse.get_nrows() - 1;

    int ci = threadIdx.x + blockDim.x * blockIdx.x;
    int cj = threadIdx.y + blockDim.y * blockIdx.y;

    if (0 < ci && ci < n2h && 0 < cj && cj < n2h) {
        int fi = 2 * ci;
        int fj = 2 * cj;
        coarse(ci, cj) = (4 * fine(fi, fj)
                + 2 * (fine(fi-1, fj) + fine(fi+1, fj) + fine(fi, fj-1) + fine(fi, fj+1))
                + fine(fi-1, fj-1) + fine(fi+1, fj-1) + fine(fi+1, fj+1) + fine(fi-1, fj+1)
        ) / 16;
    }
}

__global__
void boundaries(ArrayRaw fine, ArrayRaw coarse, BoundaryConditions bcs) {
    const int nh = fine.get_nrows() - 1;
    const int n2h = coarse.get_nrows() - 1;

    const int c = threadIdx.x + blockDim.x * blockIdx.x;
    const int f = 2 * c;

    if (0 < c && c < n2h) {
        if (bcs.is_periodic_x()) {
            coarse(0, c) = (4 * fine(0, f)
                    + 2 * (fine(nh-1, f) + fine(1, f) + fine(0, f-1) + fine(0, f+1))
                    + fine(nh-1, f-1) + fine(1, f-1) + fine(1, f+1) + fine(nh-1, f+1)
            ) / 16;
            coarse(n2h, c) = coarse(0, c);
        }
            
        if (bcs.is_periodic_y()) {
            coarse(c, 0) = (4 * fine(f, 0)
                    + 2 * (fine(f, nh-1) + fine(f, 1) + fine(f-1, 0) + fine(f+1, 0))
                    + fine(f-1, nh-1) + fine(f-1, 1) + fine(f+1, 1) + fine(f+1, nh-1)
            ) / 16;
            coarse(c, n2h) = coarse(c, 0);
        }

        if (bcs.is_north_neumann()) {
            coarse(c, n2h) = (2 * (fine(f, nh) + fine(f, nh-1))
                    + fine(f-1, nh) + fine(f-1, nh-1)
                    + fine(f+1, nh) + fine(f+1, nh-1)) / 8;
        }

        if (bcs.is_south_neumann()) {
            coarse(c, 0) = (2 * (fine(f, 0) + fine(f, 1))
                    + fine(f-1, 0) + fine(f-1, 1)
                    + fine(f+1, 0) + fine(f+1, 1)) / 8;
        }

        if (bcs.is_east_neumann()) {
            coarse(n2h, c) = (2 * (fine(nh, f) + fine(nh-1, f))
                    + fine(nh, f-1) + fine(nh-1, f-1)
                    + fine(nh, f+1) + fine(nh-1, f+1)) / 8;
        }

        if (bcs.is_west_neumann()) {
            coarse(0, c) = (2 * (fine(0, f) + fine(1, f))
                    + fine(0, f-1) + fine(1, f-1)
                    + fine(0, f+1) + fine(1, f+1)) / 8;
        }

        if (bcs.is_north_dirichlet()) {
            coarse(c, n2h) = fine(f, nh);
        }

        if (bcs.is_south_dirichlet()) {
            coarse(c, 0) = fine(f, 0);
        }

        if (bcs.is_east_dirichlet()) {
            coarse(n2h, c) = fine(nh, f);
        }

        if (bcs.is_west_dirichlet()) {
            coarse(0, c) = fine(0, f);
        }

    }
    
}

__global__
void corners(ArrayRaw fine, ArrayRaw coarse, BoundaryConditions bcs) {
    const int nh = fine.get_ncols() - 1;
    const int n2h = coarse.get_ncols() - 1;

    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        coarse(0, 0) = (4 * fine(0, 0)
                + 2 * (fine(nh-1, 0) + fine(1, 0) + fine(0, nh-1) + fine(0, 1))
                + fine(nh-1, nh-1) + fine(1, nh-1) + fine(1, 1) + fine(nh-1, 1)
        ) / 16;
        coarse(n2h, 0) = coarse(0, 0);
        coarse(0, n2h) = coarse(0, 0);
        coarse(n2h, n2h) = coarse(0, 0);
    }

    if (bcs.is_periodic_x()) {
        if (bcs.is_north_neumann()) {
            coarse(0, n2h) = (2 * (fine(0, nh) + fine(0, nh-1))
                    + fine(nh-1, nh) + fine(nh-1, nh-1)
                    + fine(1, nh) + fine(1, nh-1)) / 8;
            coarse(n2h, n2h) = coarse(0, n2h);
        }

        if (bcs.is_south_neumann()) {
            coarse(0, 0) = (2 * (fine(0, 0) + fine(0, 1))
                    + fine(nh-1, 0) + fine(nh-1, 1)
                    + fine(1, 0) + fine(1, 1)) / 8;
            coarse(n2h, 0) = coarse(0, 0);
        }
    }

    if (bcs.is_periodic_y()) {
        if (bcs.is_east_neumann()) {
            coarse(n2h, 0) = (2 * (fine(nh, 0) + fine(nh-1, 0))
                    + fine(nh, nh-1) + fine(nh-1, nh-1)
                    + fine(nh, 1) + fine(nh-1, 1)) / 8;
            coarse(n2h, n2h) = coarse(n2h, 0);
        }

        if (bcs.is_west_neumann()) {
            coarse(0, 0) = (2 * (fine(0, 0) + fine(1, 0))
                    + fine(0, nh-1) + fine(1, nh-1)
                    + fine(0, 1) + fine(1, 1)) / 8;
            coarse(0, n2h) = coarse(0, 0);
        }
    }

    if (bcs.is_east_neumann() && bcs.is_north_neumann())
        coarse(n2h, n2h) = (fine(nh, nh) + fine(nh-1, nh) + fine(nh, nh-1) + fine(nh-1, nh-1)) / 4;

    if (bcs.is_east_neumann() && bcs.is_south_neumann())
        coarse(n2h, 0) = (fine(nh, 0) + fine(nh-1, 0) + fine(nh, 1) + fine(nh-1, 1)) / 4;

    if (bcs.is_west_neumann() && bcs.is_south_neumann())
        coarse(0, 0) = (fine(0, 0) + fine(1, 0) + fine(0, 1) + fine(1, 1)) / 4;

    if (bcs.is_west_neumann() && bcs.is_north_neumann())
        coarse(0, n2h) = (fine(0, nh) + fine(1, nh) + fine(0, nh-1) + fine(1, nh-1)) / 4;

    if (bcs.is_east_dirichlet() || bcs.is_north_dirichlet())
        coarse(n2h, n2h) = fine(nh, nh);

    if (bcs.is_east_dirichlet() || bcs.is_south_dirichlet())
        coarse(n2h, 0) = fine(nh, 0);

    if (bcs.is_west_dirichlet() || bcs.is_south_dirichlet())
        coarse(0, 0) = fine(0, 0);

    if (bcs.is_west_dirichlet() || bcs.is_north_dirichlet())
        coarse(0, n2h) = fine(0, nh);
}
} // namespace restrictor_full_weighting_2d


void RestrictorFullWeighting2D::run_host(Array& fine, Array& coarse, BoundaryConditions& bcs) {
    const int nh = fine.get_nrows() - 1;
    assert(nh == fine.get_ncols() - 1);

    const int n2h = coarse.get_nrows() - 1;
    assert(n2h == coarse.get_ncols() - 1);

    // TODO: set number of threads as class member
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int ci = 1; ci < n2h; ++ci) {
        const int fi = 2 * ci;
        for (int cj = 1; cj < n2h; ++cj) {
            const int fj = cj * 2;
            coarse(ci, cj) = (4 * fine(fi, fj)
                    + 2 * (fine(fi-1, fj) + fine(fi+1, fj) + fine(fi, fj-1) + fine(fi, fj+1))
                    + fine(fi-1, fj-1) + fine(fi+1, fj-1) + fine(fi+1, fj+1) + fine(fi-1, fj+1)
            ) / 16;
        }
    }

    // Periodic
    if (bcs.is_periodic_x()) {
#pragma omp parallel for
        for (int cj = 1; cj < n2h; ++cj) {
            const int fj = cj * 2;
            coarse(0, cj) = (4 * fine(0, fj)
                    + 2 * (fine(nh-1, fj) + fine(1, fj) + fine(0, fj-1) + fine(0, fj+1))
                    + fine(nh-1, fj-1) + fine(1, fj-1) + fine(1, fj+1) + fine(nh-1, fj+1)
            ) / 16;
            coarse(n2h, cj) = coarse(0, cj);
        }
    }

    if (bcs.is_periodic_y()) {
#pragma omp parallel for
        for (int ci = 1; ci < n2h; ++ci) {
            const int fi = ci * 2;
            coarse(ci, 0) = (4 * fine(fi, 0)
                    + 2 * (fine(fi, nh-1) + fine(fi, 1) + fine(fi-1, 0) + fine(fi+1, 0))
                    + fine(fi-1, nh-1) + fine(fi-1, 1) + fine(fi+1, 1) + fine(fi+1, nh-1)
            ) / 16;
            coarse(ci, n2h) = coarse(ci, 0);
        }
    }

    // Neumann
    if (bcs.is_north_neumann()) {
        if (bcs.is_periodic_x()) {
            coarse(0, n2h) = (2 * (fine(0, nh) + fine(0, nh-1))
                    + fine(nh-1, nh) + fine(nh-1, nh-1)
                    + fine(1, nh) + fine(1, nh-1)) / 8;
            coarse(n2h, n2h) = coarse(0, n2h);
        }

#pragma omp parallel for
        for (int ci = 1; ci < n2h; ++ci) {
            const int fi = ci * 2;
            coarse(ci, n2h) = (2 * (fine(fi, nh) + fine(fi, nh-1))
                    + fine(fi-1, nh) + fine(fi-1, nh-1)
                    + fine(fi+1, nh) + fine(fi+1, nh-1)) / 8;
        }
    }

    if (bcs.is_south_neumann()) {
        if (bcs.is_periodic_x()) {
            coarse(0, 0) = (2 * (fine(0, 0) + fine(0, 1))
                    + fine(nh-1, 0) + fine(nh-1, 1)
                    + fine(1, 0) + fine(1, 1)) / 8;
            coarse(n2h, 0) = coarse(0, 0);
        }

#pragma omp parallel for
        for (int ci = 1; ci < n2h; ++ci) {
            const int fi = ci * 2;
            coarse(ci, 0) = (2 * (fine(fi, 0) + fine(fi, 1))
                    + fine(fi-1, 0) + fine(fi-1, 1)
                    + fine(fi+1, 0) + fine(fi+1, 1)) / 8;
        }
    }

    if (bcs.is_east_neumann()) {
        if (bcs.is_periodic_y()) {
            coarse(n2h, 0) = (2 * (fine(nh, 0) + fine(nh-1, 0))
                    + fine(nh, nh-1) + fine(nh-1, nh-1)
                    + fine(nh, 1) + fine(nh-1, 1)) / 8;
            coarse(n2h, n2h) = coarse(n2h, 0);
        }

#pragma omp parallel for
        for (int cj = 1; cj < n2h; ++cj) {
            const int fj = cj * 2;
            coarse(n2h, cj) = (2 * (fine(nh, fj) + fine(nh-1, fj))
                    + fine(nh, fj-1) + fine(nh-1, fj-1)
                    + fine(nh, fj+1) + fine(nh-1, fj+1)) / 8;
        }
    }

    if (bcs.is_west_neumann()) {
        if (bcs.is_periodic_y()) {
            coarse(0, 0) = (4 * (fine(0, 0) + fine(1, 0))
                    + 2 * (fine(0, nh-1) + fine(1, nh-1) + fine(0, 1) + fine(1, 1))) / 16;
            coarse(0, n2h) = coarse(0, 0);
        }

#pragma omp parallel for
        for (int cj = 1; cj < n2h; ++cj) {
            const int fj = cj * 2;
            coarse(0, cj) = (2 * (fine(0, fj) + fine(1, fj))
                    + fine(0, fj-1) + fine(1, fj-1)
                    + fine(0, fj+1) + fine(1, fj+1)) / 8;
        }
    }

    // Corners
    if (bcs.is_periodic_x() && bcs.is_periodic_y()) {
        coarse(0, 0) = (4 * fine(0, 0)
                + 2 * (fine(nh-1, 0) + fine(1, 0) + fine(0, nh-1) + fine(0, 1))
                + fine(nh-1, nh-1) + fine(1, nh-1) + fine(1, 1) + fine(nh-1, 1)
        ) / 16;
        coarse(n2h, 0) = coarse(0, 0);
        coarse(0, n2h) = coarse(0, 0);
        coarse(n2h, n2h) = coarse(0, 0);
    }

    if (bcs.is_east_neumann() && bcs.is_north_neumann())
        coarse(n2h, n2h) = (fine(nh, nh) + fine(nh-1, nh) + fine(nh, nh-1) + fine(nh-1, nh-1)) / 4;

    if (bcs.is_east_neumann() && bcs.is_south_neumann())
        coarse(n2h, 0) = (fine(nh, 0) + fine(nh-1, 0) + fine(nh, 1) + fine(nh-1, 1)) / 4;

    if (bcs.is_west_neumann() && bcs.is_south_neumann())
        coarse(0, 0) = (fine(0, 0) + fine(1, 0) + fine(0, 1) + fine(1, 1)) / 4;

    if (bcs.is_west_neumann() && bcs.is_north_neumann())
        coarse(0, n2h) = (fine(0, nh) + fine(1, nh) + fine(0, nh-1) + fine(1, nh-1)) / 4;

    // Dirichlet
    if (bcs.is_north_dirichlet()) {
#pragma omp parallel for
        for (int ci = 0; ci < n2h; ++ci) {
            const int fi = ci * 2;
            coarse(ci, n2h) = fine(fi, nh);
        }
    }

    if (bcs.is_south_dirichlet()) {
#pragma omp parallel for
        for (int ci = 0; ci < n2h; ++ci) {
            const int fi = ci * 2;
            coarse(ci, 0) = fine(fi, 0);
        }
    }

    if (bcs.is_east_dirichlet()) {
#pragma omp parallel for
        for (int cj = 0; cj < n2h; ++cj) {
            const int fj = cj * 2;
            coarse(n2h, cj) = fine(nh, fj);
        }
    }

    if (bcs.is_west_dirichlet()) {
#pragma omp parallel for
        for (int cj = 0; cj < n2h; ++cj) {
            const int fj = cj * 2;
            coarse(0, cj) = fine(0, fj);
        }
    }
}

void RestrictorFullWeighting2D::run_device(Array& fine, Array& coarse, BoundaryConditions& bcs) {
    // i.e. will not work for alternating directions
    assert(coarse.get_nrows() == coarse.get_ncols());
    assert(fine.get_nrows() == fine.get_ncols());

    const uint threads = std::min(32U, static_cast<uint>(coarse.get_nrows()) - 1);
    const uint blocks = (coarse.get_ncols() + threads - 1) / threads;
    dim3 threads_2d(threads, threads);
    dim3 blocks_2d(blocks, blocks);

    const uint n = coarse.get_ncols();
    const uint threads_1d = std::min(m_max_threads_per_block, n);
    const uint blocks_1d = (n + threads_1d - 1) / threads_1d;

    restrictor_full_weighting_2d::kernel<<<blocks_2d, threads_2d>>>(fine, coarse, bcs);
    restrictor_full_weighting_2d::boundaries<<<blocks_1d, threads_1d>>>(fine, coarse, bcs);
    restrictor_full_weighting_2d::corners<<<blocks_1d, threads_1d>>>(fine, coarse, bcs);
}

} // namespace modules
} // namespace pmf
