#include "solver_linear.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace gmf {

/*
 * This solver only works directly with the solution and RHS at the
 * finest level. Subiterations work with error and residual instead,
 * which are labeled as "coarse solution" and "coarse RHS"
 * respectively, just out of convenience.
 */
void SolverLinear::restrict(const int lvl) {
    assert(lvl > 0);
    Array& residual = m_levels[lvl].temporary;
    Array& solution = m_levels[lvl].solution;
    Array& rhs = m_levels[lvl].rhs;
    Grid& grid = m_levels[lvl].grid;

    Array& coarse_rhs = m_levels[lvl-1].rhs;
    Array& coarse_solution = m_levels[lvl-1].solution;

    if (m_levels[lvl].on_gpu) {
        m_lhs->run_device(residual, solution, *m_boundary_conditions, grid);

        const int threadsPerBlock = std::min(m_max_threads_per_block, residual.size() - 1);
        const int blocksPerGrid = (residual.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(rhs, residual, residual);

        m_restrictor->run_device(residual, coarse_rhs, *m_boundary_conditions);

        Array& coarse_solution = m_levels[lvl-1].solution;
        cudaCheck(cudaMemset(&coarse_solution[0], 0, coarse_solution.size() * sizeof(double)));

        if (!m_levels[lvl-1].on_gpu)
            cudaCheck(cudaDeviceSynchronize());

    } else {
        m_lhs->run_host(residual, solution, *m_boundary_conditions, grid);

        for (int i = 0; i < residual.size(); ++i)
            residual[i] = rhs[i] - residual[i];

        m_restrictor->run_host(residual, coarse_rhs, *m_boundary_conditions);

        for (int i = 0; i < coarse_solution.size(); ++i)
            coarse_solution[i] = 0;
    }
}

/*
 * Since subiterations solve for errors, this interpolation step not
 * only interpolates the error, but also corrects the fine
 * error/solution.
 */
void SolverLinear::correct(const int lvl) {
    assert(lvl > 0);
    Array& solution = m_levels[lvl-1].solution;
    Array& fine_error = m_levels[lvl].temporary;
    Array& fine_solution = m_levels[lvl].solution;
    if (m_levels[lvl].on_gpu) {
        m_interpolator->run_device(solution, fine_error, *m_boundary_conditions);

        const int threadsPerBlock = std::min(m_max_threads_per_block, fine_solution.size() - 1);
        const int blocksPerGrid = (fine_solution.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_add<<<blocksPerGrid, threadsPerBlock>>>(fine_solution, fine_error, fine_solution);
    } else {
        m_interpolator->run_host(solution, fine_error, *m_boundary_conditions);

        for (int i = 0; i < fine_solution.size(); ++i)
            fine_solution[i] += fine_error[i];
    }
}

} // namespace gmf
