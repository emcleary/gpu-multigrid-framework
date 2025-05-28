#include "solver_linear.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace gmf {

void SolverLinear::initialize(Grid& grid, const int n_gpu) {
    check_ready_for_run();
    
    const int N = grid.size();
    const int n_levels = (int)std::floor(std::log2(N));
    m_levels.clear();
    for (int i = 2, nl = n_levels; i < N; i += i, --nl) {
        bool on_gpu = nl < n_gpu;
        m_levels.push_back(LevelLinear(i + 1, on_gpu, grid));
    }

    Array& solution = m_levels.back().solution;
    Array& rhs = m_levels.back().rhs;
    const Array& x = grid.get_x();
    for (int i = 0; i < N; ++i) {
        solution[i] = m_eqn->initial_condition(x[i]);
        rhs[i] = m_eqn->rhs(x[i], solution[i]);
    }
}

double SolverLinear::calculate_residual_norm() {
    Array& Av = m_levels.back().temporary;
    Array& v = m_levels.back().solution;
    Array& f = m_levels.back().rhs;
    Grid& grid = m_levels.back().grid;

    if (m_levels.back().on_gpu) {
        m_lhs->run_device(Av, v, f, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels.back().temporary;
    
        const int threadsPerBlock = std::min(m_max_threads_per_block, f.size() - 1);
        const int blocksPerGrid = (f.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(Av, f, resid);

        return m_norm->run_device(resid, grid);
    } else {
        m_lhs->run_host(Av, v, f, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels.back().temporary;
        for (int i = 0; i < f.size(); ++i)
            resid[i] = Av[i] - f[i];
    
        return m_norm->run_host(resid, grid);
    }
}

void SolverLinear::relax(const int lvl, const int n_iter) {
    if (m_levels[lvl].on_gpu) {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_device(m_levels[lvl].solution,
                    m_levels[lvl].rhs, m_levels[lvl].grid);
    } else {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_host(m_levels[lvl].solution,
                    m_levels[lvl].rhs, m_levels[lvl].grid);
    }
}

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
        m_lhs->run_device(residual, solution, rhs, grid);

        const int threadsPerBlock = std::min(m_max_threads_per_block, residual.size() - 1);
        const int blocksPerGrid = (residual.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(rhs, residual, residual);

        m_restrictor->run_device(residual, coarse_rhs);

        Array& coarse_solution = m_levels[lvl-1].solution;
        cudaCheck(cudaMemset(&coarse_solution[0], 0, coarse_solution.size() * sizeof(double)));

        if (!m_levels[lvl-1].on_gpu)
            cudaCheck(cudaDeviceSynchronize());

    } else {
        m_lhs->run_host(residual, solution, rhs, grid);

        for (int i = 0; i < residual.size(); ++i)
            residual[i] = rhs[i] - residual[i];

        m_restrictor->run_host(residual, coarse_rhs);

        for (int i = 0; i < coarse_solution.size(); ++i)
            coarse_solution[i] = 0;
    }
}

/*
 * Since subiterations solve for errors, this interpolation step not
 * only interpolates the error, but also corrects the fine
 * error/solution.
 */
void SolverLinear::interpolate(const int lvl) {
    assert(lvl > 0);
    Array& solution = m_levels[lvl-1].solution;
    Array& fine_error = m_levels[lvl].temporary;
    Array& fine_solution = m_levels[lvl].solution;
    if (m_levels[lvl].on_gpu) {
        m_interpolator->run_device(solution, fine_error);

        const int threadsPerBlock = std::min(m_max_threads_per_block, fine_solution.size() - 1);
        const int blocksPerGrid = (fine_solution.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_add<<<blocksPerGrid, threadsPerBlock>>>(fine_solution, fine_error, fine_solution);
    } else {
        m_interpolator->run_host(solution, fine_error);

        for (int i = 0; i < fine_solution.size(); ++i)
            fine_solution[i] += fine_error[i];
    }
}

} // namespace gmf
