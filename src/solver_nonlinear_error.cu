#include "solver_nonlinear_error.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace gmf {

void SolverNonlinearError::initialize(Grid& grid, const int n_gpu) {
    check_ready_for_run();

    const int N = grid.size();
    const int n_levels = (int)std::floor(std::log2(N));
    m_levels.clear();
    for (int i = 2, nl = n_levels; i < N; i += i, --nl) {
        bool on_gpu = nl < n_gpu;
        m_levels.push_back(LevelNonlinearError(i + 1, on_gpu, grid));
    }

    m_rhs = Array(N);

    Array& solution = m_levels.back().solution;
    const Array& x = grid.get_x();
    for (int i = 0; i < N; ++i) {
        solution[i] = m_eqn->initial_condition(x[i]);
        m_rhs[i] = m_eqn->rhs(x[i], solution[i]);
    }
}

/*
 * This solver works with errors and residuals, rather than solutions
 * and the RHS. Errors and residuals must be reset and recalculated,
 * respectively, at the start of each iteration of a V-cycle.
 */
void SolverNonlinearError::initialize_cycle() {
    Array& solution = m_levels.back().solution;
    Array& residual = m_levels.back().residual;
    Array& error = m_levels.back().error;
    Grid& grid = m_levels.back().grid;

    if (m_levels.back().on_gpu) {
        const int threadsPerBlock = std::min(m_max_threads_per_block, residual.size() - 1);
        const int blocksPerGrid = (residual.size() + threadsPerBlock - 1) / threadsPerBlock;
    
        for (int j = 0; j < m_nu0; ++j)
            m_iterator->run_device(solution, m_rhs, grid);

        m_lhs->run_device(residual, solution, m_rhs, grid);
    
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(m_rhs, residual, residual);

        cudaCheck(cudaMemset(error.data(), 0, error.size() * sizeof(double)));

    } else {
        for (int j = 0; j < m_nu0; ++j)
            m_iterator->run_host(solution, m_rhs, grid);

        m_lhs->run_host(residual, solution, m_rhs, grid);

        for (int i = 0; i < residual.size(); ++i)
            residual[i] = m_rhs[i] - residual[i];

        for (int i = 0; i < error.size(); ++i)
            error[i] = 0;
    }
}

/*
 * At the end of each V-cycle, the solution must be corrected by the
 * error. An additional relaxation step is included.
 */
void SolverNonlinearError::finalize_cycle() {
    Array& solution = m_levels.back().solution;
    Array& error = m_levels.back().error;
    Grid& grid = m_levels.back().grid;

    if (m_levels.back().on_gpu) {
        const int threadsPerBlock = std::min(m_max_threads_per_block, error.size() - 1);
        const int blocksPerGrid = (error.size() + threadsPerBlock - 1) / threadsPerBlock;

        kernel_add<<<blocksPerGrid, threadsPerBlock>>>(solution, error, solution);

        for (int j = 0; j < m_nu1; ++j)
            m_iterator->run_device(solution, m_rhs, grid);

    } else {
        for (int i = 0; i < solution.size(); ++i)
            solution[i] += error[i];    

        for (int j = 0; j < m_nu1; ++j)
            m_iterator->run_host(solution, m_rhs, grid);
    }        
}

double SolverNonlinearError::calculate_residual_norm() {
    Array& Av = m_levels.back().residual;
    Array& v = m_levels.back().solution;
    Array& f = m_rhs;
    Grid& grid = m_levels.back().grid;

    if (m_levels.back().on_gpu) {
        m_lhs->run_device(Av, v, f, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels.back().residual;
    
        const int threadsPerBlock = std::min(m_max_threads_per_block, f.size() - 1);
        const int blocksPerGrid = (f.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(Av, f, resid);

        return m_norm->run_device(resid, grid);
    } else {
        m_lhs->run_host(Av, v, f, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels.back().residual;
        for (int i = 0; i < f.size(); ++i)
            resid[i] = Av[i] - f[i];
    
        return m_norm->run_host(resid, grid);
    }
}

void SolverNonlinearError::relax(const int lvl, const int n_iter) {
    if (m_levels[lvl].on_gpu) {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_error_device(m_levels[lvl].error, m_levels[lvl].solution,
                    m_levels[lvl].residual, m_levels[lvl].grid);
    } else {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_error_host(m_levels[lvl].error, m_levels[lvl].solution,
                    m_levels[lvl].residual, m_levels[lvl].grid);
    }
}

void SolverNonlinearError::restrict(const int lvl) {
    assert(lvl > 0);
    Array& solution = m_levels[lvl].solution;
    Array& residual = m_levels[lvl].residual;
    Array& error = m_levels[lvl].error;
    Array& coarse_solution = m_levels[lvl-1].solution;
    Array& coarse_residual = m_levels[lvl-1].residual;
    Array& coarse_error = m_levels[lvl-1].error;
        
    if (m_levels[lvl].on_gpu) {
        m_restrictor->run_device(solution, coarse_solution);
        m_restrictor->run_device(residual, coarse_residual);
        m_restrictor->run_device(error, coarse_error);

        if (!m_levels[lvl-1].on_gpu)
            cudaCheck(cudaDeviceSynchronize());

    } else {
        m_restrictor->run_host(solution, coarse_solution);
        m_restrictor->run_host(residual, coarse_residual);
        m_restrictor->run_host(error, coarse_error);
    }
}

void SolverNonlinearError::interpolate(const int lvl) {
    assert(lvl > 0);
    Array& error = m_levels[lvl-1].error;
    Array& fine_error = m_levels[lvl].error;

    if (m_levels[lvl].on_gpu) {
        m_interpolator->run_device(error, fine_error);
    } else {
        m_interpolator->run_host(error, fine_error);
    }
}

} // namespace gmf
