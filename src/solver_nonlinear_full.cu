#include "solver_nonlinear_full.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace gmf {

void SolverNonlinearFull::initialize(Grid& grid, const int n_gpu) {
    check_ready_for_run();

    const int N = grid.size();
    const int n_levels = (int)std::floor(std::log2(N));
    m_levels.clear();
    for (int i = 2, nl = n_levels; i < N; i += i, --nl) {
        bool on_gpu = nl < n_gpu;
        m_levels.push_back(LevelNonlinearFull(i + 1, on_gpu, grid));
    }

    Array& solution = m_levels.back().solution;
    Array& rhs = m_levels.back().rhs;
    const Array& x = grid.get_x();
    for (int i = 0; i < N; ++i) {
        solution[i] = m_eqn->initial_condition(x[i]);
        rhs[i] = m_eqn->rhs(x[i], solution[i]);
    }
}

double SolverNonlinearFull::calculate_residual_norm() {
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

void SolverNonlinearFull::relax(const int lvl, const int n_iter) {
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
 * We need to restrict solution v and an adjusted version of the RHS
 * of the equation.
 * 
 * If R is some restriction operator, then the adjusted coarsened RHS is
 * RHS <- R(RHS) + R(A(v)) - A(R(v))
 */
void SolverNonlinearFull::restrict(const int lvl) {
    assert(lvl > 0);
    Array& v = m_levels[lvl].solution;
    Array& Av = m_levels[lvl].temporary;
    Array& rhs = m_levels[lvl].rhs;
    Grid& grid = m_levels[lvl].grid;

    Array& coarse_v = m_levels[lvl-1].solution;
    Array& coarse_Av = m_levels[lvl-1].temporary;
    Array& coarse_rhs = m_levels[lvl-1].rhs;
    Grid& coarse_grid = m_levels[lvl-1].grid;

    if (m_levels[lvl].on_gpu) {
        m_lhs->run_device(Av, v, rhs, grid);

        m_restrictor->run_device(v, coarse_v);
        m_restrictor->run_device(Av, coarse_Av);
        m_restrictor->run_device(rhs, coarse_rhs);

        const int threadsPerBlock = std::min(m_max_threads_per_block, coarse_rhs.size() - 1);
        const int blocksPerGrid = (coarse_rhs.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(coarse_rhs, coarse_Av, coarse_rhs);

        // Recalculate Av with the coarse v, then add to RHS
        m_lhs->run_device(coarse_Av, coarse_v, coarse_rhs, coarse_grid);
        kernel_add<<<blocksPerGrid, threadsPerBlock>>>(coarse_rhs, coarse_Av, coarse_rhs);

        // Store coarse solution temporarily to calculate error AFTER
        // relaxing/solving on the coarser mesh.
        // To save memory, just store this in the FINE temporary array
        Array& temporary = m_levels[lvl].temporary;
        kernel_copy<<<blocksPerGrid, threadsPerBlock>>>(coarse_v, temporary, coarse_v.size());

        if (!m_levels[lvl-1].on_gpu)
            cudaCheck(cudaDeviceSynchronize());

    } else {
        m_lhs->run_host(Av, v, rhs, grid);

        m_restrictor->run_host(v, coarse_v);
        m_restrictor->run_host(Av, coarse_Av);
        m_restrictor->run_host(rhs, coarse_rhs);

        // Subtract coarsened LHS calculated with the fine solution
        for (int i = 0; i < coarse_rhs.size(); ++i)
            coarse_rhs[i] -= coarse_Av[i];

        // Recalculate Av with the coarse v, then add to RHS
        m_lhs->run_host(coarse_Av, coarse_v, coarse_rhs, coarse_grid);
        for (int i = 0; i < coarse_rhs.size(); ++i) {
            coarse_rhs[i] += coarse_Av[i];
        }

        // Store coarse solution temporarily to calculate error AFTER
        // relaxing/solving on the coarser mesh.
        // To save memory, just store this in the FINE temporary array
        Array& temporary = m_levels[lvl].temporary;
        for (int i = 0; i < coarse_v.size(); ++i)
            temporary[i] = coarse_v[i];
    }
}

/*
 * Interpolate errors, NOT solutions.
 *
 * Steps are the following:
 * 1) Calculate the coarse error e = v - v0
 * 2) Interpolate the coarse error to the fine grid
 * 3) Correct the fine solution with the interopolated error
 *    v = v0 + I(e)
 */
void SolverNonlinearFull::interpolate(const int lvl) {
    assert(lvl > 0);
    Array& solution_prev = m_levels[lvl].temporary; // NB: prev coarse solution intentionally stored in fine temporary array
    Array& solution = m_levels[lvl-1].solution;
    Array& error = m_levels[lvl-1].temporary;

    Array& fine_error = m_levels[lvl].temporary;
    Array& fine_solution = m_levels[lvl].solution;

    if (m_levels[lvl].on_gpu) {
        // Calculate error
        int threadsPerBlock = std::min(m_max_threads_per_block, error.size() - 1);
        int blocksPerGrid = (error.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(solution, solution_prev, error);

        m_interpolator->run_device(error, fine_error);

        // Correct the solution with the error
        threadsPerBlock = std::min(m_max_threads_per_block, fine_solution.size() - 1);
        blocksPerGrid = (fine_solution.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_add<<<blocksPerGrid, threadsPerBlock>>>(fine_solution, fine_error, fine_solution);
    } else {
        // Calculate error
        for (int i = 0; i < error.size(); ++i)
            error[i] = solution[i] - solution_prev[i];

        m_interpolator->run_host(error, fine_error);

        // Correct the solution with the error
        for (int i = 0; i < fine_solution.size(); ++i)
            fine_solution[i] += fine_error[i];
    }
}

} // namespace gmf
