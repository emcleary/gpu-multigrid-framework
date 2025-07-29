#include "solver_linear.cuh"

#include <vector>
#include <iostream>
#include <omp.h>

#include "utilities.hpp"


namespace pmf {

/*
 * This solver only works directly with the solution and RHS at the
 * finest level. Subiterations work with error and residual instead,
 * which are labeled as "coarse solution" and "coarse RHS"
 * respectively, just out of convenience.
 */
double SolverLinear::restrict(const int lvl) {
    assert(lvl > 0);
    Array& residual = m_levels->get_level(lvl).temporary;
    Array& solution = m_levels->get_level(lvl).solution;
    Array& rhs = m_levels->get_level(lvl).rhs;
    Grid& grid = m_levels->get_level(lvl).grid;

    Array& coarse_rhs = m_levels->get_level(lvl-1).rhs;
    Array& coarse_solution = m_levels->get_level(lvl-1).solution;

    double residual_norm = 0;
    
    if (m_levels->get_level(lvl).on_gpu) {
        m_lhs->run_device(residual, solution, *m_boundary_conditions, grid);

        m_sub.run_device(rhs, residual, residual);

        m_restrictor->run_device(residual, coarse_rhs, *m_boundary_conditions);

        Array& coarse_solution = m_levels->get_level(lvl-1).solution;
        cudaCheck(cudaMemset(&coarse_solution[0], 0, coarse_solution.size() * sizeof(double)));

        if (!m_levels->get_level(lvl-1).on_gpu)
            cudaCheck(cudaDeviceSynchronize());

        residual_norm = m_norm->run_device(residual, grid);
    } else {
        m_lhs->run_host(residual, solution, *m_boundary_conditions, grid);

        m_sub.run_host(rhs, residual, residual);

        m_restrictor->run_host(residual, coarse_rhs, *m_boundary_conditions);

        omp_set_num_threads(m_omp_threads);
#pragma omp parallel for
        for (int i = 0; i < coarse_solution.size(); ++i)
            coarse_solution[i] = 0;

        residual_norm = m_norm->run_host(residual, grid);
    }

    return residual_norm;
}

/*
 * Since subiterations solve for errors, this interpolation step not
 * only interpolates the error, but also corrects the fine
 * error/solution.
 */
void SolverLinear::correct(const int lvl) {
    assert(lvl > 0);
    Array& solution = m_levels->get_level(lvl-1).solution;
    Array& fine_error = m_levels->get_level(lvl).temporary;
    Array& fine_solution = m_levels->get_level(lvl).solution;
    if (m_levels->get_level(lvl).on_gpu) {
        m_interpolator->run_device(solution, fine_error, *m_boundary_conditions);
        m_add.run_device(fine_solution, fine_error, fine_solution);
    } else {
        m_interpolator->run_host(solution, fine_error, *m_boundary_conditions);
        m_add.run_host(fine_solution, fine_error, fine_solution);
    }
}

} // namespace pmf
