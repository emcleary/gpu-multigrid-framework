#include "solver_nonlinear_full.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace pmf {

/*
 * We need to restrict solution v and an adjusted version of the RHS
 * of the equation.
 * 
 * If R is some restriction operator, then the adjusted coarsened RHS is
 * RHS <- R(RHS) + R(A(v)) - A(R(v))
 */
double SolverNonlinearFull::restrict(const int lvl) {
    assert(lvl > 0);
    Array& v = m_levels->get_level(lvl).solution;
    Array& residual = m_levels->get_level(lvl).temporary;
    Array& rhs = m_levels->get_level(lvl).rhs;
    Grid& grid = m_levels->get_level(lvl).grid;

    Array& coarse_v = m_levels->get_level(lvl-1).solution;
    Array& coarse_Av = m_levels->get_level(lvl-1).temporary;
    Array& coarse_rhs = m_levels->get_level(lvl-1).rhs;
    Grid& coarse_grid = m_levels->get_level(lvl-1).grid;

    double residual_norm = 0;

    if (m_levels->get_level(lvl).on_gpu) {

        // calculate residual on the fine grid
        m_lhs->run_device(residual, v, *m_boundary_conditions, grid); // calculate Av
        m_sub.run_device(rhs, residual, residual);
        residual_norm = m_norm->run_device(residual, grid);

        // coarsen residual and solution
        m_restrictor->run_device(v, coarse_v, *m_boundary_conditions);
        m_restrictor->run_device(residual, coarse_rhs, *m_boundary_conditions);

        // Recalculate Av with the coarse v, then add to RHS
        m_lhs->run_device(coarse_Av, coarse_v, *m_boundary_conditions, coarse_grid);
        m_add.run_device(coarse_rhs, coarse_Av, coarse_rhs);

        // Store coarse solution temporarily to calculate error AFTER
        // relaxing/solving on the coarser mesh.
        // To save memory, just store this in the FINE temporary array
        Array& temporary = m_levels->get_level(lvl).temporary;
        m_copy.run_device(coarse_v, temporary);

        if (!m_levels->get_level(lvl-1).on_gpu)
            cudaCheck(cudaDeviceSynchronize());
    } else {
        // calculate residual on the fine grid
        m_lhs->run_host(residual, v, *m_boundary_conditions, grid); // calculate Av
        m_sub.run_host(rhs, residual, residual); // residsual = f - Av
        residual_norm = m_norm->run_host(residual, grid);

        // coarsen residual and solution
        m_restrictor->run_host(v, coarse_v, *m_boundary_conditions);
        m_restrictor->run_host(residual, coarse_rhs, *m_boundary_conditions);

        // correct the coarse grid residual
        m_lhs->run_host(coarse_Av, coarse_v, *m_boundary_conditions, coarse_grid);
        m_add.run_host(coarse_rhs, coarse_Av, coarse_rhs);

        // Store coarse solution temporarily to calculate error AFTER
        // relaxing/solving on the coarser mesh.
        // To save memory, just store this in the FINE temporary array
        Array& temporary = m_levels->get_level(lvl).temporary;
        m_copy.run_host(coarse_v, temporary);
    }

    return residual_norm;
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
void SolverNonlinearFull::correct(const int lvl) {
    assert(lvl > 0);
    Array& solution_prev = m_levels->get_level(lvl).temporary; // NB: prev coarse solution intentionally stored in fine temporary array
    Array& solution = m_levels->get_level(lvl-1).solution;
    Array& error = m_levels->get_level(lvl-1).temporary;

    Array& fine_error = m_levels->get_level(lvl).temporary;
    Array& fine_solution = m_levels->get_level(lvl).solution;

    if (m_levels->get_level(lvl).on_gpu) {
        // Calculate error
        m_sub.run_device(solution, solution_prev, error);

        // Correct the solution with the error
        m_interpolator->run_device(error, fine_error, *m_boundary_conditions);
        m_add.run_device(fine_solution, fine_error, fine_solution);
    } else {
        // Calculate error
        m_sub.run_host(solution, solution_prev, error);

        // Correct the solution with the error
        m_interpolator->run_host(error, fine_error, *m_boundary_conditions);
        m_add.run_host(fine_solution, fine_error, fine_solution);
    }
}

} // namespace pmf
