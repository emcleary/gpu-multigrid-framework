#include "solver_linear.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace pmf {

void Solver::initialize(Grid& grid, const int n_gpu) {
    check_ready_for_run();

    m_levels->initialize(grid, n_gpu);

    Array& solution = m_levels->back().solution;
    m_eqn->fill_initial_condition(solution, grid, *m_boundary_conditions);

    Array& rhs = m_levels->back().rhs;
    m_eqn->fill_rhs(rhs, grid, *m_boundary_conditions);
}

double Solver::calculate_residual_norm() {
    Array& Av = m_levels->back().temporary;
    Array& v = m_levels->back().solution;
    Array& f = m_levels->back().rhs;
    Grid& grid = m_levels->back().grid;

    if (m_levels->back().on_gpu) {
        m_lhs->run_device(Av, v, *m_boundary_conditions, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels->back().temporary;
    
        m_sub.run_device(Av, f, resid);
    
        return m_norm->run_device(resid, grid);
    } else {
        m_lhs->run_host(Av, v, *m_boundary_conditions, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels->back().temporary;

        m_sub.run_host(Av, f, resid);
    
        return m_norm->run_host(resid, grid);
    }
}

void Solver::smooth(const int lvl, const int n_iter) {
    if (m_levels->get_level(lvl).on_gpu) {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_device(m_levels->get_level(lvl).solution, m_levels->get_level(lvl).rhs,
                    *m_boundary_conditions, m_levels->get_level(lvl).grid);
    } else {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_host(m_levels->get_level(lvl).solution, m_levels->get_level(lvl).rhs,
                    *m_boundary_conditions, m_levels->get_level(lvl).grid);
    }
}

} // namespace pmf
