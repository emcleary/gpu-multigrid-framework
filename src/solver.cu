#include "solver_linear.cuh"

#include <vector>
#include <iostream>

#include "utilities.hpp"


namespace gmf {

void Solver::initialize(Grid& grid, const int n_gpu) {
    check_ready_for_run();
    
    const int N = grid.size();
    const int n_levels = (int)std::floor(std::log2(N));
    m_levels.clear();
    for (int i = 2, nl = n_levels - 1; i < N; i += i, --nl) {
        bool on_gpu = nl < n_gpu;
        m_levels.push_back(Level(i + 1, on_gpu, grid));
    }

    Array& solution = m_levels.back().solution;
    Array& rhs = m_levels.back().rhs;
    const Array& x = grid.get_x();
    for (int i = 0; i < N; ++i)
        solution[i] = m_eqn->initial_condition(x[i]);
    m_rhs->run(rhs, grid, *m_boundary_conditions);
}

double Solver::calculate_residual_norm() {
    Array& Av = m_levels.back().temporary;
    Array& v = m_levels.back().solution;
    Array& f = m_levels.back().rhs;
    Grid& grid = m_levels.back().grid;

    if (m_levels.back().on_gpu) {
        m_lhs->run_device(Av, v, *m_boundary_conditions, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels.back().temporary;
    
        const int threadsPerBlock = std::min(m_max_threads_per_block, f.size() - 1);
        const int blocksPerGrid = (f.size() + threadsPerBlock - 1) / threadsPerBlock;
        kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(Av, f, resid);
    
        return m_norm->run_device(resid, grid);
    } else {
        m_lhs->run_host(Av, v, *m_boundary_conditions, grid);

        // NB: resid uses same memory as Av
        Array& resid = m_levels.back().temporary;
        for (int i = 0; i < f.size(); ++i)
            resid[i] = Av[i] - f[i];
    
        return m_norm->run_host(resid, grid);
    }
}

void Solver::smooth(const int lvl, const int n_iter) {
    if (m_levels[lvl].on_gpu) {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_device(m_levels[lvl].solution, m_levels[lvl].rhs,
                    *m_boundary_conditions, m_levels[lvl].grid);
    } else {
        for (int j = 0; j < n_iter; ++j)
            m_iterator->run_host(m_levels[lvl].solution, m_levels[lvl].rhs,
                    *m_boundary_conditions, m_levels[lvl].grid);
    }
}

} // namespace gmf
