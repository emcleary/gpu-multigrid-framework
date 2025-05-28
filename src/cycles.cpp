#include "cycles.hpp"


namespace gmf {

void VCycle::run_level(int lvl) {
    m_solver->relax(lvl, m_nu0);
    if (lvl > 0) {
        m_solver->restrict(lvl);
        run_level(lvl - 1);
        m_solver->interpolate(lvl);
    }
    m_solver->relax(lvl, m_nu1);
}

double VCycle::run() {
    m_solver->initialize_cycle();
    run_level(m_solver->get_num_levels() - 1);
    m_solver->finalize_cycle(); 
    return m_solver->calculate_residual_norm();
}

} // namespace gmf
