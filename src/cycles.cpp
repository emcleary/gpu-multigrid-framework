#include "cycles.hpp"


namespace gmf {

void VCycle::run_level(int lvl) {
    m_solver->smooth(lvl, m_nu0);
    if (lvl > 0) {
        m_solver->restrict(lvl);
        run_level(lvl - 1);
        m_solver->correct(lvl);
    }
    m_solver->smooth(lvl, m_nu1);
}

double VCycle::run() {
    run_level(m_solver->get_num_levels() - 1);
    return m_solver->calculate_residual_norm();
}

} // namespace gmf
