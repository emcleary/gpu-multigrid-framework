#include "cycles.hpp"


namespace pmf {

double VCycle::run_level(int lvl) {
    m_solver->smooth(lvl, m_nu0);
    double residual_norm = 0;
    if (lvl > 0) {
        residual_norm = m_solver->restrict(lvl);
        run_level(lvl - 1);
        m_solver->correct(lvl);
    }
    m_solver->smooth(lvl, m_nu1);
    return residual_norm;
}

double VCycle::run() {
    return run_level(m_solver->get_num_levels() - 1);
}

} // namespace pmf
