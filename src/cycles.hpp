#pragma once

#include <iostream>

#include "grid.hpp"
#include "solver.cuh"


namespace gmf {

class Algorithm {
public:
    Algorithm() {}

    void set_solver(std::shared_ptr<Solver> solver) {
        m_solver = solver;
    }

    int get_num_levels() {
        return m_solver->get_num_levels();
    }

    void initialize(Grid& grid, const int n_gpu) {
        m_solver->initialize(grid, n_gpu);
    }

    virtual double run() = 0;

protected:
    std::shared_ptr<Solver> m_solver;
};

class VCycle : public Algorithm {
public:
    VCycle() : Algorithm() {}

    virtual double run() override;

    void set_relaxation_iterations(const int nu0, const int nu1) {
        m_nu0 = nu0;
        m_nu1 = nu1;
    }

private:
    void run_level(int lvl);

    int m_nu0 = 2;
    int m_nu1 = 1;
};

} // namespace gmf
