#pragma once

#include "solver.cuh"

#include <vector>

#include "array.hpp"
#include "grid.hpp"
#include "kernels.cuh"
#include "levels.hpp"

#include "modules/interfaces/equation.hpp"
#include "modules/interfaces/interpolator.cuh"
#include "modules/interfaces/iterator.cuh"
#include "modules/interfaces/lhs.cuh"
#include "modules/interfaces/norm.cuh"
#include "modules/interfaces/restrictor.cuh"


namespace gmf {

class SolverNonlinearError : public Solver {
public:
    SolverNonlinearError() : m_nu0(2), m_nu1(1) {}
    SolverNonlinearError(int nu0, int nu1) : m_nu0(nu0), m_nu1(nu1) {}
    virtual const Array& get_solution() { return m_levels.back().solution; }
    virtual int get_num_levels() override { return m_levels.size(); };

    virtual void initialize(Grid& grid, const int n_gpu) override;
    virtual void initialize_cycle() override;
    virtual void finalize_cycle() override;
    virtual double calculate_residual_norm() override;

    virtual void relax(const int n_iter, const int lvl) override;
    virtual void restrict(const int lvl) override;
    virtual void interpolate(const int lvl) override;

private:
    std::vector<LevelNonlinearError> m_levels;
    Array m_rhs;
    int m_nu0, m_nu1;
};

} // namespace gmf
