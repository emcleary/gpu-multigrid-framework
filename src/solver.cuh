#pragma once

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
#include "modules/interfaces/rhs.hpp"
#include "modules/boundary_conditions.hpp"


namespace gmf {

class Solver {
public:
    Solver() {}

    const Array& get_solution() { return m_levels.back().solution; }

    int get_num_levels() { return m_levels.size(); }

    void set_max_threads_per_block(int n_threads) {
        m_max_threads_per_block = n_threads;
    }

    // allocates all memory needed based on grid info (e.g. N),
    // store in levels, and initialize data in the finest level
    void initialize(Grid& grid, const int n_gpu);

    // calculates the residaul norm, only inteneded
    // for use on the finest level for convergence testing
    double calculate_residual_norm();
    
    void smooth(const int n_iter, const int lvl);

    virtual void restrict(const int lvl) = 0;
    virtual void correct(const int lvl) = 0;

    void set_equation(std::shared_ptr<modules::Equation> eqn) {
        m_eqn = eqn;
    }
            
    void set_lhs(std::shared_ptr<modules::LHS> lhs) {
        m_lhs = lhs;
    }
            
    void set_rhs(std::shared_ptr<modules::RHS> rhs) {
        m_rhs = rhs;
    }
            
    void set_interpolator(std::shared_ptr<modules::Interpolator> interpolator) {
        m_interpolator = interpolator;
    }
            
    void set_restrictor(std::shared_ptr<modules::Restrictor> restrictor) {
        m_restrictor = restrictor;
    }
            
    void set_iterator(std::shared_ptr<modules::Iterator> iterator) {
        m_iterator = iterator;
    }
            
    void set_residual_norm(std::shared_ptr<modules::Norm> norm) {
        m_norm = norm;
    }

    void set_boundary_conditions(std::shared_ptr<modules::BoundaryConditions> bcs) {
        m_boundary_conditions = bcs;
    }

protected:
    // used for simple kernels, like add and substract
    size_t m_max_threads_per_block = 512;

    std::shared_ptr<modules::Equation> m_eqn;
    std::shared_ptr<modules::Norm> m_norm;
    std::shared_ptr<modules::Iterator> m_iterator;
    std::shared_ptr<modules::Interpolator> m_interpolator;
    std::shared_ptr<modules::Restrictor> m_restrictor;
    std::shared_ptr<modules::LHS> m_lhs;
    std::shared_ptr<modules::RHS> m_rhs;
    std::shared_ptr<modules::BoundaryConditions> m_boundary_conditions;

    std::vector<Level> m_levels;

    void check_ready_for_run() {
        assert(m_eqn != nullptr && "Equation not set!\n");
        assert(m_norm != nullptr && "Norm not set!\n");
        assert(m_iterator != nullptr && "Iterator not set!\n");
        assert(m_interpolator != nullptr && "Interpolator not set!\n");
        assert(m_restrictor != nullptr && "Restrictor not set!\n");
        assert(m_lhs != nullptr && "LHS not set!\n");
        assert(m_rhs != nullptr && "RHS not set!\n");
        assert(m_boundary_conditions != nullptr && "BoundaryConditions not set!\n");
    }    
};

} // namespace gmf
