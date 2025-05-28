#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "equation.hpp"

class LHSNonlinear : public gmf::modules::LHS {
public:
    LHSNonlinear(const size_t max_threads_per_block, std::shared_ptr<NonlinearEquation> eqn)
            : gmf::modules::LHS(max_threads_per_block), m_eqn(eqn) {}


    virtual void run_host(gmf::Array& lhs, gmf::Array& v, const gmf::Array& f, const gmf::Grid& grid) override;
    virtual void run_device(gmf::Array& lhs, gmf::Array& v, const gmf::Array& f, const gmf::Grid& grid) override;

private:
    std::shared_ptr<NonlinearEquation> m_eqn;
};
