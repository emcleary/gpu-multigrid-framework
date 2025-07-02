#pragma once

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/modules/interfaces.hpp"
#include "src/modules/boundary_conditions.hpp"


class LHSCPU : public gmf::modules::LHS {
public:
    LHSCPU(const size_t max_threads_per_block) : gmf::modules::LHS(max_threads_per_block) {}

    virtual void run_host(gmf::Array& lhs, gmf::Array& v,
            const gmf::modules::BoundaryConditions& bcs, const gmf::Grid& grid) override;
};
