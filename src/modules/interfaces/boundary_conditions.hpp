#pragma once

#include <cmath>

#include "src/array.hpp"
#include "src/grid.hpp"

namespace gmf {
namespace modules {

enum class BC {
    None,
    Periodic,
    Dirichlet,
    Neumann,
};

class BoundaryConditions {
public:
    BoundaryConditions() {}

    bool is_periodic() { return m_periodic; }
    bool is_left_dirichlet() { return true; }
    bool is_right_dirichlet() { return true; }
    bool is_left_neumann() { return true; }
    bool is_right_neumann() { return true; }

    void set_periodic() {
        m_periodic = true;
        m_left_type = BC::None;
        m_right_type = BC::None;
    }

    void set_left_dirichlet(double value) {
        m_left = value;
        m_left_type = BC::Dirichlet;
        m_periodic = false;
    }

    void set_right_dirichlet(double value) {
        m_right = value;
        m_right_type = BC::Dirichlet;
        m_periodic = false;
    }

    void check_valid() {
        if (m_periodic)
            return;
        
        if (m_right_type == BC::None) {
            std::cerr << "Right BC not set!\n";
            exit(EXIT_FAILURE);
        }

        if (m_left_type == BC::None) {
            std::cerr << "Left BC not set!\n";
            exit(EXIT_FAILURE);
        }
    }

private:
    double m_right, m_left;
    BC m_right_type = BC::None;
    BC m_left_type = BC::None;
    bool m_periodic = false;
};

} // namespace modules
} // namespace gmf
