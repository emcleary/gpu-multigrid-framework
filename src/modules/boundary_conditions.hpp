#pragma once

#include <iostream>

#include <cuda_runtime.h>

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

    __host__ __device__
    bool is_periodic() const { return m_periodic; }

    __host__ __device__
    bool is_left_dirichlet() const { return m_left_type == BC::Dirichlet; }

    __host__ __device__
    bool is_right_dirichlet() const { return m_right_type == BC::Dirichlet; }

    __host__ __device__
    bool is_left_neumann() const { return m_left_type == BC::Neumann; }

    __host__ __device__
    bool is_right_neumann() const { return m_right_type == BC::Neumann; }

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

    void set_left_neumann(double value) {
        m_left = value;
        m_left_type = BC::Neumann;
        m_periodic = false;
    }

    void set_right_neumann(double value) {
        m_right = value;
        m_right_type = BC::Neumann;
        m_periodic = false;
    }

    void check_valid() const {
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

    __host__ __device__
    double get_right() const { return m_right; }

    __host__ __device__
    double get_left() const { return m_left; }

private:
    double m_right, m_left;
    BC m_right_type = BC::None;
    BC m_left_type = BC::None;
    bool m_periodic = false;
};

} // namespace modules
} // namespace gmf
