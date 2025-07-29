#pragma once

#include <cmath>

#include <iostream>

#include <cuda_runtime.h>

namespace pmf {
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
    bool is_periodic_x() const { return m_west_type == BC::Periodic && m_east_type == BC::Periodic; }

    __host__ __device__
    bool is_periodic_y() const { return m_north_type == BC::Periodic && m_south_type == BC::Periodic; }

    __host__ __device__
    bool is_west_dirichlet() const { return m_west_type == BC::Dirichlet; }

    __host__ __device__
    bool is_east_dirichlet() const { return m_east_type == BC::Dirichlet; }

    __host__ __device__
    bool is_north_dirichlet() const { return m_north_type == BC::Dirichlet; }

    __host__ __device__
    bool is_south_dirichlet() const { return m_south_type == BC::Dirichlet; }

    __host__ __device__
    bool is_west_neumann() const { return m_west_type == BC::Neumann; }

    __host__ __device__
    bool is_east_neumann() const { return m_east_type == BC::Neumann; }

    __host__ __device__
    bool is_north_neumann() const { return m_north_type == BC::Neumann; }

    __host__ __device__
    bool is_south_neumann() const { return m_south_type == BC::Neumann; }

    void set_periodic_x() {
        m_west_type = BC::Periodic;
        m_east_type = BC::Periodic;
    }

    void set_periodic_y() {
        m_north_type = BC::Periodic;
        m_south_type = BC::Periodic;
    }

    void set_west_dirichlet() { m_west_type = BC::Dirichlet; }
    void set_east_dirichlet() { m_east_type = BC::Dirichlet; }
    void set_north_dirichlet() { m_north_type = BC::Dirichlet; }
    void set_south_dirichlet() { m_south_type = BC::Dirichlet; }

    void set_west_neumann() { m_west_type = BC::Neumann; }
    void set_east_neumann() { m_east_type = BC::Neumann; }
    void set_north_neumann() { m_north_type = BC::Neumann; }
    void set_south_neumann() { m_south_type = BC::Neumann; }

    void check_valid() const {
        if (is_periodic_x())
            return;
        
        if (m_east_type == BC::None || m_east_type == BC::Periodic) {
            std::cerr << "East BC not set!\n";
            exit(EXIT_FAILURE);
        }

        if (m_west_type == BC::None || m_west_type == BC::Periodic) {
            std::cerr << "West BC not set!\n";
            exit(EXIT_FAILURE);
        }
        
        if (m_north_type == BC::None || m_north_type == BC::Periodic) {
            std::cerr << "North BC not set!\n";
            exit(EXIT_FAILURE);
        }

        if (m_south_type == BC::None || m_south_type == BC::Periodic) {
            std::cerr << "South BC not set!\n";
            exit(EXIT_FAILURE);
        }
    }

    // MIGHT want to keep these for nonlinear problems
    // Any better ways to handle these???
    // They are likely only needed for very user-specific cases (lhs, rhs, gs)
    // TODO: make virtual?
    __host__ __device__
    double get_dirichlet_west(double y) const { return std::nan(""); }

    __host__ __device__
    double get_dirichlet_west() const { return get_dirichlet_west(0); }

    // TODO: make virtual?
    __host__ __device__
    double get_dirichlet_east(double y) const { return std::nan(""); }

    __host__ __device__
    double get_dirichlet_east() const { return get_dirichlet_east(0); }

private:
    BC m_east_type = BC::None;
    BC m_west_type = BC::None;
    BC m_north_type = BC::None;
    BC m_south_type = BC::None;
};

} // namespace modules
} // namespace pmf
