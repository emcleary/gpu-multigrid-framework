#include "grid.hpp"

#include <iostream>

#include "array.hpp"

namespace pmf {

Grid::Grid(const double xmin, const double xmax, const size_t nx)
        : m_nx(nx), m_hx((xmax - xmin) / (nx - 1)), m_x(nx),
          m_ny(1), m_hy(0), m_y(0) {

    if (((nx - 1) & (nx - 2)) != 0) {
        std::cerr << "Grid input nx cells must be a power of 2 plus 1!\n";
        exit(EXIT_FAILURE);
    }
    
    for (int i = 1; i < m_nx - 1; ++i)
        m_x[i] = i * m_hx;
    m_x.front() = xmin;
    m_x.back() = xmax;
}

Grid::Grid(const double xmin, const double xmax, const size_t nx,
        const double ymin, const double ymax, const size_t ny)
        : m_nx(nx), m_hx((xmax - xmin) / (nx - 1)), m_x(nx),
          m_ny(ny), m_hy((ymax - ymin) / (ny - 1)), m_y(ny) {

    if (((nx - 1) & (nx - 2)) != 0) {
        std::cerr << "Grid input nx cells must be a power of 2 plus 1!\n";
        exit(EXIT_FAILURE);
    }
    
    if (((ny - 1) & (ny - 2)) != 0) {
        std::cerr << "Grid input ny cells must be a power of 2 plus 1!\n";
        exit(EXIT_FAILURE);
    }
    
    for (int i = 1; i < m_nx - 1; ++i)
        m_x[i] = i * m_hx;
    m_x.front() = xmin;
    m_x.back() = xmax;

    for (int i = 1; i < m_ny - 1; ++i)
        m_y[i] = i * m_hy;
    m_y.front() = ymin;
    m_y.back() = ymax;
}

} // namespace pmf
