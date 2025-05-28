#include "grid.hpp"

#include <iostream>

#include "array.hpp"

namespace gmf {

Grid::Grid(const double xmin, const double xmax, const size_t N)
        : m_N(N), m_h((xmax - xmin) / (N - 1)), m_x(N) {

    if (((N - 1) & (N - 2)) != 0) {
        std::cerr << "Grid input N cells must be a power of 2 plus 1!\n";
        exit(EXIT_FAILURE);
    }
    
    for (int i = 1; i < m_N - 1; ++i)
        m_x[i] = i * m_h;
    m_x.front() = xmin;
    m_x.back() = xmax;
}

} // namespace gmf
