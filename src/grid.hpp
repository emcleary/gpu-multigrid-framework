#pragma once

#include "array.hpp"


namespace pmf {

class Grid {
public:
    Grid(const double xmin, const double xmax, const size_t nx);

    Grid(const double xmin, const double xmax, const size_t nx,
            const double ymin, const double ymax, const size_t ny);

    const Array& get_x() const { return m_x; }
    const Array& get_y() const { return m_y; }

    size_t size() const { return m_nx * m_ny; }
    double get_cell_width() const { return m_hx; }
    double get_cell_height() const { return m_hy; }

    double get_xmin() const { return m_x.front(); }
    double get_xmax() const { return m_x.back(); }

    double get_ymin() const { return m_y.front(); }
    double get_ymax() const { return m_y.back(); }

private:
    const size_t m_nx, m_ny;
    const double m_hx, m_hy;
    Array m_x, m_y;
};

} // namespace pmf
