#pragma once

#include "array.hpp"


namespace gmf {

class Grid {
public:
    Grid(const double xmin, const double xmax, const size_t N);

    const Array& get_x() const { return m_x; }

    size_t size() const { return m_N; }
    double get_cell_width() const { return m_h; }
    double get_xmin() const { return m_x.front(); }
    double get_xmax() const { return m_x.back(); }

private:
    const size_t m_N;
    const double m_h;
    Array m_x;
};

} // namespace gmf
