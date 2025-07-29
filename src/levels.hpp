#pragma once

#include <cmath>
#include <iostream>
#include "grid.hpp"

namespace pmf {

struct Level {
public:
    Level(double xmin, double xmax, int n, bool on_gpu)
            : on_gpu(on_gpu), solution(n), grid(xmin, xmax, n),
              rhs(n), temporary(n) {}

    Level(double xmin, double xmax, int nx, double ymin, double ymax, int ny, bool on_gpu)
            : on_gpu(on_gpu), solution(nx, ny), grid(xmin, xmax, nx, ymin, ymax, ny),
              rhs(nx, ny), temporary(nx, ny) {}

    const bool on_gpu;
    Array solution;
    Grid grid;
    Array rhs;
    Array temporary;
};

class Levels {
public:
    Levels() {}

    virtual void initialize(Grid& grid, const int n_gpu) = 0;

    size_t size() const { return m_levels.size(); }

    Level& get_level(int idx) {
        if (idx < 0 || idx >= m_levels.size()) {
            std::cerr << "Level index " << idx << " out of bounds\n";
            exit(EXIT_FAILURE);
        }
        return m_levels[idx];
    }

    Level& back() { return m_levels.back(); }

protected:
    std::vector<Level> m_levels;
};

class Levels1D : public Levels {
public:
    Levels1D() : Levels() {}
    
    virtual void initialize(Grid& grid, const int n_gpu) override {
        const double xmin = grid.get_xmin();
        const double xmax = grid.get_xmax();
        const int nx = grid.size();
        const int n_levels = (int)std::floor(std::log2(nx));
        m_levels.clear();
        for (int i = 2, nl = n_levels - 1; i < nx; i += i, --nl) {
            bool on_gpu = nl < n_gpu;
            m_levels.push_back(Level(xmin, xmax, i+1, on_gpu));
        }
    }
};

class Levels2D : public Levels {
public:
    Levels2D() : Levels() {}

    virtual void initialize(Grid& grid, const int n_gpu) override {
        const double xmin = grid.get_xmin();
        const double xmax = grid.get_xmax();

        const double ymin = grid.get_ymin();
        const double ymax = grid.get_ymax();

        const int nx = grid.get_x().size();
        const int ny = grid.get_y().size();
        assert(nx == ny);

        const int n_levels = (int)std::floor(std::log2(nx));

        m_levels.clear();
        for (int i = 2, nl = n_levels - 1; i < nx; i += i, --nl) {
            bool on_gpu = nl < n_gpu;
            m_levels.push_back(Level(xmin, xmax, i+1, ymin, ymax, i+1, on_gpu));
        }
    }
};

// Alternating Directions
class Levels2DAlt : public Levels {
public:
    Levels2DAlt() : Levels() {}

    virtual void initialize(Grid& grid, const int n_gpu) override {
    }
};

} // namespace pmf
