#pragma once

#include "src/solver.cuh"

struct CMD {
    const int n_levels;
    const int n_points;
    const bool on_gpu;
    std::shared_ptr<gmf::Solver> solver;
};

void usage_error();

CMD command_line_parser(int argc, char* argv[]);
