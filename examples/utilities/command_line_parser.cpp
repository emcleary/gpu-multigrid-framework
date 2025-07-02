#include "examples/utilities/command_line_parser.hpp"

#include "src/solver.cuh"
#include "src/solver_linear.cuh"
#include "src/solver_nonlinear_full.cuh"


void usage_error() {
    std::cerr << "Usage: command N S G\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   S: solver ID\n";
    std::cerr << "      1: linear\n";
    std::cerr << "      2: nonlinear full\n";
    std::cerr << "   G: 0 for cpu only, otherwise gpu\n";
    exit(EXIT_FAILURE);
}

CMD command_line_parser(int argc, char* argv[]) {
    if (argc != 4)
        usage_error();

    std::shared_ptr<gmf::Solver> solver;
    switch (std::stoi(argv[2])) {
    case 1:
        std::cout << "Linear solver\n";
        solver = std::make_shared<gmf::SolverLinear>();
        break;
    case 2:
        std::cout << "Nonlinear full solver\n";
        solver = std::make_shared<gmf::SolverNonlinearFull>();
        break;
    default:
        std::cerr << "Solver ID " << argv[2] << " not allowed.\n\n";
        usage_error();
    }

    const int num_levels = std::stoi(argv[1]);
    std::cout << num_levels << " levels\n";

    const int num_points = 1 + (1 << num_levels);
    std::cout << num_points << " grid points\n";

    const bool on_gpu = std::stoi(argv[3]) != 0;
    std::cout << (on_gpu ? "on GPU" : "on CPU") << '\n';

    return CMD(num_levels, num_points, on_gpu, solver);
}
