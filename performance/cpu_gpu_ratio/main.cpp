#include <fstream>
#include <format>
#include <iostream>
#include <string>

#include "equation.hpp"
#include "iterator_naive.cuh"
#include "lhs_naive.cuh"

#include "src/array.hpp"
#include "src/cycles.hpp"
#include "src/grid.hpp"
#include "src/solver.cuh"
#include "src/solver_linear.cuh"
#include "src/solver_nonlinear_full.cuh"
#include "src/solver_nonlinear_error.cuh"
#include "src/timer.hpp"

#include "src/modules/interpolators.hpp"
#include "src/modules/norms.hpp"
#include "src/modules/restrictors.hpp"

#include "examples/utilities/command_line_parser.hpp"
#include "examples/utilities/results.hpp"

void usage_error() {
    std::cerr << "Usage: command N S n_iter\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   S: solver ID\n";
    std::cerr << "      1: linear\n";
    std::cerr << "      2: nonlinear full\n";
    std::cerr << "      3: nonlinear error\n";
    std::cerr << "   n_iter: number of V-cycles per case\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 4)
        usage_error();

    const int n_levels = std::stoi(argv[1]);
    const size_t N = 1 + (1 << n_levels);
    const int solver_id = std::stoi(argv[2]);
    const int n_iter = std::stoi(argv[3]);
    
    std::shared_ptr<gmf::Solver> solver;
    switch (solver_id) {
    case 1:
        std::cout << "Linear solver\n";
        solver = std::make_shared<gmf::SolverLinear>();
        break;
    case 2:
        std::cout << "Nonlinear full solver\n";
        solver = std::make_shared<gmf::SolverNonlinearFull>();
        break;
    case 3:
        std::cout << "Nonlinear error solver\n";
        solver = std::make_shared<gmf::SolverNonlinearError>();
        break;
    default:
        std::cerr << "Invalid solver algorithm ID " << solver_id << "\n\n";
        usage_error();
    }

    const int nu0 = 2, nu1 = 1;
    const size_t max_threads_per_block = 512;

    const double xmin = 0.0;
    const double xmax = 1.0;
    gmf::Grid grid(xmin, xmax, N);

    auto eqn = std::make_shared<HeatEquation>();
    auto lhs = std::make_shared<LHSNaive>(max_threads_per_block);
    auto iterator = std::make_shared<IteratorNaive>(max_threads_per_block);
    auto restrictor = std::make_shared<gmf::modules::RestrictorFullWeighting>(max_threads_per_block);
    auto interpolator = std::make_shared<gmf::modules::InterpolatorLinear>(max_threads_per_block);
    auto norm = std::make_shared<gmf::modules::NormL2>();

    solver->set_max_threads_per_block(max_threads_per_block);
    solver->set_equation(eqn);
    solver->set_iterator(iterator);
    solver->set_restrictor(restrictor);
    solver->set_residual_norm(norm);
    solver->set_interpolator(interpolator);
    solver->set_lhs(lhs);

    gmf::VCycle vcycle;
    vcycle.set_solver(solver);
    vcycle.set_relaxation_iterations(2, 1);

    gmf::TimerGPU timer;

    std::cout << "\n\n\n";
    std::cout << "GPU levels   CPU levels   ms/cycle\n";
    for (int n_gpu = 0; n_gpu < n_levels; ++n_gpu) {
        vcycle.initialize(grid, n_gpu);

        double duration = 0;
        for (int i = 0; i < n_iter; ++i) {
            timer.start();
            vcycle.run();
            timer.stop();
            duration += timer.duration();
        }
        duration /= n_iter;
        std::cout << std::setw(10) << n_gpu << "   ";
        std::cout << std::setw(10) << n_levels - n_gpu << "   ";
        std::cout << std::setw(10) << duration << '\n';
    }

    return 0;
}
