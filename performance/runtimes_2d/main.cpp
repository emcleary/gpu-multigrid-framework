#include <fstream>
#include <format>
#include <iostream>
#include <string>

#include "src/array.hpp"
#include "src/cycles.hpp"
#include "src/grid.hpp"
#include "src/solver.cuh"
#include "src/solver_linear.cuh"
#include "src/solver_nonlinear_full.cuh"
#include "src/timer.hpp"
#include "src/utilities.hpp"

#include "src/modules/interpolators.hpp"
#include "src/modules/norms.hpp"
#include "src/modules/restrictors.hpp"

#include "examples/utilities/command_line_parser.hpp"
#include "examples/utilities/results.hpp"

#include "examples/heat_equation_2d_2nd_order/equation.hpp"
#include "examples/heat_equation_2d_2nd_order/iterator_naive.cuh"
#include "examples/heat_equation_2d_2nd_order/lhs_naive.cuh"
#include "examples/utilities/results.hpp"

void usage_error() {
    std::cerr << "Usage: command N S [cpu_threads_1 cpu_threads_2]\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   S: solver ID\n";
    std::cerr << "      1: linear\n";
    std::cerr << "      2: nonlinear full\n";
    std::cerr << "   cpu_threads_1, cpu_threads_2: number of threads to run on cpu\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 3)
        usage_error();

    const int n_levels = std::stoi(argv[1]);
    const size_t N = 1 + (1 << n_levels);
    const int solver_id = std::stoi(argv[2]);
    int n_gpu, cpu_threads_1, cpu_threads_2;
    if (argc == 3) {
        n_gpu = n_levels;
        cpu_threads_1 = 1;
        cpu_threads_2 = 1;
    } else if (argc == 4) {
        n_gpu = 0;
        cpu_threads_1 = std::stoi(argv[3]);
        cpu_threads_2 = cpu_threads_1;
    } else if (argc == 5) {
        n_gpu = 0;
        cpu_threads_1 = std::stoi(argv[3]);
        cpu_threads_2 = std::stoi(argv[4]);
    } else {
        usage_error();
    }

    const int n_burnin = 10;
    const int n_iter = 200;
    
    std::shared_ptr<pmf::Solver> solver;
    switch (solver_id) {
    case 1:
        std::cout << "Linear solver\n";
        solver = std::make_shared<pmf::SolverLinear>();
        break;
    case 2:
        std::cout << "Nonlinear full solver\n";
        solver = std::make_shared<pmf::SolverNonlinearFull>();
        break;
    default:
        std::cerr << "Invalid solver algorithm ID " << solver_id << "\n\n";
        usage_error();
    }

    const int nu0 = 2, nu1 = 1;
    const uint gpu_threads = 512;

    const double xmin = 0.0;
    const double xmax = 1.0;
    const double ymin = 0.0;
    const double ymax = 1.0;
    auto grid = std::make_shared<pmf::Grid>(xmin, xmax, N, ymin, ymax, N);

    auto levels = std::make_shared<pmf::Levels2D>();
    auto eqn = std::make_shared<HeatEquation>();

    auto lhs = std::make_shared<LHSNaive>(gpu_threads, cpu_threads_1);
    auto norm = std::make_shared<pmf::modules::L2Norm2D>(gpu_threads, cpu_threads_1);

    auto iterator = std::make_shared<IteratorNaive>(gpu_threads, cpu_threads_2);
    auto restrictor = std::make_shared<pmf::modules::RestrictorFullWeighting2D>(gpu_threads, cpu_threads_2);
    auto interpolator = std::make_shared<pmf::modules::InterpolatorLinear2D>(gpu_threads, cpu_threads_2);

    auto boundary_conditions = std::make_shared<pmf::modules::BoundaryConditions>();
    boundary_conditions->set_west_dirichlet();
    boundary_conditions->set_east_neumann();
    boundary_conditions->set_periodic_y();

    solver->set_gpu_threads(gpu_threads);
    solver->set_cpu_threads(cpu_threads_1);
    solver->set_levels(levels);
    solver->set_equation(eqn);
    solver->set_iterator(iterator);
    solver->set_restrictor(restrictor);
    solver->set_residual_norm(norm);
    solver->set_interpolator(interpolator);
    solver->set_lhs(lhs);
    solver->set_boundary_conditions(boundary_conditions);

    pmf::VCycle vcycle;
    vcycle.set_solver(solver);
    vcycle.set_relaxation_iterations(2, 1);

    std::function<void()> function = [&]() { vcycle.run(); };

    std::cout << "device,levels,ms/cycle\n";
    vcycle.initialize(*grid, n_gpu);
    float duration = pmf::measure_performance(function, n_burnin, n_iter);
    if (n_gpu > 0)
        std::cout << "GPU,";
    else if (cpu_threads_1 > 1 || cpu_threads_2 > 1)
        std::cout << "CPU parallel,";
    else
        std::cout << "CPU serial,";
    std::cout << n_levels << "," << duration << '\n';
    return 0;
}
