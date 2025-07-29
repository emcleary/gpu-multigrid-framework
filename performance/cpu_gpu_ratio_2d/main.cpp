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
    std::cerr << "Usage: command N S n_gpu cpu_threads n_burnin n_iter\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   S: solver ID\n";
    std::cerr << "      1: linear\n";
    std::cerr << "      2: nonlinear full\n";
    std::cerr << "   n_gpu: number of levels run on gpu\n";
    std::cerr << "   cpu_threads: number of threads to run on cpu\n";
    std::cerr << "   n_burnin: number of V-cycles for burnin\n";
    std::cerr << "   n_iter: number of V-cycles for timing\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 7)
        usage_error();

    const int n_levels = std::stoi(argv[1]);
    const size_t N = 1 + (1 << n_levels);
    const int solver_id = std::stoi(argv[2]);
    const int n_gpu = std::stoi(argv[3]);
    const int cpu_threads = std::stoi(argv[4]);
    const int n_burnin = std::stoi(argv[5]);
    const int n_iter = std::stoi(argv[6]);
    
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

    auto lhs = std::make_shared<LHSNaive>(gpu_threads, cpu_threads);
    auto iterator = std::make_shared<IteratorNaive>(gpu_threads, cpu_threads);
    auto restrictor = std::make_shared<pmf::modules::RestrictorFullWeighting2D>(gpu_threads, cpu_threads);
    auto interpolator = std::make_shared<pmf::modules::InterpolatorLinear2D>(gpu_threads, cpu_threads);

    auto norm = std::make_shared<pmf::modules::L2Norm2D>(gpu_threads, cpu_threads);
    // auto norm = std::make_shared<pmf::modules::NormAmax>(gpu_threads, cpu_threads);

    // auto boundary_conditions = std::make_shared<BoundaryConditionsHeatEqn>(eqn, grid);
    auto boundary_conditions = std::make_shared<pmf::modules::BoundaryConditions>();

    // boundary_conditions->set_west_dirichlet();
    boundary_conditions->set_west_neumann();

    // boundary_conditions->set_east_dirichlet();
    boundary_conditions->set_east_neumann();

    // boundary_conditions->set_north_dirichlet();
    boundary_conditions->set_north_neumann();

    boundary_conditions->set_south_dirichlet();
    // boundary_conditions->set_south_neumann();

    // boundary_conditions->set_periodic_x();
    boundary_conditions->set_periodic_y();

    solver->set_gpu_threads(gpu_threads);
    solver->set_cpu_threads(cpu_threads);
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

    std::cout << "GPU levels,CPU levels,ms/cycle\n";
    vcycle.initialize(*grid, n_gpu);
    float duration = pmf::measure_performance(function, n_burnin, n_iter);
    std::cout << "data," << n_gpu << "," << n_levels-n_gpu << "," << duration << '\n';
    return 0;
}
