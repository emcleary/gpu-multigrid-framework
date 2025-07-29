#include <iostream>
#include <string>

#include "equation.cuh"
#include "iterator_naive.cuh"
#include "lhs_naive.cuh"

#include "src/array.hpp"
#include "src/cycles.hpp"
#include "src/grid.hpp"
#include "src/solver.cuh"
#include "src/timer.hpp"
// TODO: make this a module/interface?
#include "src/levels.hpp"

#include "src/modules/interpolators.hpp"
#include "src/modules/norms.hpp"
#include "src/modules/restrictors.hpp"

#include "examples/utilities/command_line_parser.hpp"
#include "examples/utilities/results.hpp"


int main(int argc, char* argv[]) {
    CMD args = command_line_parser(argc, argv);
    const size_t N = args.n_points;
    std::shared_ptr<pmf::Solver> solver = args.solver;
    
    const int nu0 = 2, nu1 = 1;
    const uint gpu_threads = 512;
    const uint cpu_threads = 4;

    const double gamma = 10.0;

    const double xmin = 0.0;
    const double xmax = 1.0;
    const double ymin = 0.0;
    const double ymax = 1.0;
    auto grid = std::make_shared<pmf::Grid>(xmin, xmax, N, ymin, ymax, N);

    auto levels = std::make_shared<pmf::Levels2D>();
    auto eqn = std::make_shared<NonlinearEquation>(gamma);
    auto lhs = std::make_shared<LHSNaive>(gpu_threads, cpu_threads, eqn);
    auto iterator = std::make_shared<IteratorNaive>(gpu_threads, cpu_threads, eqn);
    auto restrictor = std::make_shared<pmf::modules::RestrictorFullWeighting2D>(gpu_threads, cpu_threads);
    auto interpolator = std::make_shared<pmf::modules::InterpolatorLinear2D>(gpu_threads, cpu_threads);

    auto norm = std::make_shared<pmf::modules::L2Norm2D>();
    // auto norm = std::make_shared<pmf::modules::NormAmax>();

    auto boundary_conditions = std::make_shared<pmf::modules::BoundaryConditions>();

    // boundary_conditions->set_west_dirichlet();
    boundary_conditions->set_west_neumann();
    boundary_conditions->set_east_dirichlet();
    boundary_conditions->set_north_dirichlet();
    boundary_conditions->set_south_dirichlet();

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

    const int n_gpu = args.on_gpu ? args.n_levels : 0;
    pmf::VCycle vcycle;
    vcycle.set_solver(solver);
    vcycle.set_relaxation_iterations(2, 1);
    vcycle.initialize(*grid, n_gpu);

    const pmf::Array& soln = solver->get_solution();

    pmf::TimerGPU timer;

    const pmf::Array& x = grid->get_x();
    const pmf::Array& y = grid->get_y();
    pmf::Array solution_error(N, N);

    double duration = 0;
    dump_results(soln, *eqn, grid->get_x(), grid->get_y(), "results_vcycle_" + std::to_string(0) + ".csv");
    const int n_iter = 20;
    // const int n_iter = 7000;
    for (int i = 1; i <= n_iter; ++i) {
        timer.start();
        double resid_norm = vcycle.run();
        timer.stop();
        duration += timer.duration();
        std::cout << "Iteration " << i << " |r| = " << resid_norm << ", time = " << timer.duration() << " ms\n";
        // if (i % 1000 == 0)
        //     dump_results(soln, *eqn, grid->get_x(), grid->get_y(), "results_vcycle_" + std::to_string(i) + ".csv");
        // dump_results(soln, *eqn, grid->get_x(), grid->get_y(), "results_vcycle_" + std::to_string(i) + ".csv");

        // for (int i = 0; i < x.size(); ++i)
        //     for (int j = 0; j < y.size(); ++j)
        //         solution_error(i, j) = soln(i, j) - eqn->analytical_solution(x[i], y[j]);

        // double error_norm = norm->run_host(solution_error, *grid);
        // std::cout << "|e| = " << error_norm << '\n';
    }
    std::cout << "Total time: " << duration << " ms\n";
    std::cout << "Average time: " << duration / n_iter << " ms\n";

    for (int i = 0; i < x.size(); ++i)
        for (int j = 0; j < y.size(); ++j)
            solution_error(i, j) = soln(i, j) - eqn->analytical_solution(x[i], y[j]);

    double error_norm = norm->run_host(solution_error, *grid);
    std::cout << "|e| = " << error_norm << '\n';

    return 0;
}
