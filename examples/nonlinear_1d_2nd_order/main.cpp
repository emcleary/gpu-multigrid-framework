#include <fstream>
#include <format>
#include <iostream>
#include <string>

#include "equation.hpp"
#include "iterator.cuh"
#include "lhs.cuh"

#include "src/array.hpp"
#include "src/cycles.hpp"
#include "src/grid.hpp"
#include "src/solver.cuh"
#include "src/timer.hpp"

#include "src/modules/interpolators.hpp"
#include "src/modules/norms.hpp"
#include "src/modules/restrictors.hpp"

#include "examples/utilities/command_line_parser.hpp"
#include "examples/utilities/results.hpp"


int main(int argc, char* argv[]) {
    CMD args = command_line_parser(argc, argv);
    const size_t N = args.n_points;
    std::shared_ptr<gmf::Solver> solver = args.solver;
    
    const double gamma = 10.0;

    const int nu0 = 2, nu1 = 1;
    const size_t max_threads_per_block = 512;

    const double xmin = 0.0;
    const double xmax = 1.0;
    gmf::Grid grid(xmin, xmax, N);

    auto eqn = std::make_shared<NonlinearEquation>(gamma);

    auto lhs = std::make_shared<LHSNonlinear>(max_threads_per_block, eqn);

    auto iterator = std::make_shared<IteratorNonlinear>(max_threads_per_block, eqn);
    
    auto restrictor = std::make_shared<gmf::modules::RestrictorFullWeighting>(max_threads_per_block);
    // auto restrictor = std::make_shared<gmf::modules::RestrictorInjection>();

    auto interpolator = std::make_shared<gmf::modules::InterpolatorLinear>(max_threads_per_block);

    auto norm = std::make_shared<gmf::modules::NormL2>();
    // auto norm = std::make_shared<gmf::modules::NormAmax>();

    solver->set_max_threads_per_block(max_threads_per_block);
    solver->set_equation(eqn);
    solver->set_iterator(iterator);
    solver->set_restrictor(restrictor);
    solver->set_residual_norm(norm);
    solver->set_interpolator(interpolator);
    solver->set_lhs(lhs);

    const int n_gpu = args.on_gpu ? args.n_levels : 0;
    gmf::VCycle vcycle;
    vcycle.set_solver(solver);
    vcycle.set_relaxation_iterations(2, 1);
    vcycle.initialize(grid, n_gpu);

    const gmf::Array& soln = solver->get_solution();

    gmf::TimerGPU timer;

    double duration = 0;
    dump_results(soln, *eqn, grid.get_x(), "results_vcycle_" + std::to_string(0) + ".csv");
    const int n_iter = 20;
    for (int i = 1; i <= n_iter; ++i) {
        timer.start();
        double resid_norm = vcycle.run();
        timer.stop();
        duration += timer.duration();
        std::cout << "Iteration " << i << " |r| = " << resid_norm << ", time = " << timer.duration() << " ms\n";
        dump_results(soln, *eqn, grid.get_x(), "results_vcycle_" + std::to_string(i) + ".csv");
    }
    std::cout << "Total time: " << duration << " ms\n";
    std::cout << "Average time: " << duration / n_iter << " ms\n";

    const gmf::Array& x = grid.get_x();
    gmf::Array solution_error(N);
    for (int i = 0; i < x.size(); ++i)
        solution_error[i] = soln[i] - eqn->analytical_solution(x[i]);

    double error_norm = norm->run_host(solution_error, grid);
    std::cout << "|e| = " << error_norm << '\n';

    return 0;
}
