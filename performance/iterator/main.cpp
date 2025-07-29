#include <iostream>
#include <functional>
#include <string>

#include "examples/heat_equation_1d_2nd_order/iterator_naive.cuh"
#include "examples/heat_equation_1d_2nd_order/iterator_async.cuh"
#include "examples/heat_equation_1d_2nd_order/equation.hpp"

#include "iterator_async_smem.cuh"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/timer.hpp"
#include "src/modules/interfaces/iterator.cuh"
#include "src/modules/norms.hpp"
#include "src/utilities.hpp"


static const int N_ITER = 100;

void usage_error() {
    std::cerr << "Usage: command N I n_gpu n_burnin n_iter\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   I: iterator algorithm ID\n";
    std::cerr << "      1: naive\n";
    std::cerr << "      2: asynchronous\n";
    std::cerr << "      3: asynchronous with shared memory\n";
    std::cerr << "   n_burnin: number of V-cycles for burnin\n";
    std::cerr << "   n_iter: number of V-cycles for timing\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 5)
        usage_error();
    
    const int n_levels = std::stoi(argv[1]);
    const size_t N = 1 + (1 << n_levels);
    const int iter_id = std::stoi(argv[2]);
    const int n_burnin = std::stoi(argv[3]);
    const int n_iter = std::stoi(argv[4]);
    
    const double xmin = 0.0;
    const double xmax = 1.0;
    const size_t max_threads_per_block = 512;

    std::cout << "Grid size: " << N << '\n';
    
    std::unique_ptr<pmf::modules::Iterator> iterator;
    switch (iter_id) {
    case 1:
        std::cout << "Naive iterator algorithm\n";
        iterator = std::make_unique<IteratorNaive>(max_threads_per_block);
        break;
    case 2:
        std::cout << "Asynchronous iterator algorithm\n";
        iterator = std::make_unique<IteratorAsync>(max_threads_per_block);
        break;
    case 3:
        std::cout << "Asynchronous iterator algorithm with shared memory\n";
        iterator = std::make_unique<IteratorAsyncSMEM>(max_threads_per_block);
        break;
    default:
        std::cerr << "Invalid iterator algorithm ID " << iter_id << "\n\n";
        usage_error();
    }

    pmf::Grid grid(xmin, xmax, N);
    pmf::Array v(grid.size());
    pmf::Array f(grid.size());
    pmf::modules::BoundaryConditions boundary_conditions;
    boundary_conditions.set_periodic_x();

    HeatEquation eqn;
    const pmf::Array& x = grid.get_x();
    for (int i = 0; i < grid.size(); ++i) {
        v[i] = eqn.initial_condition(x[i]);
        f[i] = eqn.rhs(x[i]);
    }

    std::function<void()> function = [&]() { iterator->run_device(v, f, boundary_conditions, grid); };
    float duration = pmf::measure_performance(function, n_burnin, n_iter);
    std::cout << "Average time per iteration " << duration << '\n';

    pmf::Array solution_error(N);
    for (int i = 0; i < x.size(); ++i) {
        solution_error[i] = v[i] - eqn.analytical_solution(x[i]);
    }

    pmf::modules::NormL2 norm;
    double error_norm = norm.run_host(solution_error, grid);
    std::cout << "|e| = " << error_norm << '\n';

    return 0;
}
