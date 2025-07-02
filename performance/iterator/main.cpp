#include <iostream>
#include <functional>
#include <string>

#include "iterator_naive.cuh"
#include "iterator_async.cuh"
#include "iterator_async_smem.cuh"
#include "equation.hpp"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/timer.hpp"
#include "src/modules/interfaces/iterator.cuh"
#include "src/modules/norms.hpp"
#include "src/utilities.hpp"


static const int N_ITER = 100;

void usage_error() {
    std::cerr << "Usage: command N S n_gpu n_burnin n_iter\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   I: iterator algorithm ID\n";
    std::cerr << "      1: naive\n";
    std::cerr << "      2: asynchronous\n";
    std::cerr << "      3: asynchronous with shared memory\n";
    std::cerr << "   n_gpu: number of levels run on gpu\n";
    std::cerr << "   n_burnin: number of V-cycles for burnin\n";
    std::cerr << "   n_iter: number of V-cycles for timing\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 6)
        usage_error();
    
    const int n_levels = std::stoi(argv[1]);
    const size_t N = 1 + (1 << n_levels);
    const int iter_id = std::stoi(argv[2]);
    const int n_gpu = std::stoi(argv[3]);
    const int n_burnin = std::stoi(argv[4]);
    const int n_iter = std::stoi(argv[5]);
    
    const double xmin = 0.0;
    const double xmax = 1.0;
    const size_t max_threads_per_block = 512;

    std::cout << "Grid size: " << N << '\n';
    
    std::unique_ptr<gmf::modules::Iterator> iterator;
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

    gmf::Grid grid(xmin, xmax, N);
    gmf::Array v(grid.size());
    gmf::Array f(grid.size());
    gmf::modules::BoundaryConditions boundary_conditions;
    boundary_conditions.set_periodic();

    HeatEquation eqn;
    const gmf::Array& x = grid.get_x();
    for (int i = 0; i < grid.size(); ++i) {
        v[i] = eqn.initial_condition(x[i]);
        f[i] = eqn.rhs(x[i]);
    }

    std::function<void()> function = [&]() { iterator->run_device(v, f, boundary_conditions, grid); };
    float duration = gmf::measure_performance(function, n_burnin, n_iter);
    std::cout << "Duration " << duration << '\n';
    
    gmf::Array solution_error(N);
    for (int i = 0; i < x.size(); ++i) {
        solution_error[i] = v[i] - eqn.analytical_solution(x[i]);
    }

    gmf::modules::NormL2 norm;
    double error_norm = norm.run_host(solution_error, grid);
    std::cout << "|e| = " << error_norm << '\n';

    return 0;
}
