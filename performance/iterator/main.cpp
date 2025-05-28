#include <iostream>
#include <string>

#include "iterator_naive.cuh"
#include "iterator_async.cuh"
#include "iterator_async_smem.cuh"
#include "equation.hpp"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/timer.hpp"
#include "src/modules/interfaces/iterator.cuh"


static const int N_ITER = 100;

void usage_error() {
    std::cerr << "Usage: command N I <n_iter>\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   I: iterator algorithm ID\n";
    std::cerr << "      1: naive\n";
    std::cerr << "      2: asynchronous\n";
    std::cerr << "      3: asynchronous with shared memory\n";
    std::cerr << "   n_iter: number of iterations (default " << N_ITER << ")\n";
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 3)
        usage_error();
    
    const size_t N = 1 + (1 << std::stoi(argv[1]));
    const int iter_id = std::stoi(argv[2]);
    const int n_iter = argc > 3 ? std::stoi(argv[3]) : N_ITER;
    
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

    HeatEquation eqn;
    const gmf::Array& x = grid.get_x();
    for (int i = 0; i < grid.size(); ++i) {
        v[i] = eqn.initial_condition(x[i]);
        f[i] = eqn.rhs(x[i], v[i]);
    }

    float duration = 1e10;
    gmf::TimerGPU timer;
    for (int i = 0; i < n_iter; ++i) {
        timer.start();
        iterator->run_device(v, f, grid);
        timer.stop();
        duration = std::min(duration, timer.duration());
    }
    std::cout << "Duration " << duration << '\n';
    
    return 0;
}
