#include <iostream>
#include <string>

#include "lhs_naive.cuh"
#include "lhs_smem.cuh"
#include "equation.hpp"

#include "src/array.hpp"
#include "src/grid.hpp"
#include "src/timer.hpp"
#include "src/modules/interfaces/lhs.cuh"


static const int N_ITER = 100;

void usage_error() {
    std::cerr << "Usage: command N I <n_iter>\n";
    std::cerr << "   N: number of levels\n";
    std::cerr << "   L: lhs algorithm ID\n";
    std::cerr << "      1: naive\n";
    std::cerr << "      2: shared memory\n";
    std::cerr << "   n_iter: number of iterations (default " << N_ITER << ")\n";
    exit(EXIT_FAILURE);
}


int main(int argc, char* argv[]) {
    if (argc < 3)
        usage_error();
    
    const size_t N = 1 + (1 << std::stoi(argv[1]));
    const int lhs_id = std::stoi(argv[2]);
    const int n_iter = argc > 3 ? std::stoi(argv[3]) : N_ITER;
    const size_t max_threads_per_block = 512;
    
    std::unique_ptr<gmf::modules::LHS> lhs;
    switch (lhs_id) {
    case 1:
        std::cout << "Naive algorithm\n";
        lhs = std::make_unique<LHSNaive>(max_threads_per_block);
        break;
    case 2:
        std::cout << "Shared memory algorithm\n";
        lhs = std::make_unique<LHSSMEM>(max_threads_per_block);
        break;
    default:
        std::cerr << "Invalid LHS ID " << lhs_id << "\n\n";
        usage_error();
    }

    const double xmin = 0.0;
    const double xmax = 1.0;
    gmf::Grid grid(xmin, xmax, N);

    gmf::Array result(grid.size());
    gmf::Array v(grid.size());
    gmf::Array f(grid.size());

    HeatEquation eqn;
    const gmf::Array& x = grid.get_x();
    for (int i = 0; i < grid.size(); ++i) {
        v[i] = eqn.initial_condition(x[i]);
        f[i] = eqn.rhs(x[i]);
    }

    float duration = 1e10;
    gmf::TimerGPU timer;
    for (int i = 0; i < n_iter; ++i) {
        timer.start();
        lhs->run_device(result, v, f, grid);
        timer.stop();
        if (timer.duration() > 1e-5)
            duration = std::min(duration, timer.duration());
    }
    std::cout << "Duration " << duration << '\n';
    
    return 0;
}
