#include "results.hpp"

#include <format>
#include <fstream>
#include <iostream>


void dump_results(const gmf::Array& solution, gmf::modules::Equation& equation,
        const gmf::Array& x, std::string&& filename) {

    assert(solution.size() == x.size());
    std::ofstream outfile;
    outfile.open(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening results file.\n";
        exit(EXIT_FAILURE);
    }

    outfile << "#x,analytical solution,numerical solution\n";
    for (int i = 0; i < x.size(); ++i) {
        double u = equation.analytical_solution(x[i]);
        outfile << std::format("{}, {}, {}\n", x[i], u, solution[i]);
    }
    outfile.close();
    std::cout << "Results written to " << filename << '\n';
}

void dump_results(const gmf::Array& solution, gmf::modules::Equation& equation, const gmf::Array& x) {
    dump_results(solution, equation, x, "results.csv");
}
