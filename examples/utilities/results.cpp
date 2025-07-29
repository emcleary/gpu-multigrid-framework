#include "results.hpp"

#include <format>
#include <fstream>
#include <iostream>


void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x, std::string&& filename) {

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

void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation, const pmf::Array& x) {
    dump_results(solution, equation, x, "results.csv");
}

void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x, const pmf::Array& y, std::string&& filename) {

    assert(solution.size() == x.size() * y.size());
    std::ofstream outfile;
    outfile.open(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening results file.\n";
        exit(EXIT_FAILURE);
    }

    outfile << "#x,y,analytical solution,numerical solution\n";
    for (int i = 0; i < x.size(); ++i) {
        for (int j = 0; j < y.size(); ++j) {
            double u = equation.analytical_solution(x[i], y[j]);
            outfile << std::format("{}, {}, {}, {}\n", x[i], y[j], u, solution(i, j));
        }
    }
    outfile.close();
    std::cout << "Results written to " << filename << '\n';
}

void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x, const pmf::Array& y) {
    dump_results(solution, equation, x, y, "results.csv");
}
