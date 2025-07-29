#pragma once

#include <string>

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"


void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x, std::string&& filename);

void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x);

void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x, const pmf::Array& y, std::string&& filename);

void dump_results(const pmf::Array& solution, pmf::modules::Equation& equation,
        const pmf::Array& x, const pmf::Array& y);
