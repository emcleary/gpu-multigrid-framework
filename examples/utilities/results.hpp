#pragma once

#include <string>

#include "src/array.hpp"
#include "src/modules/interfaces.hpp"


void dump_results(const gmf::Array& solution, gmf::modules::Equation& equation,
        const gmf::Array& x, std::string&& filename);

void dump_results(const gmf::Array& solution, gmf::modules::Equation& equation,
        const gmf::Array& x);
