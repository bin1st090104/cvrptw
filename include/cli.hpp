#pragma once

#include "problem.hpp"

namespace cvrptw
{
    struct Arguments
    {
        std::unique_ptr<Problem> problem;
        std::optional<size_t> limit;

        static Arguments parse(int argc, char **argv);
    };
}
