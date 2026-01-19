#pragma once

#include "problem.hpp"

namespace cvrptw
{
    struct Arguments
    {
        std::unique_ptr<Problem> problem;
        std::optional<size_t> limit;
        std::optional<std::chrono::milliseconds> time_limit;

        static Arguments parse(int argc, char **argv);
    };
}
