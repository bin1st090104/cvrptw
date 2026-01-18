#include "cli.hpp"

namespace cvrptw
{
    Arguments Arguments::parse(int argc, char **argv)
    {
        if (argc < 2)
        {
            throw std::runtime_error(std::format("Usage: {} <problem_file> [limit]", argv[0]));
        }

        std::optional<size_t> limit;
        if (argc >= 3)
        {
            limit = std::stoull(argv[2]);
        }

        auto problem = Problem::from_file(argv[1], limit.value_or(-1));
        return Arguments{std::move(problem), limit};
    }
}