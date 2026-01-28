#pragma once

#include "problem.hpp"
#include "route.hpp"

namespace cvrptw
{
    class RecursionState
    {
    protected:
        std::vector<Route> _routes;
        std::vector<bool> _assigned;
        std::chrono::milliseconds _time_limit;
        std::chrono::steady_clock::time_point _timer;

        virtual std::vector<Route> _solve(const Problem &problem, uint64_t stack, uint64_t &result) = 0;

    public:
        explicit RecursionState(std::chrono::milliseconds time_limit, const Problem &problem)
            : _routes(), _assigned(problem.customers_count(), false), _time_limit(time_limit)
        {
            _assigned[problem.depot] = true;

            _routes.reserve(problem.vehicles_count());
            for (size_t v = 0; v < problem.vehicles_count(); v++)
            {
                _routes.emplace_back(v, problem);
            }
        }

        uint64_t evaluate() const
        {
            uint64_t result = 0;
            for (const auto &route : _routes)
            {
                result += route.total_cost();
            }
            return result;
        }

        inline const std::vector<Route> &routes() const
        {
            return _routes;
        }

        inline std::chrono::milliseconds elapsed() const
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - _timer);
        }

        inline std::vector<Route> solve(const Problem &problem, uint64_t &result)
        {
            _timer = std::chrono::steady_clock::now();
            return _solve(problem, 1, result);
        }
    };
}
