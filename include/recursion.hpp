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

        virtual std::vector<Route> _solve(const Problem &problem, uint64_t stack, uint64_t &result) = 0;

    public:
        explicit RecursionState(const Problem &problem) : _routes(), _assigned(problem.customers_count(), false)
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
                result = std::max(result, route.total_time());
            }
            return result;
        }

        inline const std::vector<Route> &routes() const
        {
            return _routes;
        }

        inline std::vector<Route> solve(const Problem &problem, uint64_t &result)
        {
            return _solve(problem, 1, result);
        }
    };
}