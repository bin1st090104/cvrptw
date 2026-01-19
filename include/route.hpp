#pragma once

#include "arrival.hpp"
#include "problem.hpp"

namespace cvrptw
{
    struct Route
    {
    private:
        std::vector<CustomerArrival> _customers;
        size_t _vehicle;
        uint64_t _total_time;
        uint64_t _total_demand;

        static bool _recalculate_arrival_times(std::vector<CustomerArrival> &customers, const cvrptw::Problem &problem, size_t offset = 0);

    public:
        explicit Route(size_t vehicle, const Problem &problem)
            : _customers(), _vehicle(vehicle), _total_time(0), _total_demand(0)
        {
            _customers.emplace_back(problem.depot, 0);
        }

        std::vector<size_t> get_customers() const;

        inline bool empty() const noexcept
        {
            return _customers.size() == 1;
        }

        inline uint64_t total_time() const noexcept
        {
            return _total_time;
        }

        bool try_assign(size_t customer, const cvrptw::Problem &problem, uint64_t best = 0);
        void unassign(const cvrptw::Problem &problem);
        bool two_opt(Route *other, const cvrptw::Problem &problem);
    };
}
