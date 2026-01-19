#pragma once

#include "pch.hpp"

namespace cvrptw
{
    class Problem
    {
    public:
        std::string name;
        std::vector<uint64_t> capacities;
        std::vector<std::vector<uint64_t>> time_matrix;
        std::vector<uint64_t> demands;
        std::vector<std::pair<uint64_t, uint64_t>> time_windows;
        std::vector<uint64_t> service_times;
        size_t depot;

        explicit Problem(
            std::string &&name,
            std::vector<uint64_t> &&capacities,
            std::vector<std::vector<uint64_t>> &&time_matrix,
            std::vector<uint64_t> &&demands,
            std::vector<std::pair<uint64_t, uint64_t>> &&time_windows,
            std::vector<uint64_t> &&service_times,
            size_t depot)
            : name(std::move(name)),
              capacities(std::move(capacities)),
              time_matrix(std::move(time_matrix)),
              demands(std::move(demands)),
              time_windows(std::move(time_windows)),
              service_times(std::move(service_times)),
              depot(depot) {}

        static std::unique_ptr<Problem> from_file(const std::filesystem::path &path, size_t limit = -1);

        inline size_t vehicles_count() const noexcept
        {
            return capacities.size();
        }

        inline size_t customers_count() const noexcept
        {
            return demands.size();
        }
    };
}
