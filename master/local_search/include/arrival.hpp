#pragma once

#include "pch.hpp"

namespace cvrptw
{
    struct CustomerArrival
    {
        size_t customer;

        /// @brief Contrary to its name, this is the time when the customer is served,
        /// which does not necessarily equal to the vehicle arrives.
        ///
        /// Note that: arrival_time + service_time = departure_time
        uint64_t arrival_time;

        explicit CustomerArrival(size_t customer, uint64_t arrival_time) : customer(customer), arrival_time(arrival_time) {}
    };
}
