#pragma once

#include "pch.hpp"

namespace cvrptw
{
    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    T sqrt(const T &value)
    {
        if (value < static_cast<T>(0))
        {
            throw std::out_of_range(std::format("Attempted to calculate square root of {} < 0", value));
        }

        T low = 0, high = std::max(static_cast<T>(1), value), error = 1;
        if constexpr (std::is_floating_point_v<T>)
        {
            error = 1.0e-7;
        }

        if (high * high == value)
        {
            return high;
        }

        while (high - low > error)
        {
            T mid = (low + high) / 2;
            if (mid * mid > value)
            {
                high = mid;
            }
            else
            {
                low = mid;
            }
        }

        return low;
    }
}