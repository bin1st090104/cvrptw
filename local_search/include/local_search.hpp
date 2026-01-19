#pragma once

#include "recursion.hpp"

#ifdef _WIN32
#ifdef LOCAL_SEARCH_EXPORTS
#define LS_API __declspec(dllexport)
#else
#define LS_API __declspec(dllimport)
#endif
#else
#define LS_API
#endif

extern "C"
{
    LS_API uint8_t solve(
        uint64_t vehicles_count,
        uint64_t customers_count,
        uint64_t *capacities,
        uint64_t **time_matrix,
        uint64_t *demands,
        uint64_t *ready_times,
        uint64_t *due_dates,
        uint64_t *service_times,
        uint64_t depot,
        uint64_t time_limit_ms,
        uint64_t *out_cost,
        uint64_t *out_elapsed_ms,
        uint8_t *out_timed_out,
        const char **out_error_message,
        const char **out_status);
}
