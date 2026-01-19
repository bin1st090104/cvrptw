#include "local_search.hpp"

class RecursionState : public cvrptw::RecursionState
{
protected:
    uint64_t _first_result;

    std::vector<cvrptw::Route> _solve(const cvrptw::Problem &problem, uint64_t stack, uint64_t &cost) override
    {
        std::vector<cvrptw::Route> best_routes;
        cost = uint64_t(-1);

        if (_first_result != uint64_t(-1))
        {
            return best_routes;
        }

        if (std::chrono::steady_clock::now() - _timer >= _time_limit)
        {
            timed_out = true;
            return best_routes;
        }

        if (stack == problem.customers_count())
        {
            _first_result = cost = evaluate();
            best_routes = _routes;
        }
        else
        {
            uint64_t cost_change_ignored;
            for (size_t customer = 0; customer < problem.customers_count(); customer++)
            {
                if (_assigned[customer])
                {
                    continue;
                }

                for (auto &route : _routes)
                {
                    bool initially_empty = route.empty();
                    if (route.try_assign(customer, problem, cost_change_ignored))
                    {
                        _assigned[customer] = true;

                        uint64_t r;
                        auto routes = _solve(problem, stack + 1, r);
                        if (r < cost)
                        {
                            best_routes = routes;
                            cost = r;
                        }

                        route.unassign(problem);
                        _assigned[customer] = false;
                    }

                    if (initially_empty)
                    {
                        break;
                    }
                }
            }
        }

        return best_routes;
    }

public:
    bool timed_out;

    explicit RecursionState(std::chrono::milliseconds time_limit, const cvrptw::Problem &problem)
        : cvrptw::RecursionState(time_limit, problem), _first_result(-1), timed_out(false) {}
};

const char *ERROR_NO_FEASIBLE_SOLUTION = "No feasible solution";
const char *STATUS_FEASIBLE = "FEASIBLE";
const char *STATUS_TIMEOUT = "TIMEOUT";

extern "C" LS_API uint8_t solve(
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
    const char **out_status)
{
    std::vector<std::vector<uint64_t>> time_matrix_vec;
    for (size_t i = 0; i < customers_count; i++)
    {
        time_matrix_vec.emplace_back(time_matrix[i], time_matrix[i] + customers_count);
    }

    auto problem = std::make_unique<cvrptw::Problem>(
        "external_problem",
        std::vector<uint64_t>(capacities, capacities + vehicles_count),
        std::move(time_matrix_vec),
        std::vector<uint64_t>(demands, demands + customers_count),
        std::vector<std::pair<uint64_t, uint64_t>>(),
        std::vector<uint64_t>(service_times, service_times + customers_count),
        depot);

    std::chrono::milliseconds time_limit(time_limit_ms);
    RecursionState state(time_limit, *problem);

    uint64_t cost;
    auto routes = state.solve(*problem, cost);

    if (cost == uint64_t(-1))
    {
        *out_error_message = ERROR_NO_FEASIBLE_SOLUTION;
        return false;
    }

    bool improved = true;
    while (improved && state.elapsed() < time_limit)
    {
        while (improved && state.elapsed() < time_limit)
        {
            improved = false;
            for (auto &route : routes)
            {
                for (auto &other_route : routes)
                {
                    improved = improved || route.move_10(&other_route, *problem);
                }
            }
        }

        // Here, either improved = false, or we exceeded the time limit
        if (state.elapsed() >= time_limit)
        {
            break;
        }

        // improved = false here, try 2-opt
        for (auto &route : routes)
        {
            for (auto &other_route : routes)
            {
                improved = improved || route.two_opt(&other_route, *problem);
            }
        }
    }

    auto elapsed = state.elapsed();
    *out_elapsed_ms = elapsed.count();
    *out_timed_out = state.timed_out = state.elapsed() >= time_limit;

    *out_cost = 0;
    for (const auto &route : routes)
    {
        *out_cost += route.total_cost();
    }

    *out_status = (*out_cost == -1 ? STATUS_TIMEOUT : STATUS_FEASIBLE);

    // TODO: Return route as well
    // std::vector<std::vector<size_t>> route_customers;
    // for (auto &route : routes)
    // {
    //     route_customers.push_back(route.get_customers());
    // }

    // std::cout << route_customers << "}" << std::endl;

    return true;
}
