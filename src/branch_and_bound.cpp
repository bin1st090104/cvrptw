#include "cli.hpp"
#include "recursion.hpp"

class RecursionState : public cvrptw::RecursionState
{
protected:
    uint64_t _best;
    uint64_t _min_distance;
    uint64_t _cost;

    std::vector<cvrptw::Route> _solve(const cvrptw::Problem &problem, uint64_t stack, uint64_t &result) override
    {
        std::vector<cvrptw::Route> best_routes;
        result = uint64_t(-1);

        if (std::chrono::steady_clock::now() - _timer >= _time_limit)
        {
            timed_out = true;
            return best_routes;
        }

        auto left = problem.customers_count() - stack;
        if (_cost + left * _min_distance >= _best)
        {
            return best_routes;
        }

        if (left == 0)
        {
            result = evaluate();
            _best = std::min(_best, result);
            best_routes = _routes;
        }
        else
        {
            uint64_t cost_change;
            for (size_t customer = 0; customer < problem.customers_count(); customer++)
            {
                if (_assigned[customer])
                {
                    continue;
                }

                for (auto &route : _routes)
                {
                    bool initially_empty = route.empty();
                    if (route.try_assign(customer, problem, cost_change))
                    {
                        _cost += cost_change;
                        _assigned[customer] = true;

                        uint64_t r;
                        auto routes = _solve(problem, stack + 1, r);
                        if (r < result)
                        {
                            best_routes = routes;
                            result = r;
                        }

                        _assigned[customer] = false;
                        _cost -= cost_change;

                        route.unassign(problem);
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

    explicit RecursionState(std::chrono::milliseconds time_limit, uint64_t min_distance, const cvrptw::Problem &problem)
        : cvrptw::RecursionState(time_limit, problem), _best(-1), _min_distance(min_distance), _cost(0), timed_out(false) {}
};

int main(int argc, char **argv)
{
    auto arguments = cvrptw::Arguments::parse(argc, argv);
    auto problem = std::move(arguments.problem);
    std::cerr << argv[0] << " loaded " << problem->name << " with "
              << problem->customers_count() << " customers (including depot) and "
              << problem->vehicles_count() << " vehicles" << std::endl;

    uint64_t min_distance = -1;
    for (size_t i = 0; i < problem->customers_count(); i++)
    {
        for (size_t j = 0; j < problem->customers_count(); j++)
        {
            if (i != j)
            {
                min_distance = std::min(min_distance, problem->time_matrix[i][j]);
            }
        }
    }

    std::chrono::milliseconds time_limit = arguments.time_limit.value_or(std::chrono::milliseconds(86400000));
    RecursionState state(time_limit, min_distance, *problem);

    uint64_t cost;
    auto routes = state.solve(*problem, cost);
    auto elapsed = state.elapsed();
    std::cout << "{\"cost\":" << cost << ",\"status\":\"";

    if (state.timed_out)
    {
        std::cout << (cost == -1 ? "TIMEOUT" : "FEASIBLE");
    }
    else
    {
        std::cout << "OPTIMAL";
    }

    std::cout << "\",\"elapsed_ms\":" << elapsed.count() << ",\"routes\":";

    std::vector<std::vector<size_t>> route_customers;
    for (auto &route : routes)
    {
        route_customers.push_back(route.get_customers());
    }

    std::cout << route_customers << "}" << std::endl;

    return 0;
}
