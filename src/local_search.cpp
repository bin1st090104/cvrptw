#include "cli.hpp"
#include "recursion.hpp"

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

int main(int argc, char **argv)
{
    auto arguments = cvrptw::Arguments::parse(argc, argv);
    auto problem = std::move(arguments.problem);
    std::cerr << argv[0] << " loaded " << problem->name << " with "
              << problem->customers_count() << " customers (including depot) and "
              << problem->vehicles_count() << " vehicles" << std::endl;

    std::chrono::milliseconds time_limit = arguments.time_limit.value_or(std::chrono::milliseconds(86400000));
    RecursionState state(time_limit, *problem);

    uint64_t cost;
    auto routes = state.solve(*problem, cost);

    if (cost == uint64_t(-1))
    {
        throw std::runtime_error("No feasible solution");
    }

    bool improved = true;
    while (improved && state.elapsed() < time_limit)
    {
        while (improved && state.elapsed() < time_limit)
        {
            for (auto &route : routes)
            {
                for (auto &other_route : routes)
                {
                    improved = improved || route.move_10(&other_route, *problem);
                }
            }
        }

        for (auto &route : routes)
        {
            for (auto &other_route : routes)
            {
                improved = improved || route.two_opt(&other_route, *problem);
            }
        }
    }

    auto elapsed = state.elapsed();
    state.timed_out = state.elapsed() >= time_limit;

    cost = 0;
    for (const auto &route : routes)
    {
        cost += route.total_cost();
    }

    std::cout << "{\"cost\":" << cost << ",\"status\":\"";

    if (cost == -1)
    {
        std::cout << "TIMEOUT";
    }
    else
    {
        std::cout << "FEASIBLE";
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
