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
            // Do not solve anymore, keep `cost` as the worst possible value (-1)
            return best_routes;
        }

        if (stack == problem.customers_count())
        {
            _first_result = cost = evaluate();
            best_routes = _routes;
        }
        else
        {
            for (size_t customer = 0; customer < problem.customers_count(); customer++)
            {
                if (_assigned[customer])
                {
                    continue;
                }

                for (auto &route : _routes)
                {
                    bool initially_empty = route.empty();
                    if (route.try_assign(customer, problem))
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
    explicit RecursionState(const cvrptw::Problem &problem) : cvrptw::RecursionState(problem), _first_result(-1) {}
};

int main(int argc, char **argv)
{
    auto arguments = cvrptw::Arguments::parse(argc, argv);
    auto problem = std::move(arguments.problem);
    std::cout << "Loaded " << problem->name << " with "
              << problem->customers_count() << " customers (including depot) and "
              << problem->vehicles_count() << " vehicles" << std::endl;

    RecursionState state(*problem);
    uint64_t cost;
    auto routes = state.solve(*problem, cost);

    if (cost == uint64_t(-1))
    {
        throw std::runtime_error("No feasible solution");
    }

    bool improved = true;
    while (improved)
    {
        improved = false;
        for (auto &route : routes)
        {
            for (auto &other_route : routes)
            {
                improved = improved || route.two_opt(&other_route, *problem);
            }
        }
    }

    cost = 0;
    for (const auto &route : routes)
    {
        cost = std::max(cost, route.total_time());
    }

    std::cout << "cost = " << cost << std::endl;
    for (size_t v = 0; v < problem->vehicles_count(); v++)
    {
        if (!routes[v].empty())
        {
            std::cout << "Vehicle " << v << ": " << routes[v].get_customers() << std::endl;
        }
    }

    return 0;
}
