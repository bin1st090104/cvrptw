#include "cli.hpp"

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

struct Route
{
private:
    std::vector<CustomerArrival> _customers;
    size_t _vehicle;
    uint64_t _total_time;
    uint64_t _total_demand;

public:
    explicit Route(size_t vehicle, const cvrptw::Problem &problem) : _vehicle(vehicle), _total_time(0), _total_demand(0)
    {
        _customers.emplace_back(problem.depot, 0);
    }

    std::vector<size_t> get_customers() const
    {
        std::vector<size_t> result;
        for (const auto &c : _customers)
        {
            result.push_back(c.customer);
        }
        return result;
    }

    bool empty() const noexcept
    {
        return _customers.size() == 1;
    }

    uint64_t total_time() const noexcept
    {
        return _total_time;
    }

    bool try_assign(size_t customer, const cvrptw::Problem &problem)
    {
        if (_total_demand + problem.demands[customer] > problem.capacities[_vehicle])
        {
            return false;
        }

        auto last = _customers.back();
        auto arrival_time = last.arrival_time + problem.service_times[last.customer] + problem.time_matrix[last.customer][customer];
        if (arrival_time < problem.time_windows[customer].first)
        {
            arrival_time = problem.time_windows[customer].first;
        }
        else if (arrival_time > problem.time_windows[customer].second)
        {
            return false;
        }

        _total_time = arrival_time + problem.service_times[customer] + problem.time_matrix[customer][problem.depot] - problem.time_matrix[last.customer][problem.depot];
        _total_demand += problem.demands[customer];

        _customers.emplace_back(customer, arrival_time);
        return true;
    }

    void unassign(const cvrptw::Problem &problem)
    {
        auto customer = _customers.back(), last = _customers[_customers.size() - 2];
        _total_time = last.arrival_time + problem.service_times[last.customer] + problem.time_matrix[last.customer][problem.depot];
        _total_demand -= problem.demands[customer.customer];

        _customers.pop_back();
    }
};

struct IterationState
{
private:
    std::vector<Route> _routes;
    std::vector<bool> _assigned;

    std::vector<Route> _solve(const cvrptw::Problem &problem, uint64_t stack, uint64_t &result)
    {
        std::vector<Route> best_routes;
        result = uint64_t(-1);

        if (stack == problem.customers_count())
        {
            result = 0;
            for (const auto &route : _routes)
            {
                best_routes = _routes;
                result = std::max(result, route.total_time());
            }
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
                        if (r < result)
                        {
                            best_routes = routes;
                            result = r;
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
    explicit IterationState(const cvrptw::Problem &problem) : _assigned(problem.customers_count(), false)
    {
        _assigned[problem.depot] = true;

        _routes.reserve(problem.vehicles_count());
        for (size_t v = 0; v < problem.vehicles_count(); v++)
        {
            _routes.emplace_back(v, problem);
        }
    }

    const std::vector<Route> &routes() const
    {
        return _routes;
    }

    std::vector<Route> solve(const cvrptw::Problem &problem, uint64_t &result)
    {
        return _solve(problem, 1, result);
    }
};

int main(int argc, char **argv)
{
    auto arguments = cvrptw::Arguments::parse(argc, argv);
    auto problem = std::move(arguments.problem);
    std::cout << "Loaded " << problem->name << " with "
              << problem->customers_count() << " customers (including depot) and "
              << problem->vehicles_count() << " vehicles" << std::endl;

    IterationState state(*problem);

    uint64_t cost;
    auto result = state.solve(*problem, cost);
    std::cout << "cost = " << cost << std::endl;
    for (size_t v = 0; v < problem->vehicles_count(); v++)
    {
        if (!result[v].empty())
        {
            std::cout << "Vehicle " << v << ": " << result[v].get_customers() << std::endl;
        }
    }

    return 0;
}
