#include "route.hpp"

namespace cvrptw
{
    bool Route::_recalculate_arrival_times(std::vector<CustomerArrival> &customers, const cvrptw::Problem &problem, size_t offset)
    {
        auto n = customers.size();
        for (size_t i = std::max(offset, size_t(1)); i < n; i++)
        {
            auto &last = customers[i - 1];
            auto &current = customers[i];
            current.arrival_time = last.arrival_time + problem.service_times[last.customer] + problem.time_matrix[last.customer][current.customer];
            if (current.arrival_time < problem.time_windows[current.customer].first)
            {
                current.arrival_time = problem.time_windows[current.customer].first;
            }
            else if (current.arrival_time > problem.time_windows[current.customer].second)
            {
                return false;
            }
        }

        return true;
    }

    std::vector<size_t> Route::get_customers() const
    {
        std::vector<size_t> result;
        for (const auto &c : _customers)
        {
            result.push_back(c.customer);
        }
        return result;
    }

    bool Route::try_assign(size_t customer, const cvrptw::Problem &problem, uint64_t best)
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

        auto total_time = arrival_time + problem.service_times[customer] + problem.time_matrix[customer][problem.depot];
        if (best > 0 && total_time >= best)
        {
            return false;
        }

        _total_time = total_time;
        _total_demand += problem.demands[customer];

        _customers.emplace_back(customer, arrival_time);
        return true;
    }

    void Route::unassign(const cvrptw::Problem &problem)
    {
        auto customer = _customers.back(), last = _customers[_customers.size() - 2];
        _total_time = last.arrival_time + problem.service_times[last.customer] + problem.time_matrix[last.customer][problem.depot];
        _total_demand -= problem.demands[customer.customer];

        _customers.pop_back();
    }

    bool Route::two_opt(Route *other, const cvrptw::Problem &problem)
    {
        if (this == other)
        {
            // Reverse [i, j)
            auto n = _customers.size();
            for (size_t i = 1; i < n; i++)
            {
                for (size_t j = i + 2; j < n + 1; j++)
                {
                    std::vector<CustomerArrival> new_customers(_customers.begin(), _customers.begin() + i);
                    new_customers.insert(new_customers.end(), _customers.rbegin() + (n - j), _customers.rbegin() + (n - i));
                    new_customers.insert(new_customers.end(), _customers.begin() + j, _customers.end());

                    if (!_recalculate_arrival_times(new_customers, problem, i))
                    {
                        continue;
                    }

                    auto &last = new_customers.back();
                    auto new_total_time = last.arrival_time + problem.service_times[last.customer] + problem.time_matrix[last.customer][problem.depot];
                    if (new_total_time < _total_time)
                    {
                        _customers = std::move(new_customers);
                        _total_time = new_total_time;
                        return true;
                    }
                }
            }
        }
        else
        {
            // Swap [i..) and [j..) tails
            auto n = _customers.size();
            auto m = other->_customers.size();
            auto best = std::max(_total_time, other->_total_time);
            for (size_t i = 1; i < n + 1; i++)
            {
                for (size_t j = 1; j < m + 1; j++)
                {
                    if (i == 1 && j == 1)
                    {
                        continue;
                    }

                    std::vector<CustomerArrival> new_customers_i(_customers.begin(), _customers.begin() + i);
                    std::vector<CustomerArrival> new_customers_j(other->_customers.begin(), other->_customers.begin() + j);
                    new_customers_i.insert(new_customers_i.end(), other->_customers.begin() + j, other->_customers.end());
                    new_customers_j.insert(new_customers_j.end(), _customers.begin() + i, _customers.end());

                    if (!_recalculate_arrival_times(new_customers_i, problem, i) || !_recalculate_arrival_times(new_customers_j, problem, j))
                    {
                        continue;
                    }

                    auto &last_i = new_customers_i.back();
                    auto &last_j = new_customers_j.back();
                    auto new_total_time_i = last_i.arrival_time + problem.service_times[last_i.customer] + problem.time_matrix[last_i.customer][problem.depot];
                    auto new_total_time_j = last_j.arrival_time + problem.service_times[last_j.customer] + problem.time_matrix[last_j.customer][problem.depot];
                    if (new_total_time_i < best && new_total_time_j < best)
                    {
                        _customers = std::move(new_customers_i);
                        other->_customers = std::move(new_customers_j);

                        _total_time = new_total_time_i;
                        other->_total_time = new_total_time_j;

                        return true;
                    }
                }
            }
        }

        return false;
    }
}
