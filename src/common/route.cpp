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

    bool Route::try_assign(size_t customer, const cvrptw::Problem &problem, uint64_t &cost_change)
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

        cost_change = problem.time_matrix[last.customer][customer] + problem.time_matrix[customer][problem.depot] - problem.time_matrix[last.customer][problem.depot];
        auto total_cost = _total_cost + cost_change;

        _total_cost = total_cost;
        _total_demand += problem.demands[customer];

        _customers.emplace_back(customer, arrival_time);
        return true;
    }

    void Route::unassign(const cvrptw::Problem &problem)
    {
        auto customer = _customers.back(), last = _customers[_customers.size() - 2];
        _total_cost -= problem.time_matrix[last.customer][customer.customer] + problem.time_matrix[customer.customer][problem.depot] - problem.time_matrix[last.customer][problem.depot];
        _total_demand -= problem.demands[customer.customer];

        _customers.pop_back();
    }

    bool Route::move_10(Route *other, const cvrptw::Problem &problem)
    {
        if (this == other)
        {
            auto n = _customers.size();
            for (size_t i = 1; i + 1 < n; i++)
            {
                auto base_gain = problem.time_matrix[_customers[i - 1].customer][_customers[i + 1].customer];
                auto base_loss = problem.time_matrix[_customers[i - 1].customer][_customers[i].customer] +
                                 problem.time_matrix[_customers[i].customer][_customers[i + 1].customer];

                for (size_t j = i + 2; j < n + 1; j++)
                {
                    auto gain = base_gain + problem.time_matrix[_customers[j - 1].customer][_customers[i].customer] +
                                (j < n ? problem.time_matrix[_customers[i].customer][_customers[j].customer] : problem.time_matrix[_customers[i].customer][problem.depot]);
                    auto loss = base_loss + (j < n ? problem.time_matrix[_customers[j - 1].customer][_customers[j].customer] : problem.time_matrix[_customers[j - 1].customer][problem.depot]);
                    if (gain >= loss)
                    {
                        continue;
                    }

                    std::vector<CustomerArrival> new_customers(_customers.begin(), _customers.end());
                    std::rotate(new_customers.begin() + i, new_customers.begin() + i + 1, new_customers.begin() + j);

                    if (!_recalculate_arrival_times(new_customers, problem, i))
                    {
                        continue;
                    }

                    _customers = std::move(new_customers);
                    _total_cost = _total_cost + gain - loss;
                    return true;
                }
            }
        }
        else
        {
            auto n = _customers.size();
            auto m = other->_customers.size();
            for (size_t i = 1; i < n; i++)
            {
                auto this_gain = (i + 1 < n ? problem.time_matrix[_customers[i - 1].customer][_customers[i + 1].customer] : problem.time_matrix[_customers[i - 1].customer][problem.depot]);
                auto this_loss = problem.time_matrix[_customers[i - 1].customer][_customers[i].customer] +
                                 (i + 1 < n ? problem.time_matrix[_customers[i].customer][_customers[i + 1].customer] : problem.time_matrix[_customers[i].customer][problem.depot]);

                for (size_t j = 1; j < m + 1; j++)
                {
                    auto new_this_demand = _total_demand - problem.demands[_customers[i].customer];
                    auto new_other_demand = other->_total_demand + problem.demands[_customers[i].customer];
                    if (new_other_demand > problem.capacities[other->_vehicle])
                    {
                        continue;
                    }

                    auto other_gain = problem.time_matrix[other->_customers[j - 1].customer][_customers[i].customer] +
                                      (j < m ? problem.time_matrix[_customers[i].customer][other->_customers[j].customer] : problem.time_matrix[_customers[i].customer][problem.depot]);
                    auto other_loss = (j < m ? problem.time_matrix[other->_customers[j - 1].customer][other->_customers[j].customer] : problem.time_matrix[other->_customers[j - 1].customer][problem.depot]);
                    if (this_gain + other_gain >= this_loss + other_loss)
                    {
                        continue;
                    }

                    std::vector<CustomerArrival> new_customers_i(_customers.begin(), _customers.begin() + i);
                    std::vector<CustomerArrival> new_customers_j(other->_customers.begin(), other->_customers.begin() + j);

                    new_customers_i.insert(new_customers_i.end(), _customers.begin() + (i + 1), _customers.end());
                    new_customers_j.push_back(_customers[i]);
                    new_customers_j.insert(new_customers_j.end(), other->_customers.begin() + j, other->_customers.end());

                    if (!_recalculate_arrival_times(new_customers_i, problem, i) || !_recalculate_arrival_times(new_customers_j, problem, j))
                    {
                        continue;
                    }

                    _customers = std::move(new_customers_i);
                    other->_customers = std::move(new_customers_j);

                    _total_cost = _total_cost + this_gain - this_loss;
                    other->_total_cost = other->_total_cost + other_gain - other_loss;

                    _total_demand = new_this_demand;
                    other->_total_demand = new_other_demand;

                    return true;
                }
            }
        }

        return false;
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
                    auto gain = problem.time_matrix[_customers[i - 1].customer][_customers[j - 1].customer] +
                                (j < n ? problem.time_matrix[_customers[i].customer][_customers[j].customer] : problem.time_matrix[_customers[i].customer][problem.depot]);
                    auto loss = problem.time_matrix[_customers[i - 1].customer][_customers[i].customer] +
                                (j < n ? problem.time_matrix[_customers[j - 1].customer][_customers[j].customer] : problem.time_matrix[_customers[j - 1].customer][problem.depot]);

                    if (gain >= loss)
                    {
                        continue;
                    }

                    std::vector<CustomerArrival> new_customers(_customers.begin(), _customers.begin() + i);
                    new_customers.insert(new_customers.end(), _customers.rbegin() + (n - j), _customers.rbegin() + (n - i));
                    new_customers.insert(new_customers.end(), _customers.begin() + j, _customers.end());

                    if (!_recalculate_arrival_times(new_customers, problem, i))
                    {
                        continue;
                    }

                    _customers = std::move(new_customers);
                    _total_cost = _total_cost + gain - loss;
                    return true;
                }
            }
        }
        else
        {
            // Swap [i..) and [j..) tails
            auto n = _customers.size();
            auto m = other->_customers.size();

            uint64_t total_this_demand = 0;
            for (size_t i = 1; i < n; i++)
            {
                total_this_demand += problem.demands[_customers[i].customer];
            }

            uint64_t total_other_demand = 0;
            for (size_t j = 1; j < m; j++)
            {
                total_other_demand += problem.demands[other->_customers[j].customer];
            }

            for (size_t i = 1; i < n + 1; i++)
            {
                auto this_demand = total_this_demand;
                if (i < n)
                {
                    total_this_demand -= problem.demands[_customers[i].customer];
                }

                auto other_demand = total_other_demand;
                for (size_t j = 1; j < m + 1; j++)
                {
                    // Check capacity constraints after swap
                    uint64_t new_this_demand = _total_demand + other_demand - this_demand;
                    uint64_t new_other_demand = other->_total_demand + this_demand - other_demand;
                    if (j < m)
                    {
                        other_demand -= problem.demands[other->_customers[j].customer];
                    }

                    if (new_this_demand > problem.capacities[_vehicle] || new_other_demand > problem.capacities[other->_vehicle])
                    {
                        continue;
                    }

                    auto i_gain = j < m ? problem.time_matrix[_customers[i - 1].customer][other->_customers[j].customer] : problem.time_matrix[_customers[i - 1].customer][problem.depot];
                    auto j_gain = i < n ? problem.time_matrix[other->_customers[j - 1].customer][_customers[i].customer] : problem.time_matrix[other->_customers[j - 1].customer][problem.depot];
                    auto i_loss = i < n ? problem.time_matrix[_customers[i - 1].customer][_customers[i].customer] : problem.time_matrix[_customers[i - 1].customer][problem.depot];
                    auto j_loss = j < m ? problem.time_matrix[other->_customers[j - 1].customer][other->_customers[j].customer] : problem.time_matrix[other->_customers[j - 1].customer][problem.depot];
                    if (i_gain + j_gain >= i_loss + j_loss)
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

                    _customers = std::move(new_customers_i);
                    other->_customers = std::move(new_customers_j);

                    _total_cost = _total_cost + i_gain - i_loss;
                    other->_total_cost = other->_total_cost + j_gain - j_loss;

                    _total_demand = new_this_demand;
                    other->_total_demand = new_other_demand;

                    return true;
                }
            }
        }

        return false;
    }
}
