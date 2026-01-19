#include "problem.hpp"
#include "utils.hpp"

namespace cvrptw
{
    std::unique_ptr<Problem> Problem::from_file(const std::filesystem::path &path, size_t limit)
    {
        std::ifstream input(path);
        if (!input.is_open())
        {
            throw std::runtime_error(std::format("Unable to open {}", path.string()));
        }

        std::string name, ignored;
        input >> name >> ignored >> ignored >> ignored;

        size_t vehicles_count;
        uint64_t vehicle_capacity;
        input >> vehicles_count >> vehicle_capacity;

        input >> ignored;
        input >> ignored >> ignored >> ignored >> ignored >> ignored >> ignored >> ignored >> ignored >> ignored >> ignored >> ignored;

        std::vector<int64_t> xs, ys;
        std::vector<uint64_t> demands, service_times;
        std::vector<std::pair<uint64_t, uint64_t>> time_windows;
        while (input >> ignored && xs.size() < limit)
        {
            int64_t x, y;
            uint64_t demand, ready_time, due_date, service_time;
            input >> x >> y >> demand >> ready_time >> due_date >> service_time;
            xs.push_back(x);
            ys.push_back(y);
            demands.push_back(demand);
            time_windows.emplace_back(ready_time, due_date);
            service_times.push_back(service_time);
        }

        auto distance = [&](size_t i, size_t j)
        {
            uint64_t dx = xs[i] > xs[j] ? xs[i] - xs[j] : xs[j] - xs[i];
            uint64_t dy = ys[i] > ys[j] ? ys[i] - ys[j] : ys[j] - ys[i];
            return dx + dy;
        };

        std::vector<size_t> reordered(xs.size());
        std::iota(reordered.begin(), reordered.end(), 0);
        std::sort(
            reordered.begin(), reordered.end(),
            [&](size_t a, size_t b)
            {
                return distance(0, a) < distance(0, b);
            });

        std::vector<uint64_t> capacities(vehicles_count, vehicle_capacity);
        std::vector<std::vector<uint64_t>> time_matrix(xs.size(), std::vector<uint64_t>(xs.size()));
        for (size_t i = 0; i < xs.size(); i++)
        {
            for (size_t j = 0; j < xs.size(); j++)
            {
                time_matrix[i][j] = distance(i, j);
            }
        }

        return std::make_unique<Problem>(
            std::move(name),
            std::move(capacities),
            std::move(time_matrix),
            std::move(demands),
            std::move(time_windows),
            std::move(service_times),
            0);
    }
}
