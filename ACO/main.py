import math
import random
import copy
import sys
import time
import os

class Node:
    def __init__(self, id, x, y, demand, ready_time, due_time, service_time):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        self.demand = float(demand)
        self.ready_time = float(ready_time)
        self.due_time = float(due_time)
        self.service_time = float(service_time)

def read_file(file_path):
    nodes = []
    vehicle_capacity = 0
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "CAPACITY" in line:
                data_line = lines[i+1].strip().split()
                vehicle_capacity = float(data_line[-1])
                break
        start_parsing = False
        for line in lines:
            if "CUST NO." in line:
                start_parsing = True
                continue
            if start_parsing:
                parts = line.strip().split()
                if len(parts) >= 7:
                    nodes.append(Node(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]))
        return nodes, vehicle_capacity
    except Exception:
        return [], 0


# LOCAL SEARCH (INTRA & INTER ROUTE)
class LocalSearch:
    @staticmethod
    def calculate_cost(route, dist_matrix):
        return sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))

    @staticmethod
    def is_valid(route, nodes, capacity, dist_matrix):
        """Kiểm tra tính hợp lệ của 1 lộ trình (Capacity & Time)"""
        load = 0
        time = 0
        curr = 0 # Depot
        
        for i in range(1, len(route)):
            next_node = route[i]
            node_obj = nodes[next_node]
            
            # Capacity
            load += node_obj.demand
            if load > capacity: return False
            
            # Time
            arrival = time + nodes[curr].service_time + dist_matrix[curr][next_node]
            start = max(arrival, node_obj.ready_time)
            if start > node_obj.due_time: return False
            
            time = start
            curr = next_node
        return True

    @staticmethod
    def run_2opt(route, nodes, capacity, dist_matrix):
        best_route = route[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    # Chỉ check nếu cost giảm
                    if LocalSearch.calculate_cost(new_route, dist_matrix) < LocalSearch.calculate_cost(best_route, dist_matrix):
                        if LocalSearch.is_valid(new_route, nodes, capacity, dist_matrix):
                            best_route = new_route
                            improved = True
                            break
                if improved: break
        return best_route

    @staticmethod
    def run_relocate(routes, nodes, capacity, dist_matrix):
        improved = True
        best_routes = copy.deepcopy(routes)
        
        while improved:
            improved = False
            # Duyệt qua từng cặp tuyến đường
            for r1_idx in range(len(best_routes)):
                for r2_idx in range(len(best_routes)):
                    if r1_idx == r2_idx: continue                 
                    route1 = best_routes[r1_idx]
                    route2 = best_routes[r2_idx]
                    
                    # Thử lấy khách i từ route1 chèn vào route2
                    # Duyệt ngược để dễ xóa
                    for i in range(1, len(route1) - 1): 
                        customer = route1[i]
                        
                        # Thử chèn vào mọi vị trí j trong route2
                        best_insert_pos = -1
                        current_total_cost = LocalSearch.calculate_cost(route1, dist_matrix) + \
                                             LocalSearch.calculate_cost(route2, dist_matrix)
                        best_saving = 0
                        
                        # Tạo route1 tạm thời (đã bỏ customer)
                        temp_r1 = route1[:i] + route1[i+1:]
                        if not LocalSearch.is_valid(temp_r1, nodes, capacity, dist_matrix):
                            continue
                        
                        cost_r1_new = LocalSearch.calculate_cost(temp_r1, dist_matrix)
                        
                        for j in range(1, len(route2)):
                            # Tạo route2 tạm thời (chèn customer vào vị trí j)
                            temp_r2 = route2[:j] + [customer] + route2[j:]
                            
                            # Check nhanh cost trước
                            cost_r2_new = LocalSearch.calculate_cost(temp_r2, dist_matrix)
                            new_total_cost = cost_r1_new + cost_r2_new                         
                            if new_total_cost < current_total_cost:
                                if LocalSearch.is_valid(temp_r2, nodes, capacity, dist_matrix):
                                    saving = current_total_cost - new_total_cost
                                    if saving > best_saving:
                                        best_saving = saving
                                        best_insert_pos = j
                        
                        # Nếu tìm được vị trí chèn tốt
                        if best_insert_pos != -1:
                            # Thực hiện thay đổi
                            best_routes[r1_idx] = temp_r1
                            best_routes[r2_idx] = best_routes[r2_idx][:best_insert_pos] + [customer] + best_routes[r2_idx][best_insert_pos:]                      
                            # Xóa route rỗng nếu có
                            if len(best_routes[r1_idx]) <= 2: 
                                del best_routes[r1_idx]                              
                            improved = True
                            break # Restart loop để cập nhật lại cấu trúc routes
                    if improved: break
                if improved: break               
        return best_routes

# ACO
class ACO_CVRPTW:
    def __init__(self, nodes, vehicle_capacity, num_ants=40, max_iter=100, alpha=1.0, beta=4.0, rho=0.1):
        self.nodes = nodes
        self.capacity = vehicle_capacity
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.num_nodes = len(nodes)
        self.dist_matrix = self._calculate_manhattan_matrix()
        self.pheromone = [[1.0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]
        
        # Candidate List: Chỉ lưu 20 node gần nhất cho mỗi node
        self.candidate_list = self._build_candidate_list(k=20)
        
        self.best_global_solution = None
        self.best_global_cost = float('inf')

    def _calculate_manhattan_matrix(self):
        matrix = [[0.0] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                matrix[i][j] = abs(self.nodes[i].x - self.nodes[j].x) + abs(self.nodes[i].y - self.nodes[j].y)
        return matrix

    def _build_candidate_list(self, k):
        candidates = []
        for i in range(self.num_nodes):
            # Lấy danh sách các node khác i, sắp xếp theo khoảng cách tăng dần
            dists = [(j, self.dist_matrix[i][j]) for j in range(self.num_nodes) if j != i and j != 0]
            dists.sort(key=lambda x: x[1])
            # Lấy k node gần nhất
            top_k = [x[0] for x in dists[:k]]
            candidates.append(top_k)
        return candidates

    def _check_constraints(self, curr, next_n, load, time):
        node = self.nodes[next_n]
        if load + node.demand > self.capacity: return False, 0, 0
        arr = time + self.nodes[curr].service_time + self.dist_matrix[curr][next_n]
        start = max(arr, node.ready_time)
        if start > node.due_time: return False, 0, 0
        return True, load + node.demand, start

    def solve(self, time_limit=None):
        start_time_process = time.time()
    
        initial_beta = self.beta
        
        for iteration in range(self.max_iter):
            if time_limit is not None:
                elapsed = time.time() - start_time_process
                if elapsed > time_limit:
                    print(f"time out")
                    break

            iter_solutions = []
            current_beta = initial_beta + (iteration / self.max_iter)
            
            for k in range(self.num_ants):
                unvisited = set(range(1, self.num_nodes))
                routes = []
                
                while unvisited:
                    route = [0]
                    curr, load, t_curr = 0, 0, 0
                    
                    while True:
                        candidates_pool = [n for n in self.candidate_list[curr] if n in unvisited]
                        if not candidates_pool: candidates_pool = list(unvisited)
                        
                        probs, next_nodes = [], []
                        denom = 0.0
                        
                        for next_n in candidates_pool:
                            is_ok, _, _ = self._check_constraints(curr, next_n, load, t_curr)
                            if is_ok:
                                tau = self.pheromone[curr][next_n]
                                dist = self.dist_matrix[curr][next_n]
                                eta = 1.0 / (dist + 0.0001)
                                val = (tau ** self.alpha) * (eta ** current_beta)
                                probs.append(val)
                                next_nodes.append(next_n)
                                denom += val
                        
                        if not next_nodes:
                            route.append(0)
                            break
                        
                        if random.random() < 0.9: 
                            if denom == 0: selected = random.choice(next_nodes)
                            else:
                                r = random.uniform(0, denom)
                                cum = 0
                                selected = next_nodes[-1]
                                for i, p in enumerate(probs):
                                    cum += p
                                    if r <= cum: selected = next_nodes[i]; break
                        else:
                            max_idx = probs.index(max(probs))
                            selected = next_nodes[max_idx]
                        
                        _, load, t_curr = self._check_constraints(curr, selected, load, t_curr)
                        route.append(selected)
                        unvisited.remove(selected)
                        curr = selected
                        if not unvisited: route.append(0); break
                    
                    optimized_route = LocalSearch.run_2opt(route, self.nodes, self.capacity, self.dist_matrix)
                    routes.append(optimized_route)
                
                total_dist = sum(LocalSearch.calculate_cost(r, self.dist_matrix) for r in routes)
                iter_solutions.append((routes, total_dist))

            best_iter_sol = min(iter_solutions, key=lambda x: x[1])
            best_iter_routes, best_iter_cost = best_iter_sol
            
            final_routes = LocalSearch.run_relocate(best_iter_routes, self.nodes, self.capacity, self.dist_matrix)
            final_cost = sum(LocalSearch.calculate_cost(r, self.dist_matrix) for r in final_routes)
            
            if final_cost < self.best_global_cost:
                self.best_global_cost = final_cost
                self.best_global_solution = copy.deepcopy(final_routes)
                # In tiến độ
                print(f"Iter {iteration+1:03d} | New Best: {self.best_global_cost:.2f} | Time: {time.time() - start_time_process:.2f}s")

            self._update_pheromone(final_routes, final_cost)

        return self.best_global_solution, self.best_global_cost
    def _update_pheromone(self, best_routes, best_cost):
        # Bay hơi
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromone[i][j] *= (1.0 - self.rho)
                if self.pheromone[i][j] < 0.01: self.pheromone[i][j] = 0.01 # Min pheromone
                if self.pheromone[i][j] > 10.0: self.pheromone[i][j] = 10.0 # Max pheromone
        
        # Deposit
        delta = 100.0 / best_cost
        for route in best_routes:
            for k in range(len(route)-1):
                i, j = route[k], route[k+1]
                self.pheromone[i][j] += delta

if __name__ == "__main__":
    output_result_file = "aco_results-100-100.txt"
    instance_list = ["testcases\C101.txt","testcases\C102.txt","testcases\C103.txt","testcases\C104.txt","testcases\C105.txt","testcases\C106.txt","testcases\C107.txt","testcases\C108.txt","testcases\C109.txt"] 

    # cấu hình
    NUM_CUSTOMERS = 100       
    MAX_ITERATIONS = 1000    
    MAX_TIME_SECONDS = 100 

    header_str = f"{'Instance':<10} | {'Time(s)':<10} | {'Obj Val':<10} | {'Valid':<5} | {'Routes'}\n"
    if not os.path.exists(output_result_file):
        with open(output_result_file, "w", encoding="utf-8") as f:
            f.write(header_str + "-" * 120 + "\n")
    
    for input_file in instance_list:
        print(f"\n--- Đang chạy {input_file} (Limit: {MAX_TIME_SECONDS}s hoặc {MAX_ITERATIONS} iters) ---")
        
        try:
            all_nodes, capacity = read_file(input_file)
            if not all_nodes: continue
            
            nodes = all_nodes[:1 + NUM_CUSTOMERS]
            
            aco = ACO_CVRPTW(
                nodes, capacity, 
                num_ants=30,     
                max_iter=MAX_ITERATIONS,
                alpha=1.0, beta=3.0, rho=0.1
            )
            
            start_clock = time.time()    
            best_routes, best_cost = aco.solve(time_limit=MAX_TIME_SECONDS)
            end_clock = time.time()
            actual_time = end_clock - start_clock
            
            is_valid_total = True
            for r in best_routes:
                if not LocalSearch.is_valid(r, nodes, capacity, aco.dist_matrix):
                    is_valid_total = False; break
            res_line = f"{input_file:<10}| {actual_time:<10.4f} | {best_cost:<10.2f} | {('OK' if is_valid_total else 'NO'):<5} | {str(best_routes)}\n"
            
            with open(output_result_file, "a", encoding="utf-8") as f:
                f.write(res_line)
                
            print(f"Done. Time: {actual_time:.2f}s. Cost: {best_cost:.2f}")

        except Exception as e:
            print(f"Error {input_file}: {e}")