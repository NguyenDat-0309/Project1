import math
import copy
from typing import List, Tuple
import os

class Customer:
    def __init__(self, cid: int, x: float, y: float, demand: float,
                 tw_open: float, tw_close: float, service_time: float):
        self.id = cid
        self.x = x
        self.y = y
        self.demand = demand
        self.tw_open = tw_open
        self.tw_close = tw_close
        self.service_time = service_time

    def __repr__(self):
        return f"C{self.id}"

class Route:
    def __init__(self, depot_x, depot_y, capacity):
        self.depot_x = depot_x
        self.depot_y = depot_y
        self.capacity_limit = capacity
        self.customers: List[Customer] = []
        self.current_load = 0.0
        self.total_distance = 0.0

    def __repr__(self):
        return "->".join(str(c) for c in self.customers)

def read_solomon_data(file_path: str) -> Tuple[Customer, List[Customer], int, float]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n\r') for line in f]

    num_vehicles = 0
    vehicle_capacity = 0.0
    customers: List[Customer] = []

    i = 0
    while i < len(lines):
        line_upper = lines[i].strip().upper()
        if "VEHICLE" in line_upper:
            i += 2
            if i < len(lines):
                parts = lines[i].split()
                num_vehicles = int(parts[0])
                vehicle_capacity = float(parts[1])
        elif "CUSTOMER" in line_upper:
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip():
                parts = lines[i].split()
                cust_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_time = float(parts[5])
                service_time = float(parts[6])
                c = Customer(cust_id, x, y, demand, ready_time, due_time, service_time)
                customers.append(c)
                i += 1
        i += 1

    depot = None
    for c in customers:
        if c.id == 0:
            depot = c
            break
    if depot is None:
        raise ValueError("Depot with id=0 not found in the data.")

    customers = [c for c in customers if c.id != 0]

    return depot, customers, num_vehicles, vehicle_capacity

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def travel_time(distance, speed=1.0):
    return distance / speed

def compute_route_distance(route: Route) -> float:
    dist = 0.0
    px, py = route.depot_x, route.depot_y
    for c in route.customers:
        dist += euclidean_distance(px, py, c.x, c.y)
        px, py = c.x, c.y
    dist += euclidean_distance(px, py, route.depot_x, route.depot_y)
    return dist

def compute_arrival_times(route: Route, insert_cust=None, insert_pos=None):
    temp_custs = route.customers[:]
    if insert_cust is not None and insert_pos is not None:
        temp_custs.insert(insert_pos, insert_cust)

    time_now = 0.0
    px, py = route.depot_x, route.depot_y
    arrivals = []

    for c in temp_custs:
        dist = euclidean_distance(px, py, c.x, c.y)
        time_now += travel_time(dist)
        if time_now < c.tw_open:
            time_now = c.tw_open
        if time_now > c.tw_close:
            return None
        arrivals.append(time_now)
        time_now += c.service_time
        px, py = c.x, c.y

    return arrivals

def compute_insertion_metrics(route: Route, cust: Customer, insert_pos: int, alpha=0.5):
    temp_r = copy.deepcopy(route)
    temp_r.customers.insert(insert_pos, cust)
    new_dist = compute_route_distance(temp_r)
    old_dist = route.total_distance
    distance_increase = new_dist - old_dist

    old_arrivals = compute_arrival_times(route)
    new_arrivals = compute_arrival_times(route, insert_cust=cust, insert_pos=insert_pos)
    if old_arrivals is None or new_arrivals is None:
        return (False, None)

    delay_increase = sum(
        max(0, new_arrivals[i] - old_arrivals[i])
        for i in range(min(len(old_arrivals), len(new_arrivals)))
    )

    if route.current_load + cust.demand > route.capacity_limit:
        return (False, None)

    cost_val = alpha * distance_increase + (1 - alpha) * delay_increase
    return (True, cost_val)

def compute_generalized_regret(customers: List[Customer], routes: List[Route], alpha=0.5):
    regret_values = []
    for cust in customers:
        insertion_costs = []
        for route in routes:
            for pos in range(len(route.customers) + 1):
                feasible, cost_val = compute_insertion_metrics(route, cust, pos, alpha)
                if feasible:
                    insertion_costs.append(cost_val)

        if len(insertion_costs) == 0:
            regret = float('inf')
        else:
            regret = sum(sorted(insertion_costs)[1:]) - min(insertion_costs)
        regret_values.append((cust, regret))

    return regret_values

def adjust_routes(seeds, customers, depot_x, depot_y, increase=False):
    if increase:
        unrouted_customers = [c for c in customers if c not in seeds]
        if unrouted_customers:
            farthest_customer = max(unrouted_customers, key=lambda c: euclidean_distance(seeds[-1].x, seeds[-1].y, c.x, c.y))
            seeds.append(farthest_customer)
    else:
        if len(seeds) > 1:
            closest_pair = min(
                [(s1, s2) for i, s1 in enumerate(seeds) for s2 in seeds[i + 1:]],
                key=lambda pair: euclidean_distance(pair[0].x, pair[0].y, pair[1].x, pair[1].y)
            )
            seeds.remove(closest_pair[0])

    return seeds


def solomon_sequential(customers: List[Customer], capacity: float, depot_x: float, depot_y: float):
    routes = []
    unassigned = customers[:]

    while unassigned:
        route = Route(depot_x, depot_y, capacity)
        seed = max(unassigned, key=lambda c: euclidean_distance(depot_x, depot_y, c.x, c.y))
        route.customers.append(seed)
        route.current_load += seed.demand
        route.total_distance = compute_route_distance(route)
        unassigned.remove(seed)

        while True:
            best_cust, best_pos, best_cost = None, None, float('inf')
            for cust in unassigned:
                for pos in range(len(route.customers) + 1):
                    feasible, cost_val = compute_insertion_metrics(route, cust, pos)
                    if feasible and cost_val < best_cost:
                        best_cust, best_pos, best_cost = cust, pos, cost_val

            if best_cust is None:
                break

            route.customers.insert(best_pos, best_cust)
            route.current_load += best_cust.demand
            route.total_distance = compute_route_distance(route)
            unassigned.remove(best_cust)

        routes.append(route)

    return routes

def parallel_route_building(customers, capacity, depot_x, depot_y, alpha=0.5):
    sequential_routes = solomon_sequential(customers, capacity, depot_x, depot_y)
    num_routes = len(sequential_routes)
    seeds = [route.customers[0] for route in sequential_routes]

    routes = [Route(depot_x, depot_y, capacity) for _ in range(len(seeds))]
    for route, seed in zip(routes, seeds):
        route.customers.append(seed)
        route.current_load += seed.demand
        route.total_distance = compute_route_distance(route)

    unassigned = [c for c in customers if c not in seeds]

    while unassigned:
        regret_values = compute_generalized_regret(unassigned, routes, alpha)
        best_cust, _ = max(regret_values, key=lambda x: x[1])

        best_route, best_pos, best_cost = None, None, float('inf')
        for route in routes:
            for pos in range(len(route.customers) + 1):
                feasible, cost_val = compute_insertion_metrics(route, best_cust, pos, alpha)
                if feasible and cost_val < best_cost:
                    best_cost = cost_val
                    best_route = route
                    best_pos = pos

        if best_route:
            best_route.customers.insert(best_pos, best_cust)
            best_route.current_load += best_cust.demand
            best_route.total_distance = compute_route_distance(best_route)
            unassigned.remove(best_cust)
        else:
            seeds = adjust_routes(seeds, customers, depot_x, depot_y, increase=True)
            new_route = Route(depot_x, depot_y, capacity)
            new_seed = seeds[-1]
            new_route.customers.append(new_seed)
            new_route.current_load += new_seed.demand
            new_route.total_distance = compute_route_distance(new_route)
            routes.append(new_route)

            if new_seed in unassigned:
                unassigned.remove(new_seed)

    return routes


if __name__ == "__main__":
    FILE_PATH = "C:/Users/Admin/vs code/Project 1/Solomon set/C103.txt"
    depot, customers, num_vehicles, vehicle_capacity = read_solomon_data(FILE_PATH)
    depot_x, depot_y = depot.x, depot.y

    ALPHAS = [0.5, 0.75, 1.0]
    best_alpha = None
    best_routes_result = None
    best_total_cost = float('inf')

    for alpha in ALPHAS:
        routes_result = parallel_route_building(
            customers=customers,
            capacity=vehicle_capacity,
            depot_x=depot_x,
            depot_y=depot_y,
            alpha=alpha
        )

        total_cost = sum(r.total_distance for r in routes_result)

        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_alpha = alpha
            best_routes_result = routes_result

    dataset_name = os.path.basename(FILE_PATH)
    print(f"Dataset: {dataset_name}\n")
    print(f"Best alpha: {best_alpha:.2f}")

    for i, r in enumerate(best_routes_result, start=1):
        route_list = [0] + [c.id for c in r.customers] + [0]
        route_str = ", ".join(str(x) for x in route_list)
        cost = r.total_distance
        print(f"Vehicle {i}: Route = [{route_str}], Cost = {cost:.2f}")

    print(f"\nTotal cost of all routes for best alpha: {best_total_cost:.2f}")
