#!/usr/bin/env python3
"""
main.py

Genetic Algorithm solution for a Transportation Problem.

Expected optional CSVs (simple formats described below). If they are not found,
the script generates a random instance.

- suppliers.csv:
    id,capacity
    S0,100
    S1,80
    ...

- consumers.csv:
    id,demand
    C0,60
    C1,90
    ...

- costs.csv:
    (matrix rows = suppliers, cols = consumers, header optional)
    4,8,5
    6,7,3
    ...

Output:
- transport_output.csv: each row supplier,consumer,amount
- prints summary to stdout
"""

import csv
import os
import random
import math
from typing import List, Tuple, Dict, Optional

# -----------------------------
# Data structures
# -----------------------------
class Supplier:
    def __init__(self, id: str, capacity: int):
        self.id = id
        self.capacity = int(capacity)

class Consumer:
    def __init__(self, id: str, demand: int):
        self.id = id
        self.demand = int(demand)

class TransportSolution:
    """
    Represents a shipment matrix: allocations[i][j] = amount shipped from supplier i to consumer j.
    """
    def __init__(self, allocations: List[List[int]], suppliers: List[Supplier], consumers: List[Consumer]):
        self.allocations = allocations  # m x n int matrix
        self.suppliers = suppliers
        self.consumers = consumers
        self._cached_cost = None
        self._cached_penalty = None

    def copy(self):
        return TransportSolution([row[:] for row in self.allocations], self.suppliers, self.consumers)

    def total_cost(self, cost_matrix: List[List[float]]) -> float:
        cost = 0.0
        for i, row in enumerate(self.allocations):
            for j, val in enumerate(row):
                cost += val * cost_matrix[i][j]
        return cost

    def total_supply_used(self) -> List[int]:
        return [sum(row) for row in self.allocations]

    def total_demand_received(self) -> List[int]:
        n = len(self.consumers)
        totals = [0] * n
        for row in self.allocations:
            for j, v in enumerate(row):
                totals[j] += v
        return totals

    def is_feasible(self) -> bool:
        # supply <= capacity and demand satisfied exactly (or at least not violated)
        supply_used = self.total_supply_used()
        for i, s in enumerate(self.suppliers):
            if supply_used[i] > s.capacity:
                return False
        demand_rec = self.total_demand_received()
        for j, c in enumerate(self.consumers):
            if demand_rec[j] < c.demand:
                return False
        # also nonnegative
        for row in self.allocations:
            for v in row:
                if v < 0:
                    return False
        return True

# -----------------------------
# Utilities for CSV IO / random instance
# -----------------------------
def load_suppliers(path="suppliers.csv") -> Optional[List[Supplier]]:
    if not os.path.exists(path):
        return None
    suppliers = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].strip().startswith('#'):
                continue
            if row[0].lower() == 'id' or row[0].lower() == 'supplier':
                continue
            suppliers.append(Supplier(row[0].strip(), int(row[1])))
    return suppliers

def load_consumers(path="consumers.csv") -> Optional[List[Consumer]]:
    if not os.path.exists(path):
        return None
    consumers = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].strip().startswith('#'):
                continue
            if row[0].lower() == 'id' or row[0].lower() == 'consumer':
                continue
            consumers.append(Consumer(row[0].strip(), int(row[1])))
    return consumers

def load_costs(path="costs.csv") -> Optional[List[List[float]]]:
    if not os.path.exists(path):
        return None
    costs = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # allow rows with string headers: skip non-numeric cells at start
            numeric = []
            for cell in row:
                cell = cell.strip()
                if cell == '':
                    continue
                try:
                    numeric.append(float(cell))
                except:
                    # skip header-like or id columns
                    continue
            if numeric:
                costs.append(numeric)
    return costs

def save_solution_csv(solution: TransportSolution, path="transport_output.csv"):
    with open(path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["supplier", "consumer", "amount"])
        for i, s in enumerate(solution.suppliers):
            for j, c in enumerate(solution.consumers):
                amt = solution.allocations[i][j]
                if amt != 0:
                    writer.writerow([s.id, c.id, amt])

def random_instance(m=4, n=5, cap_low=50, cap_high=150, dem_low=30, dem_high=120, cost_low=1, cost_high=20):
    suppliers = [Supplier(f"S{i}", random.randint(cap_low, cap_high)) for i in range(m)]
    consumers = [Consumer(f"C{j}", random.randint(dem_low, dem_high)) for j in range(n)]
    # ensure total supply >= total demand (balance by scaling supplies up if necessary)
    total_supply = sum(s.capacity for s in suppliers)
    total_demand = sum(c.demand for c in consumers)
    if total_supply < total_demand:
        # scale up suppliers proportionally
        factor = math.ceil(total_demand / max(1, total_supply))
        for s in suppliers:
            s.capacity *= factor
    costs = [[random.randint(cost_low, cost_high) for _ in range(n)] for _ in range(m)]
    return suppliers, consumers, costs

# -----------------------------
# Solution construction / repair helpers
# -----------------------------
def build_random_feasible_solution(suppliers: List[Supplier], consumers: List[Consumer]) -> List[List[int]]:
    """
    Build a simple feasible integer solution: greedily fill consumer demands by iterating suppliers
    in random order and assigning as much as possible to unmet demands (also random consumer order).
    """
    m = len(suppliers)
    n = len(consumers)
    alloc = [[0 for _ in range(n)] for _ in range(m)]
    remaining_supply = [s.capacity for s in suppliers]
    remaining_demand = [c.demand for c in consumers]

    # random supplier and consumer orders
    supplier_order = list(range(m))
    random.shuffle(supplier_order)

    for i in supplier_order:
        # random consumer preference each supplier
        consumer_order = list(range(n))
        random.shuffle(consumer_order)
        for j in consumer_order:
            if remaining_supply[i] <= 0:
                break
            need = remaining_demand[j]
            if need <= 0:
                continue
            give = min(remaining_supply[i], need)
            # give integer amount
            alloc[i][j] += give
            remaining_supply[i] -= give
            remaining_demand[j] -= give

    # if some demand still unmet (due to rounding), try to assign from any supply that still has capacity
    for j in range(n):
        if remaining_demand[j] > 0:
            for i in range(m):
                if remaining_supply[i] <= 0:
                    continue
                give = min(remaining_supply[i], remaining_demand[j])
                alloc[i][j] += give
                remaining_supply[i] -= give
                remaining_demand[j] -= give
                if remaining_demand[j] == 0:
                    break
    return alloc

def repair_solution(alloc: List[List[int]], suppliers: List[Supplier], consumers: List[Consumer]) -> List[List[int]]:
    """
    Repair an allocation matrix to reduce violations:
    - Make sure no supplier exceeds capacity (by proportional scaling of its row)
    - Try to meet consumer demands by drawing from supplies with remaining capacity
    - Make all entries integers (they are int)
    This repair is heuristic; the GA will still use penalties for remaining infeasibilities.
    """
    m = len(suppliers)
    n = len(consumers)

    # Step 0: clip negatives
    for i in range(m):
        for j in range(n):
            if alloc[i][j] < 0:
                alloc[i][j] = 0

    # Step 1: enforce supplier capacities
    for i in range(m):
        row_sum = sum(alloc[i])
        cap = suppliers[i].capacity
        if row_sum > cap and row_sum > 0:
            factor = cap / row_sum
            # scale down row proportionally, keeping integers
            new_row = [int(math.floor(v * factor)) for v in alloc[i]]
            # ensure we didn't undershoot too much: distribute remaining capacity greedily
            curr = sum(new_row)
            remain = cap - curr
            if remain > 0:
                # add back to largest fractions from original row
                fracs = [(alloc[i][j] * factor - new_row[j], j) for j in range(n)]
                fracs.sort(reverse=True)
                for _, j in fracs:
                    if remain <= 0:
                        break
                    new_row[j] += 1
                    remain -= 1
            alloc[i] = new_row

    # Step 2: satisfy consumer demands if possible
    demand_received = [0] * n
    for j in range(n):
        for i in range(m):
            demand_received[j] += alloc[i][j]
    for j in range(n):
        need = consumers[j].demand - demand_received[j]
        if need <= 0:
            continue
        # try to find suppliers with free capacity
        for i in range(m):
            free = suppliers[i].capacity - sum(alloc[i])
            if free <= 0:
                continue
            give = min(free, need)
            alloc[i][j] += give
            need -= give
            if need <= 0:
                break
    # Step 3: final clipping to integers and capacities
    for i in range(m):
        for j in range(n):
            alloc[i][j] = int(max(0, round(alloc[i][j])))
    # Ensure no supplier exceeded
    for i in range(m):
        row_sum = sum(alloc[i])
        if row_sum > suppliers[i].capacity:
            # cut from largest allocations
            over = row_sum - suppliers[i].capacity
            while over > 0:
                j_max = max(range(n), key=lambda j: alloc[i][j])
                cut = min(over, alloc[i][j_max])
                alloc[i][j_max] -= cut
                over -= cut
    return alloc

# -----------------------------
# Fitness
# -----------------------------
def evaluate_fitness(solution: TransportSolution, cost_matrix: List[List[float]], fitness_type="basic"):
    """
    Higher fitness = better. We convert cost minimization into fitness maximization.
    Use penalties for infeasibilities.
    """
    base_cost = solution.total_cost(cost_matrix)
    # penalties
    penalty = 0.0
    # supply violation penalty
    supply_used = solution.total_supply_used()
    for i, used in enumerate(supply_used):
        cap = solution.suppliers[i].capacity
        if used > cap:
            penalty += (used - cap) * 1000.0  # heavy penalty

    # demand shortfall penalty
    demand_received = solution.total_demand_received()
    for j, rec in enumerate(demand_received):
        need = solution.consumers[j].demand
        if rec < need:
            penalty += (need - rec) * 1000.0

    # small penalties to discourage fractional-ish behavior (we are integer though)
    # advanced fitness can add balancing considerations: prefer using cheaper suppliers etc.
    balance_bonus = 0.0
    if fitness_type == "advanced":
        # reward solutions that balance supplier usage (reduce variance) and avoid tiny shipments
        usages = supply_used
        mean_u = sum(usages) / max(1, len(usages))
        var = sum((u - mean_u) ** 2 for u in usages) / max(1, len(usages))
        # lower variance is better -> add to fitness
        balance_bonus = -0.1 * var

        # reward meeting demands exactly (no oversupply)
        oversupply = 0
        for i, used in enumerate(usages):
            if used < solution.suppliers[i].capacity:
                # small reward for leaving capacity (not required but may indicate efficient allocation)
                pass

    # convert to fitness: smaller cost & penalties -> higher fitness
    fitness_value = -base_cost - penalty + balance_bonus
    return fitness_value, base_cost, penalty

# -----------------------------
# Genetic Algorithm
# -----------------------------
class TransportGA:
    def __init__(self,
                 suppliers: List[Supplier],
                 consumers: List[Consumer],
                 cost_matrix: List[List[float]],
                 population_size: int = 50,
                 selection_strategy: str = "rain",  # or "greedy"
                 fitness_type: str = "basic",
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 rng_seed: Optional[int] = None):
        self.suppliers = suppliers
        self.consumers = consumers
        self.cost_matrix = cost_matrix
        self.m = len(suppliers)
        self.n = len(consumers)
        self.pop_size = population_size
        self.selection_strategy = selection_strategy
        self.fitness_type = fitness_type
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        if rng_seed is not None:
            random.seed(rng_seed)

        self.population: List[TransportSolution] = []
        self.fitnesses: List[float] = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            alloc = build_random_feasible_solution(self.suppliers, self.consumers)
            alloc = repair_solution(alloc, self.suppliers, self.consumers)
            sol = TransportSolution(alloc, self.suppliers, self.consumers)
            self.population.append(sol)
        self._eval_population()

    def _eval_population(self):
        self.fitnesses = []
        for sol in self.population:
            f, cost, pen = evaluate_fitness(sol, self.cost_matrix, self.fitness_type)
            self.fitnesses.append(f)

    def select_parents(self) -> Tuple[TransportSolution, TransportSolution]:
        if self.selection_strategy == "greedy":
            # pick two best
            sorted_idx = sorted(range(len(self.population)), key=lambda k: self.fitnesses[k], reverse=True)
            return self.population[sorted_idx[0]], self.population[sorted_idx[1]]
        else:
            # rain: keep elite top 10% and fill rest by roulette + random
            # roulette selection by fitness (shifted to positive)
            min_f = min(self.fitnesses)
            shifted = [f - min_f + 1e-6 for f in self.fitnesses]  # make positive
            total = sum(shifted)
            probs = [s / total for s in shifted]
            p1 = random.choices(self.population, weights=probs, k=1)[0]
            p2 = random.choices(self.population, weights=probs, k=1)[0]
            return p1, p2

    def crossover(self, parent1: TransportSolution, parent2: TransportSolution) -> TransportSolution:
        """
        Crossover by row (supplier) mixing:
        For each supplier row i, choose from parent1 or parent2 with 50% probability.
        Then repair child.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy() if random.random() < 0.5 else parent2.copy()

        child_alloc = [[0 for _ in range(self.n)] for _ in range(self.m)]
        for i in range(self.m):
            if random.random() < 0.5:
                child_alloc[i] = parent1.allocations[i][:]  # copy row
            else:
                child_alloc[i] = parent2.allocations[i][:]

        child_alloc = repair_solution(child_alloc, self.suppliers, self.consumers)
        return TransportSolution(child_alloc, self.suppliers, self.consumers)

    def mutate(self, sol: TransportSolution):
        """
        Several mutation operators:
        - Move some units from one (i,j) to another (i',j') or within same supplier row
        - Randomly shuffle a supplier's distribution
        """
        if random.random() > self.mutation_rate:
            return

        m, n = self.m, self.n
        alloc = sol.allocations
        # choose operator
        op = random.choice(["move", "shuffle_row", "swap_cols"])
        if op == "move":
            # move k units from a random cell with >0 to another cell
            nonzero = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0]
            if nonzero:
                i, j = random.choice(nonzero)
                max_move = alloc[i][j]
                move = random.randint(1, max(1, int(max_move * 0.5)))
                # choose destination (prefer same consumer or same supplier)
                i2 = random.randrange(m)
                j2 = random.randrange(n)
                alloc[i][j] -= move
                alloc[i2][j2] += move
        elif op == "shuffle_row":
            i = random.randrange(m)
            row = alloc[i]
            # convert to list of units and redistribute randomly
            total = sum(row)
            if total > 0:
                parts = [0] * n
                for _ in range(total):
                    parts[random.randrange(n)] += 1
                alloc[i] = parts
        else:  # swap_cols
            j1 = random.randrange(n)
            j2 = random.randrange(n)
            if j1 != j2:
                for i in range(m):
                    alloc[i][j1], alloc[i][j2] = alloc[i][j2], alloc[i][j1]

        # final repair
        sol.allocations = repair_solution(alloc, sol.suppliers, sol.consumers)

    def evolve(self, generations: int = 200, verbose: bool = True):
        self.initialize_population()
        best_history = []
        for gen in range(generations):
            new_population: List[TransportSolution] = []
            # elitism: carry the best solution
            best_idx = max(range(len(self.population)), key=lambda k: self.fitnesses[k])
            best_sol = self.population[best_idx].copy()
            best_fit = self.fitnesses[best_idx]
            new_population.append(best_sol)

            while len(new_population) < self.pop_size:
                p1, p2 = self.select_parents()
                child = self.crossover(p1, p2)
                self.mutate(child)
                new_population.append(child)

            self.population = new_population
            self._eval_population()

            # record best
            best_idx = max(range(len(self.population)), key=lambda k: self.fitnesses[k])
            best_fit = self.fitnesses[best_idx]
            best_sol = self.population[best_idx]
            fval, cost, pen = evaluate_fitness(best_sol, self.cost_matrix, self.fitness_type)
            best_history.append((gen, fval, cost, pen))

            if verbose and (gen % max(1, generations//10) == 0 or gen < 10):
                print(f"Gen {gen}: best fitness={fval:.2f}, cost={cost:.2f}, penalty={pen:.2f}")

        # final best
        best_idx = max(range(len(self.population)), key=lambda k: self.fitnesses[k])
        return self.population[best_idx], best_history

# -----------------------------
# CLI and example run
# -----------------------------
def main():
    # Try to load CSVs
    suppliers = load_suppliers("suppliers.csv")
    consumers = load_consumers("consumers.csv")
    costs = load_costs("costs.csv")

    if suppliers is None or consumers is None or costs is None:
        print("One or more CSV input files not found. Generating random instance.")
        suppliers, consumers, costs = random_instance(m=5, n=6)
    else:
        # validate size matches
        m = len(suppliers)
        n = len(consumers)
        if len(costs) != m or any(len(row) != n for row in costs):
            print("Warning: cost matrix shape doesn't match suppliers/consumers. Attempting to adapt...")
            # try to adapt if costs provided transposed or with header; else regenerate random costs
            if len(costs) == n and all(len(row) == m for row in costs):
                # transpose
                costs = [list(col) for col in zip(*costs)]
            else:
                print("Failed to adapt cost matrix shape. Generating random costs for given sizes.")
                costs = [[random.randint(1, 20) for _ in range(n)] for _ in range(m)]

    # Print instance summary
    print("Suppliers:")
    for s in suppliers:
        print(f"  {s.id}: capacity={s.capacity}")
    print("Consumers:")
    for c in consumers:
        print(f"  {c.id}: demand={c.demand}")
    print("Cost matrix (rows=suppliers, cols=consumers):")
    for r in costs:
        print("  ", r)

    # instantiate GA
    ga = TransportGA(
        suppliers=suppliers,
        consumers=consumers,
        cost_matrix=costs,
        population_size=60,
        selection_strategy="rain",  # "greedy" or "rain"
        fitness_type="advanced",    # "basic" or "advanced"
        mutation_rate=0.25,
        crossover_rate=0.8,
        rng_seed=42
    )

    best_sol, history = ga.evolve(generations=200, verbose=True)
    best_fitness, best_cost, best_penalty = evaluate_fitness(best_sol, costs, ga.fitness_type)
    print("\n=== Best solution summary ===")
    print(f"Fitness: {best_fitness:.2f}")
    print(f"Total transport cost: {best_cost:.2f}")
    print(f"Penalty: {best_penalty:.2f}")
    print("Allocations (nonzero):")
    for i, s in enumerate(suppliers):
        for j, c in enumerate(consumers):
            amt = best_sol.allocations[i][j]
            if amt:
                print(f"  {s.id} -> {c.id} : {amt}")

    # Save CSV
    save_solution_csv(best_sol, "transport_output.csv")
    print("Saved best allocation to transport_output.csv")

if __name__ == "__main__":
    main()
