# Transportation Problem Optimizer

## Overview
This project implements a **Genetic Algorithm (GA)** to solve the **Transportation Problem** — determining optimal shipments from suppliers to consumers to minimize total transportation cost, while respecting capacity and demand constraints.

The system supports random instance generation or CSV-based input for custom datasets.

## Features
- **CSV-based input** for easy data management:
  - `suppliers.csv` → supplier IDs and capacities
  - `consumers.csv` → consumer IDs and demands
  - `costs.csv` → cost matrix (rows = suppliers, columns = consumers)
- **Automatic random instance generation** if CSVs are not provided
- **Two selection strategies**:
  - **Greedy:** always selects best individuals
  - **Rain:** keeps elite 10% and fills rest by probabilistic selection
- **Two fitness evaluation modes**:
  - **Basic:** minimizes cost with heavy penalties for infeasible allocations
  - **Advanced:** includes cost, balance of supplier usage, and demand satisfaction
- **Diverse mutation operators**:
  - Move shipment between random cells
  - Shuffle supplier’s shipments
  - Swap consumer columns
- **Repair mechanism** to restore near-feasible solutions after crossover/mutation
- **Output:** best found transportation plan saved as `transport_output.csv`

## Hard Constraints
- Total supply from each supplier ≤ its capacity
- Total demand of each consumer ≥ its required demand
- No negative shipments (all allocations ≥ 0)
- Integer-valued shipment quantities

## Example Usage
```bash
# Activate isolated python environment
source venv/bin/activate

# Run algorithm (with or without CSVs)
python main.py
