import os
import random
import os
import numpy as np
import pandas as pd

from Instance.instance import Instance
from Solver.solver import GlobalSolver

# from Net.network_manager import NetworkManager

# os.system("Path/CPP/install.sh")

seed = 0


n_commodities  = 2 # numero utenti (nodi verdi)
n_paths = 3 # percorsi a pedaggio (nodi rossi)

VERBOSE = False
random.seed(0)
np.random.seed(0)

# Crea l’istanza del problema (per costruire una rete casuale)
npp = Instance(n_paths=n_paths, n_commodities=n_commodities, seed=seed)
npp.show()
# type: #commodity/path (utente, cioè uno che deve scegliere un percorso/percorso a pagamento)
# color: g/r
# n_commodities: numero di gruppi diversi

# n_users: numero di utenti che fanno lo stesso viaggio
# c_od: costo del percorso gratuito (senza pedaggio)

# N_p: massimo pedaggio che si può mettere su quel percorso (upper bound)
# L_p: minimo pedaggio “conveniente” (lower bound).
# Quindi ogni nodo rosso è un possibile percorso dove il leader può impostare un pedaggio compreso tra lower bound e upper bound


# Solver
solver = GlobalSolver(npp)
solver.solve()
print("Step 0: Pedaggi ottimali per ciascun percorso:", solver.solution_array)

# mostra il profitto totale calcolato
print("Profitto totale massimo:", solver.obj)


###
###
# Step 1: Pedaggi casuali
print("\nStep 1: Pedaggi casuali")
# Genera pedaggi casuali tra L_p e N_p per ogni percorso
random_prices = np.array([np.random.uniform(p.L_p, p.N_p) for p in npp.paths])
print("Pedaggi casuali generati:", random_prices)

# Calcola il profitto totale del leader per questi pedaggi
profit_random = npp.compute_solution_value(random_prices)
print("Profitto totale con pedaggi casuali:", profit_random)


# Step 2: tanti pedaggi casuali: calcola
print("\n Step 2: tanti pedaggi casuali")
N_TRIALS = 1000
profits = []
prices_list = []

for _ in range(N_TRIALS):
    random_prices = np.array([np.random.uniform(p.L_p, p.N_p) for p in npp.paths])
    # calcolo
    profit = npp.compute_solution_value(random_prices)
    profits.append(profit)
    prices_list.append(random_prices)

profits = np.array(profits)
prices_list = np.array(prices_list)
# Trova il massimo profitto casuale
max_profit_idx = np.argmax(profits)
print("Massimo profitto casuale:", profits[max_profit_idx])
print("Pedaggi corrispondenti:", prices_list[max_profit_idx])

# Media dei profitti casuali
print("Profitto medio dei pedaggi casuali:", profits.mean())