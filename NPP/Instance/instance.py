import os
import sys
from typing import List

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric

# import torch
# import torch_geometric
from torch_geometric.data import HeteroData
import pandas as pd
import torch


def get_feature_size(n_paths, n_commodities):
    n_users, cf_cost = 1, 1
    l_p, n_p = n_paths, n_paths
    return l_p + n_p + (n_users + cf_cost + n_paths) * n_commodities

class Commodity:

    def __init__(self, name, nr_users, com_names, path_names, graph):
        self.name = name
        self.node = graph.nodes[self.name]
        self.n_users = np.random.choice(range(*nr_users))
        commodities = com_names.copy()
        commodities.remove(self.name)
        self.destination = np.random.choice(commodities)
        self.c_od = graph[self.name][self.destination]['transfer']
        self.c_p = {p: graph[self.name][p]['transfer'] for p in path_names}
        self.c_p_vector = np.array(list(self.c_p.values()))
        self.M_p = {p: max([0, self.c_od - self.c_p[p]]) for p in path_names}
        self.node.update(
            {'n_users': self.n_users, 'type_int': 0, 'c_od': self.c_od, 'N_p': 0, 'n_commodities': len(com_names)})

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


class Path:

    def __init__(self, name, commodities: List[Commodity], graph):
        self.name = name
        self.node = graph.nodes[self.name]
        self.cost = None
        self.N_p = max([c.M_p[self.name] for c in commodities])
        self.L_p = min([c.M_p[self.name] for c in commodities])
        self.node.update({'n_users': 0, 'type_int': 1, 'c_od': 0, 'N_p': self.N_p, 'L_p': self.L_p,
                          'n_commodities': len(commodities)})

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name




class Instance(nx.Graph):
    def __init__(self, n_paths, n_commodities, partial=True, seed=None,
                 **attr):
        super().__init__(**attr)
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self.n_paths = n_paths
        self.n_commodities = n_commodities
        self.commodities: List[Commodity]
        self.paths: List[Path]
        self.users = []
        self.partial = partial

        self.MIN_CR_FREE, self.MAX_CR_FREE = (10, 30) if partial else (20, 30)
        self.MIN_CR_TRANSFER, self.MAX_CR_TRANSFER = (20, 30)
        self.MIN_N_USERS, self.MAX_N_USERS = (1, 10)
        self.cr_free = (self.MIN_CR_FREE, self.MAX_CR_FREE)
        self.cr_transfer = (self.MIN_CR_TRANSFER, self.MAX_CR_TRANSFER)
        self.nr_users = (self.MIN_N_USERS, self.MAX_N_USERS)

        com_names = [self.get_com_name(i) for i in range(self.n_commodities)]
        paths_names = [self.get_path_name(i) for i in range(self.n_paths)]

        self.add_nodes_from([(name, {'type': 'commodity', 'color': 'g'}) for name in com_names])
        self.add_nodes_from([(name, {'type': 'path', 'color': 'r'}) for name in paths_names])

        for i in range(n_commodities):
            for j in range(i + 1, n_commodities):
                self.add_edge(com_names[i], com_names[j], color='y', transfer=np.random.uniform(*self.cr_free))

        for i in range(self.n_commodities):
            for j in range(self.n_paths):
                self.add_edge(com_names[i], self.get_path_name(j), color='y',
                              transfer=np.random.uniform(*self.cr_transfer))

        self.commodities = [Commodity(name, self.nr_users, com_names, paths_names, self) for name in com_names]
        self.paths = [Path(name, self.commodities, self) for name in paths_names]

        self.commodities_tax_free = np.array([comm.c_od for comm in self.commodities])
        self.transfer_costs = np.array([comm.c_p[path.name]
                                        for comm in self.commodities for path in self.paths])
        self.upper_bounds = np.array([p.N_p for p in self.paths])
        self.lower_bounds = np.array([p.L_p for p in self.paths])
        self.n_users = np.array([comm.n_users for comm in self.commodities])

    def show(self):
        for n in self.nodes:
            print(self.nodes[n], 'kkk')
        nx.draw(self, node_color=[self.nodes[n]['color'] for n in self.nodes],
                edge_color=[self[u][v]['color'] for u, v in self.edges],
                with_labels=True, font_size=7)
        plt.show()

    def show_original(self):
        nx.draw(self,
                node_color=[self.nodes[n]['color'] for n in self.nodes],
                edge_color=[self[u][v]['color'] for u, v in self.edges],
                with_labels=True, font_size=7)
        plt.show()


    def save_problem(self, folder_name=None):
        comm: Commodity
        transfer_costs = self.transfer_costs.reshape((self.n_commodities, self.n_paths))
        if folder_name is None:
            folder_name = "TestCases/" + "comm_" + str(self.n_commodities) + "_tolls_" + str(self.n_paths)
        try:
            os.mkdir(folder_name)
            pd.DataFrame(self.commodities_tax_free).to_csv(folder_name + '/commodities_tax_free.csv',
                                                           index=False, index_label=False, header=False)
            pd.DataFrame(transfer_costs).to_csv(folder_name + '/transfer_costs.csv',
                                                index=False, index_label=False, header=False)
            pd.DataFrame(self.upper_bounds).to_csv(folder_name + '/upper_bounds.csv',
                                                   index=False, index_label=False, header=False)
            lower_bounds = np.zeros_like(self.upper_bounds)
            pd.DataFrame(lower_bounds).to_csv(folder_name + '/lower_bounds.csv',
                                              index=False, index_label=False, header=False)
            pd.DataFrame(self.n_users).to_csv(folder_name + '/n_users.csv',
                                              index=False, index_label=False, header=False)

        except:
            print(sys.exc_info())

    @staticmethod
    def get_com_name(i):
        return r"$U_{}$".format('{' + str(i) + '}')

    @staticmethod
    def get_path_name(i):
        return r"$p_{}$".format('{' + str(i) + '}')

    def compute_solution_value(self, sol):
        total_profit = 0
        i = 0
        for commodity in self.commodities:
            costs = sol + commodity.c_p_vector
            idxs = np.argsort(np.append(costs, commodity.c_od))
            s = np.append(sol, [0])
            c = np.sort(np.append(costs, commodity.c_od))
            duplicates = np.where(c == c[0])[0]
            if duplicates.shape[0] > 1:
                prices = np.append(sol, [0])
                # print(i, np.argmax(idxs[duplicates]))
                total_profit += prices[idxs[duplicates]].max() * commodity.n_users
            else:
                total_profit += s[idxs[0]] * commodity.n_users
                # print(i, idxs[0])
            i += 1
        return total_profit


    def compute_solution_value_with_tol(self, sol, tol=1e-9):
        if type(sol) == torch.Tensor:
            sol = sol.detach().cpu().numpy().flatten()
        total_profit = 0

        for commodity in self.commodities:
            costs = sol + commodity.c_p_vector
            com_cost, com_profit = commodity.c_od, 0
            for i, c in enumerate(costs):
                if c <= com_cost + tol:
                    if c < com_cost - tol:
                        com_profit = sol[i]
                        com_cost = c
                    elif sol[i] > com_profit:
                        com_profit = sol[i]
                        com_cost = c
            total_profit += com_profit * commodity.n_users
        return total_profit



    def make_torch_graph(self, solution=None):
        for com in self.commodities:
            self.nodes[com.name]['x'] = [com.n_users, com.c_od, 0, self.n_commodities]
            self.nodes[com.name]['y'] = 0
        for i, toll in enumerate(self.paths):
            self.nodes[toll.name]['x'] = [0, 0, toll.N_p, self.n_commodities]
            self.nodes[toll.name]['y'] = solution[i] if solution is not None else 0
        data_homo = torch_geometric.utils.convert.from_networkx(self.graph,
                                                                group_node_attrs=['type_int', 'n_users',
                                                                                  'free_path',
                                                                                  'N_p', 'n_commodities'],
                                                                group_edge_attrs=['transfer'])
        return data_homo

    def make_torch_hetero_graph(self):
        for com in self.commodities:
            self.nodes[com.name]['x'] = [com.n_users, com.c_od, 0]
            self.nodes[com.name]['y'] = 0
        for i, toll in enumerate(self.paths):
            self.nodes[toll.name]['x'] = [0, 0, toll.N_p]
            self.nodes[toll.name]['y'] = 9

        data_homo = self.make_torch_graph()
        data_hetero = HeteroData()
        data_hetero['commodities'].x = data_homo.x[:8, 1:3]
        data_hetero['tolls'].x = data_homo.x[self.n_commodities:, -1]
        data_hetero['tolls'].y = data_homo.y[self.n_commodities:]

        comm_tolls_idxs = torch.where(data_homo.edge_index[0] < self.n_commodities)[0]
        from_comm = data_homo.edge_index[0][comm_tolls_idxs]
        to_tolls = data_homo.edge_index[1][comm_tolls_idxs] - self.n_commodities
        data_hetero['commodities', 'transfer', 'tolls'].edge_index = torch.stack([from_comm, to_tolls])
        data_hetero['commodities', 'transfer', 'tolls'].edge_attr = data_homo.edge_attr[comm_tolls_idxs]
        return data_hetero