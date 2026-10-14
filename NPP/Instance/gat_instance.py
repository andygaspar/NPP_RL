from typing import List
import torch
from NPP.Instance.instance import Instance
from torch_geometric.data import Data, Batch
import numpy as np


class GATInstance(Instance):
    def __init__(self, n_paths, n_commodities, **kwargs):
        super().__init__(n_paths, n_commodities, **kwargs)
        self.ub_minus_lb = self.upper_bounds - self.lower_bounds
        self.data = self.to_graph()

    def to_graph(self):
        """
        Bipartite commodity <-> path graph, no path-path edges.
        An edge k-p exists only if M_k^p > 0, i.e. commodity k can use path p at a positive toll.
        Commodity nodes: [n_users, c_od, 0, 0, 0, 1, 0]; path nodes: [0, 0, L_p, N_p, N_p - L_p, 0, 1].
        Edge features: [M_k^p, c_k^p, n_users_k, position of the breakpoint M_k^p in [L_p, N_p]].
        """
        n_comm = len(self.commodities)
        scale = max(self.upper_bounds.max(), 1e-6)
        span = np.maximum(self.upper_bounds - self.lower_bounds, 1e-6)

        x_comm = [[c.n_users / 10, c.c_od / 30, 0, 0, 0, 1, 0] for c in self.commodities]
        x_path = [[0, 0, p.L_p / scale, p.N_p / scale, (p.N_p - p.L_p) / scale, 0, 1] for p in self.paths]

        m = np.array([[c.M_p[p.name] for p in self.paths] for c in self.commodities])  # (commodities, paths)
        c_p = np.array([[c.c_p[p.name] for p in self.paths] for c in self.commodities])
        n_users = np.array([c.n_users for c in self.commodities], dtype=float)
        l_p = np.array([p.L_p for p in self.paths])

        k, j = np.nonzero(m > 0)
        attr = np.stack([m[k, j] / scale, c_p[k, j] / 20, n_users[k] / 10,
                         np.clip((m[k, j] - l_p[j]) / span[j], 0, 1)], axis=1)
        src, dst = np.concatenate([k, n_comm + j]), np.concatenate([n_comm + j, k])

        return Data(x=torch.tensor(x_comm + x_path, dtype=torch.float),
                    edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
                    edge_attr=torch.tensor(np.concatenate([attr, attr]), dtype=torch.float),
                    num_commodities=n_comm, num_paths=len(self.paths))

    def to_graph_egat(self):
        """Graph of the original EGAT (complete path-path subgraph), kept for the old networks."""

        # Combine all nodes with type encoding
        num_commodities = len(self.commodities)
        num_paths = len(self.paths)

        # Create node features with type indicator
        node_features = []
        node_types = []

        # Commodity features with type=0
        for comm in self.commodities:
            feature = [comm.n_users, comm.c_od, np.mean(list(comm.c_p.values())), 0, 0, 1, 0.]  # Type indicator: 0 for commodity
            node_features.append(feature)
            node_types.append(0)

        # Path features with type=1
        for path in self.paths:
            feature = [0, 0, 0, path.L_p, path.N_p, 0, 1.]  # 0 = Padding to match dimension  # Type indicator: 1 for path
            node_features.append(feature)
            node_types.append(1)

        # Create edge indices with offset for path nodes
        edge_indices = []
        edge_attrs = []

        # 1. commodity -> path edges (with offset)
        for i, comm in enumerate(self.commodities):
            for j, path in enumerate(self.paths):
                source = i  # commodity index
                target = num_commodities + j  # path index (offset by num_commodities)
                edge_indices.append([source, target])
                edge_attrs.append([comm.c_p[path.name]] + [1.0, 0.0])
                edge_indices.append([target, source])
                edge_attrs.append([comm.c_p[path.name]] + [1.0, 0.0])

        # 2. path <-> path edges (free transfers)
        for i in range(num_paths):
            for j in range(num_paths):
                source = num_commodities + i
                target = num_commodities + j
                edge_indices.append([source, target])
                edge_attrs.append([0.0, 0.0, 1.0])  # Different edge type encoding

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
            node_type=torch.tensor(node_types, dtype=torch.long),
            num_commodities=num_commodities,
            num_paths=num_paths
        )

        # Normalise features
        norm_vals = data.x.max(dim=0)[0]
        # Lp might be 0
        norm_vals[3] = 1 if norm_vals[3] == 0 else norm_vals[3]
        for i in range(data.x.shape[1]):
            data.x[:, i] /= norm_vals[i]

        edge_norm_vals = data.edge_attr.max(dim=0)[0]
        for i in range(data.edge_attr.shape[1]):
            data.edge_attr[:, i] /= edge_norm_vals[i]

        return data

    def rescale_prices(self, t: torch.Tensor) -> np.ndarray:
        prices = np.ascontiguousarray(t.detach().to('cpu').numpy())
        lb = np.repeat(self.lower_bounds, t.shape[0]).reshape(self.n_paths, -1).T
        ub_minus_lb = np.repeat(self.ub_minus_lb, t.shape[0]).reshape(self.n_paths, -1).T
        return lb + prices * ub_minus_lb

    def eval_sample(self, sample_tensor: torch.Tensor):
        from NPP.Solver.genetic_solver import Genetic
        g = Genetic(self, pop_size=sample_tensor.shape[0], verbose=False)
        g.eval(init_population=self.rescale_prices(sample_tensor))
        return g.final_vals, g.best_val

    def eval(self, prices: torch.Tensor):
        prices = torch.stack([prices]*2)
        _, val = self.eval_sample(prices)
        return val



    def random_baseline(self, sample_size):
        random_sol = np.random.uniform(self.lower_bounds, self.upper_bounds, (sample_size, self.n_paths))
        from NPP.Solver.genetic_solver import Genetic
        g = Genetic(self, pop_size=sample_size, verbose=False)
        g.run(1, init_population=random_sol)
        return g.best_val


def create_batch(instances: List[GATInstance], device=torch.device('cpu')):
    """Create a batch of heterographs"""

    hetero_data_list = [inst.data for inst in instances]
    batch = Batch.from_data_list(hetero_data_list)
    if device.type == 'cuda':
        batch.to(device)

    return batch

