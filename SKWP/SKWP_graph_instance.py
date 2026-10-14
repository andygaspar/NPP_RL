import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

from SKWP.skwp_instance import SKWP_instance


class SKWPGraph(SKWP_instance):
    def __init__(self, M, K, extended=True):
        super().__init__(M, K)
        self.extended = extended
        self.data = self.to_graph()

    def to_graph(self):
        """
        Bipartite item <-> follower graph, no item-item edges.
        For follower k, e_k is the efficiency of the first follower item that does not fit in the greedy knapsack
        filled with follower items only: a leader item i keeps being chosen by k roughly while
        p[k, i] / w_i >= e_k, i.e. up to the threshold weight p[k, i] / e_k (the analogue of the NPP breakpoint).
        Item nodes: [mean profit, mean weight (follower items), max_w (leader items), 0, 0, follower item, leader item]
        Follower nodes: [capacity / sum of the leader max_w, 0, 0, capacity, e_k, 0, 0]
        Edge features, leader item: [p[k, i], log((p[k, i] / max_w_i) / e_k), threshold / max_w_i, 1]
                       follower item: [p[k, i], w[k, i], taken by the greedy, 0]
        """
        cap_max = self.c.max()
        p_f, w_f = self.p[:, self.L:], self.w[:, self.L:]
        e_break, taken = np.zeros(self.K), np.zeros((self.K, self.F))
        for k in range(self.K):
            capacity_left = self.c[k]
            for idx in np.argsort(-p_f[k] / w_f[k]):
                if w_f[k, idx] <= capacity_left:
                    taken[k, idx] = 1
                    capacity_left -= w_f[k, idx]
                elif e_break[k] == 0:
                    e_break[k] = p_f[k, idx] / w_f[k, idx]
        threshold = np.where(e_break[:, None] > 0, self.p[:, :self.L] / np.maximum(e_break[:, None], 1e-12),
                             self.c[:, None])
        threshold = np.minimum(threshold, self.c[:, None])
        rel_threshold = np.clip(threshold / self.max_w[None, :], 0, 1)  # (K, L)

        x_items = [[self.p[:, i].mean() / 100,
                    self.w[:, i].mean() / cap_max if i >= self.L else 0,
                    self.max_w[i] / cap_max if i < self.L else 0,
                    0, 0,
                    float(i >= self.L), float(i < self.L)] for i in range(self.M)]
        x_users = [[self.c[k] / self.max_w.sum(), 0, 0, self.c[k] / cap_max, e_break[k] / max(e_break.max(), 1e-12), 0, 0] for k in range(self.K)]

        k, i = np.meshgrid(np.arange(self.K), np.arange(self.M), indexing='ij')
        k, i = k.ravel(), i.ravel()
        leader = i < self.L
        third = np.where(leader, rel_threshold[k, np.minimum(i, self.L - 1)],
                         taken[k, np.maximum(i - self.L, 0)])
        # efficiency of leader item i at its maximum weight, relative to the break efficiency of follower k
        eff_ratio = np.where(e_break[k] > 0, (self.p[k, i] / self.max_w[np.minimum(i, self.L - 1)]) / np.maximum(e_break[k], 1e-12), np.e ** 6)
        second = np.where(leader, np.clip(np.log(np.maximum(eff_ratio, 1e-12)), -6, 6) / 6, self.w[k, i] / cap_max)
        attr = np.stack([self.p[k, i] / 100, second, third,
                         leader.astype(float)], axis=1)
        src, dst = np.concatenate([i, self.M + k]), np.concatenate([self.M + k, i])

        return Data(x=torch.tensor(x_items + x_users, dtype=torch.float),
                    edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
                    edge_attr=torch.tensor(np.concatenate([attr, attr]), dtype=torch.float),
                    num_items=self.M, num_users=self.K, L=self.L)

    def to_graph_egat(self):
        """Graph of the original EGAT (complete item-item subgraph), kept for the old networks."""

        # Node features
        # Item nodes (type 0): [value, type_encoding, 0]
        # Type A items: first L items
        # Type B items: remaining M-L items

        # Create item node features
        item_features = []
        for i in range(self.M):
            item_type = 1 if i < self.L else 0  # 1 for type A, 0 for type B
            # Feature: [value, *, one hot]

            item_feature = [self.p[:, i].mean(),
                            self.w[:, i].mean() if i >= self.L else 0,
                            self.max_w[i] if i < self.L else 0,
                            0,  # corresponding to user features
                            0,  # corresponding to user one hot encoding
                            1 - item_type,
                            item_type]

            item_features.append(item_feature)

        # Create user node features
        user_features = []
        for k in range(self.K):
            # Feature: [capacity, 0 for padding, 1 for indicating user node]
            user_feature = [0, 0, 0,  # all corresponding to item features
                            self.c[k], 1,
                            0, 0  # corresponding to item one hot encoding
                            ]

            user_features.append(user_feature)

        # Concatenate all node features
        node_features = torch.tensor(item_features + user_features, dtype=torch.float)

        # Create edges (bipartite: items ↔ users)
        edge_indices = []
        edge_features = []

        # Edge from items to users (directed for now)
        for i in range(self.M):  # item index
            for k in range(self.K):  # user index
                # Item i -> User k
                edge_indices.append([i, self.M + k])  # M item nodes come first

                # Edge feature: [weight, item_type]
                item_type = 0 if i < self.L else 1

                edge_feature = [self.w[k, i] * item_type,
                                self.p[k, i],
                                self.p[k, i] * item_type / self.w[k, i],
                                0,
                                item_type,
                                1 - item_type]
                edge_features.append(edge_feature)

        #  empty link between items
        for i in range(self.M):
            for j in range(i + 1, self.M):
                edge_indices.append([i, j])
                edge_feature = [0, 0, 0, 1, 0, 0]
                edge_features.append(edge_feature)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Make graph undirected (optional)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

        # Create PyG Data object
        data = Data(
            x=node_features,  # Node features
            edge_index=edge_index,  # Edge connections
            edge_attr=edge_attr,  # Edge features
            # node_type=node_type,  # Node types (0=item, 1=user)
            num_items=self.M,
            num_users=self.K,
            L=self.L,  # Number of type A items
        )

        cap_max = self.c.max()
        data.x[:, :4] /= cap_max
        data.edge_attr[:, :3] /= cap_max

        return data

    def rescale_w(self, t: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(t.detach().to('cpu').numpy()) * self.max_w

    def eval_sample(self, sample_tensor: torch.Tensor, method='e'):
        w = self.rescale_w(sample_tensor)
        from KP_Solver.knap_cpp import KnapCpp
        bb = KnapCpp(self, batch_size=sample_tensor.shape[0])
        if method == 'e':
            print('DEBUG')
            return bb.solve_skwp(w)
        else:
            return bb.solve_skwp_greedy(w)

    def eval(self, w: torch.Tensor, method='e'):
        w = torch.stack([w] * 2)
        _, val = self.eval_sample(w, method)
        return val

    def random_baseline(self, sample_size, method='e'):
        random_sol = np.random.uniform(0, self.max_w, (sample_size, self.L))
        from KP_Solver.knap_cpp import KnapCpp
        bb = KnapCpp(self, batch_size=sample_size)
        if method == 'e':
            print('DEBUG')
            _, max_val =  bb.solve_skwp(random_sol)
        else:
            _, max_val =  bb.solve_skwp_greedy(random_sol)
        return max_val


def create_SKWP_batch(instances, device=torch.device('cpu')):
    data_list = [inst.data for inst in instances]
    batch = Batch.from_data_list(data_list)
    if device.type == 'cuda':
        batch.to(device)

    return batch

