import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

from SKPP.skpp_instance import SPKK_instance


class SKKGraph(SPKK_instance):
    def __init__(self, M, K):
        super().__init__(M, K)
        self.data = self.to_graph()

    def to_graph(self):

        # Node features
        # Item nodes (type 0): [value, type_encoding, 0]
        # Type A items: first L items
        # Type B items: remaining M-L items

        # Create item node features
        item_features = []
        for i in range(self.M):
            item_type = 1 if i < self.L else 0  # 1 for type A, 0 for type B
            # Feature: [value, *, one hot]
            item_feature = [self.p[:, i].mean(), self.w[:, i].mean(), 0, 0, 1 - item_type, item_type]
            item_features.append(item_feature)

        # Create user node features
        user_features = []
        for k in range(self.K):
            # Feature: [capacity, 0 for padding, 1 for indicating user node]
            user_feature = [0, 0, self.c[k], 1, 0, 0]
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
                edge_feature = [self.w[k, i], self.p[k, i], item_type, 1 - item_type]
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

        data.x[:, 0] /= self.max_p
        data.x[:, 1] /= self.c.max()
        data.edge_attr[:, 0] /= self.w.max()

        return data

    def rescale_p(self, t: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(t.detach().to('cpu').numpy()) * self.max_p

    def eval_sample(self, sample_tensor: torch.Tensor):
        p = self.rescale_p(sample_tensor)
        from KP_Solver.knap_cpp import KnapCpp
        bb = KnapCpp(self, pop_size=sample_tensor.shape[0])
        return bb.solve_cpp(p)

    def eval(self, p: torch.Tensor):
        p = torch.stack([p] * 2)
        _, val = self.eval_sample(p)
        return val

    def random_baseline(self, sample_size):
        random_sol = np.random.uniform(0, self.max_p, (sample_size, self.L))
        from KP_Solver.knap_cpp import KnapCpp
        bb = KnapCpp(self, pop_size=sample_size)
        _, max_val = bb.solve_cpp(random_sol)
        return max_val


def create_SKPP_batch(instances, device=torch.device('cpu')):
    data_list = [inst.data for inst in instances]
    batch = Batch.from_data_list(data_list)
    if device.type == 'cuda':
        batch.to(device)

    return batch

