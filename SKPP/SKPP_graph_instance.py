import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

from SKPP.skpp_instance import SPKK_instance


class SKKGraph(SPKK_instance):
    def __init__(self, M, K):
        super().__init__(M, K)
        self.max_p = self.p.max()
        self.graph = self.create_graph_from_problem()


    def create_graph_from_problem(self):
        """
        Create a PyG graph from a Problem instance.

        Args:
            problem: Problem instance

        Returns:
            torch_geometric.data.Data object
        """

        # Node features
        # Item nodes (type 0): [value, type_encoding, 0]
        # Type A items: first L items
        # Type B items: remaining M-L items

        # Create item node features
        item_features = []
        for i in range(self.M):
            item_type = 0 if i < self.L else 1  # 0 for type A, 1 for type B
            # Feature: [value, type_encoding, 0 for padding/indicating item node]
            item_feature = [self.p[0, i], 0, item_type, 1 - item_type, 0]
            item_features.append(item_feature)

        # Create user node features
        user_features = []
        for k in range(self.K):
            # Feature: [capacity, 0 for padding, 1 for indicating user node]
            user_feature = [0, self.c[k], 0, 0, 1]
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
                edge_feature = [self.w[k, i], item_type, 1 - item_type]
                edge_features.append(edge_feature)

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Make graph undirected (optional)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr=edge_attr)

        # # Node types (for convenience)
        # node_type = torch.cat([
        #     torch.zeros(self.M, dtype=torch.long),  # 0 for items
        #     torch.ones(self.K, dtype=torch.long)  # 1 for users
        # ])

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

def create_SKK_batch(instances, device=torch.device('cpu')):
    """
    Create a batch of graphs from multiple Problem instances.

    Args:
        problems: List of Problem instances

    Returns:
        torch_geometric.data.Batch object
    """
    data_list = [inst.graph for inst in instances]
    batch = Batch.from_data_list(data_list)
    if device.type == 'cuda':
        batch.to(device)

    return batch



# Create multiple problems
problems = [SKKGraph(M=4, K=3) for _ in range(3)]

# Create batch of graphs
# batch = SKKGraph.create_batch(problems)
batch = create_SKK_batch(problems)

print("Batch Information:")
print(f"Number of graphs: {batch.num_graphs}")
print(f"Total nodes: {batch.num_nodes}")
print(f"Total edges: {batch.num_edges}")
print(f"Node feature dim: {batch.num_node_features}")
print(f"Edge feature dim: {batch.num_edge_features}")
print(f"\nBatch attributes: {batch.keys}")

# Access individual graph information
print(f"\nFirst graph in batch:")
print(f"  Items: {batch.num_items[0]}, Users: {batch.num_users[0]}")
print(f"  Type A items: {batch.L[0]}")

