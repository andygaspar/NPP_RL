import torch
from NPP.Instance.instance import Instance
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData
import numpy as np



# Node Classification Model using EGAT
class EGATNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=3, dropout=0.5):
        super().__init__()

        # EGAT layers
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=8,  # Match edge_attr dimension
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=8,
            dropout=dropout,
            concat=True
        )

        self.conv3 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,  # Single head for output
            edge_dim=8,
            dropout=dropout,
            concat=False  # Don't concat for final layer
        )

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First EGAT layer
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second EGAT layer
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third EGAT layer
        x = self.conv3(x, edge_index, edge_attr)

        return F.log_softmax(x, dim=-1)


class HeteroInstance(Instance):
    def __init__(self, n_paths, n_commodities, **kwargs):
        super().__init__(n_paths, n_commodities, **kwargs)
        self.hetero_data = self.to_heterograph()

    def to_heterograph(self):
        """Convert to PyTorch Geometric Heterograph"""
        data = HeteroData()

        # 1. Commodity nodes features
        commodity_features = []
        for comm in self.commodities:
            # You can add more features here
            feature = [
                comm.n_users,  # Number of users
                comm.c_od,  # Tax-free cost
                np.mean(list(comm.c_p.values()))  # Average transfer cost
            ]
            commodity_features.append(feature)

        data['commodity'].x = torch.tensor(commodity_features, dtype=torch.float)

        # 2. Path nodes features
        path_features = []
        for path in self.paths:
            feature = [
                path.L_p,  # Lower bound
                path.N_p,  # Upper bound
            ]
            path_features.append(feature)

        data['path'].x = torch.tensor(path_features, dtype=torch.float)

        # 3. Edges: commodity -> path (with transfer costs)
        edge_index_cp = []
        edge_attr_cp = []

        for i, comm in enumerate(self.commodities):
            for j, path in enumerate(self.paths):
                edge_index_cp.append([i, j])
                # Get transfer cost from commodity to this path
                edge_attr_cp.append([comm.c_p[path.name]])

        if edge_index_cp:
            data['commodity', 'requires', 'path'].edge_index = torch.tensor(
                edge_index_cp, dtype=torch.long).t().contiguous()
            data['commodity', 'requires', 'path'].edge_attr = torch.tensor(
                edge_attr_cp, dtype=torch.float)

        # 4. Edges: commodity <-> commodity (free transfers)
        edge_index_pp = []
        edge_attr_pp = []

        for i, path1 in enumerate(self.paths):
            for j, path2 in enumerate(self.paths):
                edge_index_pp.append([i, j])
                # edge_attr_pp.append([edge_attr])


        data['path', 'free_transfer', 'path'].edge_index = torch.tensor(
            edge_index_pp, dtype=torch.long).t().contiguous()

        # Store node indices for reference
        data.commodity_idx = {comm.name: i for i, comm in enumerate(self.commodities)}
        data.path_idx = {path.name: i for i, path in enumerate(self.paths)}

        return data


# Example usage:

# Create an instance
instance = HeteroInstance(
    n_paths=2,
    n_commodities=2,
)

# Access the heterograph data
hetero_data = instance.hetero_data

print("Heterograph created:")
print(f"Commodity nodes: {hetero_data['commodity'].x.shape}")
print(f"Path nodes: {hetero_data['path'].x.shape}")

if 'commodity' in hetero_data.edge_types:
    for edge_type in hetero_data.edge_types:
        print(f"Edge type {edge_type}: {hetero_data[edge_type].edge_index.shape}")


# You can also create a batch of instances
def create_batch(instances):
    """Create a batch of heterographs"""
    from torch_geometric.data import Batch

    hetero_data_list = [inst.hetero_data for inst in instances]
    batch = Batch.from_data_list(hetero_data_list)
    return batch


# Create multiple instances
instances = [HeteroInstance(5, 10, seed=i) for i in range(3)]
batch = create_batch(instances)

print(f"\nBatch created:")
print(f"Total commodity nodes in batch: {batch['commodity'].x.shape[0]}")
print(f"Total path nodes in batch: {batch['path'].x.shape[0]}")

# Access batch information
print(f"Commodity batch assignment: {batch['commodity'].batch}")
print(f"Path batch assignment: {batch['path'].batch}")


net = EGATNodeClassifier()