import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt

from Instance.instance import Instance


# Node Classification Model using EGAT
class EGATNodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout=0.5):
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


import torch
import numpy as np
from torch_geometric.data import HeteroData
import networkx as nx


class InstanceToHeterograph:
    def __init__(self, instance):
        """
        Convert an Instance (nx.Graph) to PyTorch Geometric Heterograph

        Args:
            instance: Instance object from your code
        """
        self.instance = instance
        self.data = HeteroData()

    def convert(self):
        """Convert the instance to heterograph format"""

        # Extract node indices and features
        commodity_nodes = []
        path_nodes = []

        # Separate commodity and path nodes
        for node, attrs in self.instance.nodes(data=True):
            if attrs['type'] == 'commodity':
                commodity_nodes.append(node)
            elif attrs['type'] == 'path':
                path_nodes.append(node)

        # Create node mapping dictionaries
        self.commodity_idx = {node: idx for idx, node in enumerate(commodity_nodes)}
        self.path_idx = {node: idx for idx, node in enumerate(path_nodes)}

        # 1. Commodity node features (n_users)
        commodity_features = []
        for node in commodity_nodes:
            # Find the commodity object for this node
            commodity_obj = next(c for c in self.instance.commodities if c.name == node)
            # Feature: number of users (you can add more features here)
            feature = [commodity_obj.n_users]  # This could be expanded
            commodity_features.append(feature)

        self.data['commodity'].x = torch.tensor(commodity_features, dtype=torch.float)

        # 2. Path node features (L_p and N_p)
        path_features = []
        for node in path_nodes:
            # Find the path object for this node
            path_obj = next(p for p in self.instance.paths if p.name == node)
            # Features: L_p (lower bound) and N_p (upper bound)
            feature = [path_obj.L_p, path_obj.N_p]  # This could be expanded
            path_features.append(feature)

        self.data['path'].x = torch.tensor(path_features, dtype=torch.float)

        # 3. Edge indices and features for different edge types

        # Edges between commodities and paths (commodity_path edges)
        edge_index_commodity_path = []
        edge_attr_commodity_path = []

        # Edges between commodities (commodity_commodity edges)
        edge_index_commodity_commodity = []
        edge_attr_commodity_commodity = []

        for u, v, attrs in self.instance.edges(data=True):
            u_type = self.instance.nodes[u]['type']
            v_type = self.instance.nodes[v]['type']

            # Edge between commodity and path
            if (u_type == 'commodity' and v_type == 'path') or (u_type == 'path' and v_type == 'commodity'):
                if u_type == 'commodity':
                    src_idx = self.commodity_idx[u]
                    tgt_idx = self.path_idx[v]
                else:
                    src_idx = self.path_idx[u]
                    tgt_idx = self.commodity_idx[v]
                    # For bidirectional edges, we might want both directions
                    # Let's create edges from commodity to path only
                    continue  # Skip path->commodity direction for now

                edge_index_commodity_path.append([src_idx, tgt_idx])
                edge_attr_commodity_path.append([attrs.get('transfer', 0.0)])

            # Edge between two commodities
            elif u_type == 'commodity' and v_type == 'commodity':
                src_idx = self.commodity_idx[u]
                tgt_idx = self.commodity_idx[v]
                edge_index_commodity_commodity.append([src_idx, tgt_idx])
                edge_attr_commodity_commodity.append([attrs.get('transfer', 0.0)])

        # Add edge indices and features to heterograph
        if edge_index_commodity_path:
            self.data['commodity', 'transfer', 'path'].edge_index = torch.tensor(
                edge_index_commodity_path, dtype=torch.long).t().contiguous()
            self.data['commodity', 'transfer', 'path'].edge_attr = torch.tensor(
                edge_attr_commodity_path, dtype=torch.float)

        if edge_index_commodity_commodity:
            self.data['commodity', 'free_transfer', 'commodity'].edge_index = torch.tensor(
                edge_index_commodity_commodity, dtype=torch.long).t().contiguous()
            self.data['commodity', 'free_transfer', 'commodity'].edge_attr = torch.tensor(
                edge_attr_commodity_commodity, dtype=torch.float)

        # Store metadata
        self.data['commodity'].num_nodes = len(commodity_nodes)
        self.data['path'].num_nodes = len(path_nodes)

        return self.data


# Alternative: Modify your Instance class to directly create Heterograph
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
                len(path.commodities)  # Number of connected commodities
            ]
            path_features.append(feature)

        data['path'].x = torch.tensor(path_features, dtype=torch.float)

        # 3. Edges: commodity -> path (with transfer costs)
        edge_index_cp = []
        edge_attr_cp = []

        for i, comm in enumerate(self.commodities):
            for j, path in enumerate(self.paths):
                # Check if edge exists in original graph
                if self.has_edge(comm.name, path.name):
                    edge_index_cp.append([i, j])
                    # Get transfer cost from commodity to this path
                    edge_attr_cp.append([comm.c_p[path.name]])

        if edge_index_cp:
            data['commodity', 'requires', 'path'].edge_index = torch.tensor(
                edge_index_cp, dtype=torch.long).t().contiguous()
            data['commodity', 'requires', 'path'].edge_attr = torch.tensor(
                edge_attr_cp, dtype=torch.float)

        # 4. Edges: commodity <-> commodity (free transfers)
        edge_index_cc = []
        edge_attr_cc = []

        for i, comm1 in enumerate(self.commodities):
            for j, comm2 in enumerate(self.commodities):
                if i >= j:  # Avoid duplicates for undirected graph
                    continue
                if self.has_edge(comm1.name, comm2.name):
                    # Add both directions for undirected edges
                    edge_attr = self[comm1.name][comm2.name]['transfer']
                    edge_index_cc.append([i, j])
                    edge_attr_cc.append([edge_attr])
                    edge_index_cc.append([j, i])
                    edge_attr_cc.append([edge_attr])

        if edge_index_cc:
            data['commodity', 'free_transfer', 'commodity'].edge_index = torch.tensor(
                edge_index_cc, dtype=torch.long).t().contiguous()
            data['commodity', 'free_transfer', 'commodity'].edge_attr = torch.tensor(
                edge_attr_cc, dtype=torch.float)

        # Store node indices for reference
        data.commodity_idx = {comm.name: i for i, comm in enumerate(self.commodities)}
        data.path_idx = {path.name: i for i, path in enumerate(self.paths)}

        return data


# Example usage:
if __name__ == "__main__":
    # Create an instance
    instance = HeteroInstance(
        n_paths=5,
        n_commodities=10,
        partial=True,
        cr_transfer=(1, 20),
        nr_users=(1, 10),
        seed=42
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