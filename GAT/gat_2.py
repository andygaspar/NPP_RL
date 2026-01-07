import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
import numpy as np


class ImprovedEGAT(torch.nn.Module):
    def __init__(self, node_channels, edge_channels, hidden_channels=128, out_channels=2,
                 heads=4, dropout=0.3, lr=0.001, wd=1e-5, device=torch.device('cpu')):
        super().__init__()

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.device = device

        # ENHANCEMENT 1: Add skip connections to prevent over-smoothing
        self.skip1 = nn.Linear(node_channels, hidden_channels * heads)
        self.skip2 = nn.Linear(hidden_channels * heads, hidden_channels * heads)

        # ENHANCEMENT 2: Layer normalization instead of batch norm (better for graphs)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        # ENHANCEMENT 3: Add edge feature projection
        self.edge_proj = nn.Linear(edge_channels, hidden_channels)

        # ENHANCEMENT 4: Use fewer heads in later layers (prevent attention collapse)
        self.conv1 = GATv2Conv(
            in_channels=node_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_channels,
            dropout=dropout,
            concat=True,
            add_self_loops=True  # Helps isolated nodes
        )

        # ENHANCEMENT 5: Add residual connection
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads // 2,  # Reduce heads
            edge_dim=edge_channels,
            dropout=dropout,
            concat=True
        )

        # ENHANCEMENT 6: Separate heads for mu and sigma
        self.mu_head = GATv2Conv(
            in_channels=hidden_channels * (heads // 2),
            out_channels=1,
            heads=1,
            edge_dim=edge_channels,
            dropout=dropout,
            concat=False
        )

        self.sigma_head = GATv2Conv(
            in_channels=hidden_channels * (heads // 2),
            out_channels=1,
            heads=1,
            edge_dim=edge_channels,
            dropout=dropout,
            concat=False
        )

        # ENHANCEMENT 7: Add node type embedding (if you have path nodes vs other nodes)
        self.node_type_embed = nn.Embedding(2, 16)  # Assuming binary node type

        # ENHANCEMENT 8: Add global context (graph-level features)
        self.global_pool = global_add_pool
        self.global_proj = nn.Linear(hidden_channels * (heads // 2), 32)

        # ENHANCEMENT 9: Output scaling parameters (learnable)
        self.mu_scale = nn.Parameter(torch.tensor(1.0))
        self.mu_bias = nn.Parameter(torch.tensor(0.0))

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        if self.device.type == 'cuda':
            self.to(self.device)

    def forward(self, batch, n_samples=1, return_attention=False):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        # Add node type information if available
        if x.shape[1] > self.node_channels:
            # Assume last column is node type indicator
            node_types = x[:, -1].long()
            type_embed = self.node_type_embed(node_types)
            x = torch.cat([x[:, :self.node_channels], type_embed], dim=1)
            node_channels_actual = self.node_channels + 16
        else:
            node_channels_actual = self.node_channels

        # Project edge features
        edge_attr_proj = F.elu(self.edge_proj(edge_attr))

        # First GATv2 layer with skip connection
        x1 = self.conv1(x[:, :node_channels_actual], edge_index, edge_attr=edge_attr_proj)
        x1_skip = self.skip1(x[:, :self.node_channels])
        x1 = x1 + x1_skip  # Skip connection
        x1 = F.elu(self.norm1(x1))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        # Second GATv2 layer
        x2 = self.conv2(x1, edge_index, edge_attr=edge_attr_proj)
        x2_skip = self.skip2(x1)
        x2 = x2 + x2_skip  # Skip connection
        x2 = F.elu(self.norm2(x2))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        # Separate heads for mu and sigma
        mu_raw = self.mu_head(x2, edge_index, edge_attr=edge_attr_proj).squeeze(-1)
        sigma_raw = self.sigma_head(x2, edge_index, edge_attr=edge_attr_proj).squeeze(-1)

        # ENHANCEMENT 10: Apply different activations with learnable scaling
        mu = 0.5 + 0.5 * torch.tanh(self.mu_scale * mu_raw + self.mu_bias)

        # ENHANCEMENT 11: Ensure sigma has sufficient range
        sigma = 0.01 + 0.49 * torch.sigmoid(sigma_raw)  # sigma in [0.01, 0.5]

        # Sampling
        if n_samples > 0:
            eps = torch.randn((n_samples, len(mu)), device=mu.device)
            samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
            samples = torch.clamp(samples, 0, 1)

            # Calculate log probabilities
            mu_expanded = mu.unsqueeze(0).expand(n_samples, -1)
            sigma_expanded = sigma.unsqueeze(0).expand(n_samples, -1)
            log_probs = self.truncated_normal_log_prob(
                samples, mu_expanded, sigma_expanded, low=0, high=1
            )

            return samples, log_probs, mu, sigma

        return mu, sigma

    @staticmethod
    def truncated_normal_log_prob(x, mu, sigma, low=0, high=1):
        """Vectorized truncated normal log probability"""
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(x)

        # Truncation correction
        Z = dist.cdf(torch.tensor(high, device=x.device)) - dist.cdf(torch.tensor(low, device=x.device))
        log_Z = torch.log(Z + 1e-10)

        mask = (x >= low) & (x <= high)
        log_prob = torch.where(mask, log_prob - log_Z, torch.tensor(-1e10, device=x.device))

        return log_prob


class EGATWithFeatures(ImprovedEGAT):
    """
    Enhanced version with additional node features to differentiate nodes
    """

    def __init__(self, node_channels, edge_channels, hidden_channels=128, out_channels=2,
                 heads=4, dropout=0.3, lr=0.001, wd=1e-5, device=torch.device('cpu')):
        super().__init__(node_channels, edge_channels, hidden_channels, out_channels,
                         heads, dropout, lr, wd, device)

        # Additional structural features
        self.degree_encoder = nn.Linear(1, 16)
        self.positional_encoder = PositionalEncoder(dim=16)

    def forward(self, batch, n_samples=1):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        # Compute node degrees as additional feature
        if hasattr(batch, 'degree') and batch.degree is not None:
            degree_feat = batch.degree.float().unsqueeze(-1)
        else:
            # Compute degree from edge_index
            degree = torch.zeros(x.shape[0], device=x.device)
            unique, counts = torch.unique(edge_index[0], return_counts=True)
            degree[unique] = counts.float()
            degree_feat = degree.unsqueeze(-1)

        degree_encoded = F.elu(self.degree_encoder(degree_feat))

        # Add positional encoding (helps differentiate nodes)
        pos_encoded = self.positional_encoder(x.shape[0]).to(x.device)

        # Combine all features
        x_augmented = torch.cat([
            x[:, :self.node_channels],
            degree_encoded,
            pos_encoded
        ], dim=-1)

        # Update batch with augmented features
        batch.x = x_augmented

        return super().forward(batch, n_samples)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, dim, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, n_nodes):
        return self.pe[:n_nodes]


# ------------------------------------------------------------
# TRAINING IMPROVEMENTS
# ------------------------------------------------------------

def train_with_improvements(model, train_loader, val_loader, n_epochs=100):
    """
    Improved training procedure
    """
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        all_mus = []
        all_sigmas = []

        for batch_idx, batch in enumerate(train_loader):
            if model.device.type == 'cuda':
                batch = batch.to(model.device)

            n_samples = 4  # Multiple samples for better gradient estimates

            # Forward pass
            samples, log_probs, mu, sigma = model(batch, n_samples)

            # Collect statistics
            all_mus.append(mu.detach())
            all_sigmas.append(sigma.detach())

            # ENHANCEMENT: Add entropy regularization to encourage exploration
            entropy = -log_probs.mean()

            # Compute reward (this depends on your task)
            reward = compute_reward(samples, batch)  # Implement your reward function

            # ENHANCEMENT: Use baseline with variance reduction
            if batch_idx == 0:
                baseline = reward.mean()
            else:
                baseline = 0.95 * baseline + 0.05 * reward.mean()

            advantage = reward - baseline

            # ENHANCEMENT: Normalize advantage per batch
            if advantage.std() > 0:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # ENHANCEMENT: Clip advantages
            advantage = torch.clamp(advantage, -5, 5)

            # Policy loss
            policy_loss = -(advantage * log_probs).mean()

            # ENHANCEMENT: Add entropy bonus (encourages exploration)
            entropy_coeff = max(0.1, 1.0 - epoch / n_epochs * 0.8)  # Anneal entropy
            loss = policy_loss - entropy_coeff * entropy

            # ENHANCEMENT: Add regularization to prevent sigma collapse
            sigma_penalty = -torch.log(sigma + 1e-8).mean() * 0.01
            loss += sigma_penalty

            # Optimize
            model.optimizer.zero_grad()
            loss.backward()

            # ENHANCEMENT: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            model.optimizer.step()

            total_loss += loss.item()

        # Log statistics
        all_mus = torch.cat(all_mus)
        all_sigmas = torch.cat(all_sigmas)

        print(f"Epoch {epoch + 1}:")
        print(f"  Loss: {total_loss / len(train_loader):.4f}")
        print(f"  Mu stats - Mean: {all_mus.mean():.4f}, Std: {all_mus.std():.4f}")
        print(f"    Min: {all_mus.min():.4f}, Max: {all_mus.max():.4f}")
        print(f"  Sigma stats - Mean: {all_sigmas.mean():.4f}")

        # Validate
        if val_loader is not None:
            validate_model(model, val_loader)


def compute_reward(samples, batch):
    """
    Implement your task-specific reward function here
    """
    # Example: Reward based on some target (modify for your task)
    target = torch.rand_like(samples)  # Replace with actual target
    reward = -torch.abs(samples - target).mean(dim=-1)
    return reward


def validate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        all_mus = []
        for batch in val_loader:
            if model.device.type == 'cuda':
                batch = batch.to(model.device)
            mu, _ = model(batch, n_samples=0)
            all_mus.append(mu)

        all_mus = torch.cat(all_mus)
        print(f"  Validation - Mu diversity: {all_mus.std():.4f}")
        print(f"    Range: [{all_mus.min():.4f}, {all_mus.max():.4f}]")
    model.train()


def diagnose_model(model, data_loader):
    """Check if model outputs are collapsing"""
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            mu, sigma = model(batch, n_samples=0)
            print(f"Mu statistics:")
            print(f"  Mean: {mu.mean():.4f}")
            print(f"  Std: {mu.std():.4f}")
            print(f"  Min: {mu.min():.4f}, Max: {mu.max():.4f}")
            print(f"  Unique values: {torch.unique(mu.round(decimals=3)).shape[0]}")
            print(f"Sigma statistics:")
            print(f"  Mean: {sigma.mean():.4f}")

            # Check attention weights
            if hasattr(model.conv1, '_attention'):
                attn = model.conv1._attention
                print(f"Attention diversity: {attn.std():.4f}")

            break