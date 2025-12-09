import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


def truncated_normal_log_prob(x, mu, sigma, low=0, high=1):
    """
    Calculate log probability for truncated normal distribution.
    x, mu, sigma: tensors of same shape
    Returns: log_prob of same shape as x
    """
    # Standard normal log prob
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(x)

    # Correction for truncation
    Z = normal.cdf(torch.tensor(high)) - normal.cdf(torch.tensor(low))  # Normalization constant
    log_Z = torch.log(Z + 1e-10)

    # Apply truncation (0 probability outside [low, high])
    mask = (x >= low) & (x <= high)
    log_prob = torch.where(mask, log_prob - log_Z, torch.tensor(-float('inf')))

    return log_prob


# Node Classification Model using EGAT
class EGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, n_samples, lr, wd, heads=3, dropout=0.5):
        super().__init__()

        self.n_samples = n_samples
        # EGAT layers
        self.conv1 = GATv2Conv(
            in_channels=4,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=3,
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=3,
            dropout=dropout,
            concat=True
        )

        self.conv3 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,  # Single head for output
            edge_dim=3,
            dropout=dropout,
            concat=False  # Don't concat for final layer
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.dropout = dropout

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)

        # Split and transform
        mu_raw = x[:, 0]
        sigma_raw = x[:, 1]

        mu = 0.5 + 0.5 * torch.tanh(mu_raw)
        sigma = 0.05 + 0.005 * torch.sigmoid(sigma_raw) * 0.3

        eps = torch.randn((self.n_samples, len(mu)), device=mu.device)
        sample = mu + sigma * eps
        sample = torch.clamp(sample, 0, 1)  # Truncate to [0, 1]

        # Calculate log probabilities - MANUALLY with broadcasting
        # Expand mu and sigma to match sample shape
        mu_expanded = mu.unsqueeze(0).expand(self.n_samples, -1)
        sigma_expanded = sigma.unsqueeze(0).expand(self.n_samples, -1)

        log_action = truncated_normal_log_prob(sample, mu_expanded, sigma_expanded, low=0, high=1)

        return sample, log_action

    def train_policy(self, log_action, reward, baseline):

        advantage = reward - baseline

        # Normalize advantage for stability
        if advantage.std() > 0:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        loss = -(advantage * log_action).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

