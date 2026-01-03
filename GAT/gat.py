import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from NPP.Instance.gat_instance import create_batch


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
    def __init__(self, node_channels, edge_channels, hidden_channels=64, out_channels=2, lr=0.001, wd=0.0001, heads=3,
                 dropout=0.5, device=torch.device('cpu')):
        super().__init__()

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.hidden_init = hidden_channels
        self.output_init = out_channels
        self.heads_init = heads
        self.dropout_init = dropout
        self.device = device

        self.best_gap = 0

        # EGAT layers
        self.conv1 = GATv2Conv(
            in_channels=node_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_channels,
            dropout=dropout,
            concat=True
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_channels,
            dropout=dropout,
            concat=True
        )

        self.conv3 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,  # Single head for output
            edge_dim=edge_channels,
            dropout=dropout,
            concat=False  # Don't concat for final layer
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.dropout = dropout

        if self.device.type == 'cuda':
            self.to(self.device)

    def forward(self, batch, n_samples):
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
        sigma = 0.005 + torch.sigmoid(sigma_raw) * 0.03

        eps = torch.randn((n_samples, len(mu)), device=mu.device)
        sample = mu + sigma * eps
        sample = torch.clamp(sample, 0, 1)  # Truncate to [0, 1]

        # Calculate log probabilities - MANUALLY with broadcasting
        # Expand mu and sigma to match sample shape
        mu_expanded = mu.unsqueeze(0).expand(n_samples, -1)
        sigma_expanded = sigma.unsqueeze(0).expand(n_samples, -1)

        log_action = truncated_normal_log_prob(sample, mu_expanded, sigma_expanded, low=0, high=1)

        return sample, log_action, mu

    def get_distribution(self, instance, n_samples):
        batch = create_batch([instance])
        if self.device.type == 'cuda':
            batch = batch.to(self.device)
        with torch.no_grad():
            sample, _, _ = self.forward(batch, n_samples)
            mask = batch.x[:, -1] == 1  # get only paths (i.e. x[:, -1 == 1) of the batch
            return sample[:, mask]

    def get_mean(self, instance):
        batch = create_batch([instance])
        if self.device.type == 'cuda':
            batch = batch.to(self.device)
        with torch.no_grad():
            _, _, mu = self.forward(batch, 1)
            mask = batch.x[:, -1] == 1  # get only paths (i.e. x[:, -1 == 1) of the batch
            return mu[mask]

    def train_policy(self, log_action, reward, baseline):

        if reward.max() > 0:
            advantage = reward - baseline

            # Normalize advantage for stability
            if advantage.std() > 0:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        else:
            advantage = reward / (-reward.min())

        loss = -(advantage * log_action).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path):

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'hidden_channels': self.conv1.out_channels * self.conv1.heads,
                'out_channels': self.conv3.out_channels,
                'lr': self.optimizer.param_groups[0]['lr'],
                'wd': self.optimizer.param_groups[0]['weight_decay'],
                'heads': self.conv1.heads,
                'dropout': self.dropout
            },
            'init_params': {'node_channels': self.node_channels, 'edge_channels': self.edge_channels,
                            'hidden_init': self.hidden_init, 'output_init': self.output_init,
                            'heads_init': self.heads_init, 'dropout_init': self.dropout_init,
                            'best_gap': self.best_gap
                            },
        }

        torch.save(save_dict, path)

    def load(self, path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = torch.load(path, map_location=device, weights_only=False)
        self.device = device

        # Carica i pesi del modello
        self.load_state_dict(net['model_state_dict'])

        # Carica lo stato dell'ottimizzatore
        self.optimizer.load_state_dict(net['optimizer_state_dict'])
        self.to(device)

        print(f"Modello caricato da: {path}")
        print(f"Modello spostato su: {device}")

        return net


def load_agent(path, device=None) -> EGAT:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_params = torch.load(path, map_location=device, weights_only=False)['init_params']
    agent = EGAT(node_channels=init_params['node_channels'], edge_channels=init_params['edge_channels'],
                 hidden_channels=init_params['hidden_init'], out_channels=init_params['output_init'],
                 heads=init_params['heads_init'], dropout=init_params['dropout_init'])
    agent.best_gap = init_params['best_gap']
    agent.load(path, device=device)
    return agent
