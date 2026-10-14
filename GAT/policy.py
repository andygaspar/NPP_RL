import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv

from GAT.gat import truncated_normal_log_prob, truncated_normal_sample


class TransformerPolicy(nn.Module):
    """
    Policy for the GA initialisation: one truncated normal N(mu, sigma) on [0, 1] per leader-controlled node.
    TransformerConv layers, so that edge features enter the messages (in GATv2 they only weight the attention).
    Leader-controlled nodes are flagged by x[:, -1] == 1.
    """

    def __init__(self, node_channels=7, edge_channels=4, hidden_channels=64, layers=3, heads=4, lr=3e-4, wd=1e-5,
                 sigma_min=0.02, sigma_max=0.3, device=torch.device('cpu')):
        super().__init__()
        self.init_params = dict(node_channels=node_channels, edge_channels=edge_channels,
                                hidden_channels=hidden_channels, layers=layers, heads=heads,
                                sigma_min=sigma_min, sigma_max=sigma_max)
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.device = device
        self.best_val = 0

        self.enc = nn.Linear(node_channels, hidden_channels)
        self.convs = nn.ModuleList([TransformerConv(hidden_channels, hidden_channels // heads, heads=heads,
                                                    edge_dim=edge_channels) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ELU(), nn.Linear(hidden_channels, 2))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.to(self.device)

    def distribution(self, batch):
        h = self.enc(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(F.elu(conv(h, batch.edge_index, batch.edge_attr)))
        out = self.head(h)
        mu = torch.sigmoid(out[:, 0])
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(out[:, 1])
        return mu, sigma

    def forward(self, batch, n_samples):
        """Returns sample (n_samples, n_nodes), its log-prob per node (with gradient wrt mu and sigma), and mu."""
        mu, sigma = self.distribution(batch)
        sample = truncated_normal_sample(mu, sigma, n_samples, low=0, high=1)
        log_action = truncated_normal_log_prob(sample, mu.unsqueeze(0).expand(n_samples, -1),
                                               sigma.unsqueeze(0).expand(n_samples, -1), low=0, high=1)
        return sample, log_action, mu

    def _instance_batch(self, instance):
        return Batch.from_data_list([instance.data]).to(self.device)

    def get_distribution(self, instance, n_samples):
        batch = self._instance_batch(instance)
        with torch.no_grad():
            sample, _, _ = self.forward(batch, n_samples)
            mask = batch.x[:, -1] == 1  # leader-controlled nodes
            return sample[:, mask]

    def get_mean(self, instance):
        batch = self._instance_batch(instance)
        with torch.no_grad():
            mu, _ = self.distribution(batch)
            mask = batch.x[:, -1] == 1  # leader-controlled nodes
            return mu[mask]

    def train_policy(self, log_action, reward, groups, max_norm=1.0):
        """
        REINFORCE step.
        log_action: joint log-prob of each sample (sum over the leader-controlled nodes of its instance)
        reward: value of each sample
        groups: instance index of each sample; the advantage is normalised within each instance (the other samples
        of the same instance act as baseline), so that instances of different scale are not mixed
        """
        advantage = reward.clone()
        for g in torch.unique(groups):
            sel = groups == g
            adv = advantage[sel]
            advantage[sel] = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.std() > 0 else adv - adv.mean()

        loss = -(advantage * log_action).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'init_params': self.init_params,
                    'best_val': self.best_val}, path)


def load_policy(path, device=None) -> TransformerPolicy:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(path, map_location=device, weights_only=False)
    agent = TransformerPolicy(**net['init_params'], device=device)
    agent.load_state_dict(net['model_state_dict'])
    agent.optimizer.load_state_dict(net['optimizer_state_dict'])
    agent.best_val = net['best_val']
    agent.eval()  # no dropout / batch statistics at inference
    print(f"Modello caricato da: {path}")
    return agent
