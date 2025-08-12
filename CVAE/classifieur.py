import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentClassifier(nn.Module):
    """
    Petit classifieur/régressseur pour prédire la condition C à partir de mu (vecteur latent).
    hidden_dim : liste des dimensions cachées (ex: [32, 16])
    """
    def __init__(self, latent_dim, cond_dim, hidden_dim=None, dropout=None):
        super().__init__()
        if hidden_dim is None or len(hidden_dim) == 0:
            hidden_dim = [latent_dim]

        layers = []
        in_dim = latent_dim
        for h in hidden_dim:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cond_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, mu):
        return self.net(mu)
