from __future__ import annotations

import torch
from torch import nn

from .common import make_activation


class DeepPurposeMLP(nn.Module):
    """Feed-forward regressor over fixed drug and target descriptors."""

    def __init__(
        self,
        drug_feature_dim: int,
        protein_feature_dim: int,
        hidden_dims: list[int],
        activation: str,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError(f"Only scalar regression output is supported, got output_dim={output_dim}")

        layers: list[nn.Module] = []
        input_dim = drug_feature_dim + protein_feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        drug_indices: torch.Tensor,
        protein_indices: torch.Tensor,
        drug_features: torch.Tensor,
        protein_features: torch.Tensor,
    ) -> torch.Tensor:
        del drug_indices, protein_indices
        features = torch.cat([drug_features, protein_features], dim=-1)
        return self.network(features).squeeze(-1)
