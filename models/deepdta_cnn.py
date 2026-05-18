from __future__ import annotations

import torch
from torch import nn

from .common import make_activation


class DeepDTASequenceCNN(nn.Module):
    def __init__(
        self,
        smiles_vocab_size: int,
        protein_vocab_size: int,
        sequence_embedding_dim: int,
        drug_filters: list[int],
        protein_filters: list[int],
        drug_kernel_sizes: list[int],
        protein_kernel_sizes: list[int],
        dense_dims: list[int],
        activation: str,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError(f"Only scalar regression output is supported, got output_dim={output_dim}")
        if len(drug_filters) != len(drug_kernel_sizes):
            raise ValueError("model.drug_filters and model.drug_kernel_sizes must have equal length")
        if len(protein_filters) != len(protein_kernel_sizes):
            raise ValueError("model.protein_filters and model.protein_kernel_sizes must have equal length")

        self.drug_embedding = nn.Embedding(
            smiles_vocab_size,
            sequence_embedding_dim,
            padding_idx=0,
        )
        self.protein_embedding = nn.Embedding(
            protein_vocab_size,
            sequence_embedding_dim,
            padding_idx=0,
        )
        self.drug_encoder = self.make_encoder(sequence_embedding_dim, drug_filters, drug_kernel_sizes)
        self.protein_encoder = self.make_encoder(
            sequence_embedding_dim,
            protein_filters,
            protein_kernel_sizes,
        )

        layers: list[nn.Module] = []
        input_dim = drug_filters[-1] + protein_filters[-1]
        for hidden_dim in dense_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.regressor = nn.Sequential(*layers)

    @staticmethod
    def make_encoder(
        input_dim: int,
        filters: list[int],
        kernel_sizes: list[int],
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        channels = input_dim
        for output_channels, kernel_size in zip(filters, kernel_sizes):
            layers.append(nn.Conv1d(channels, output_channels, kernel_size=kernel_size))
            layers.append(nn.ReLU())
            channels = output_channels
        return nn.Sequential(*layers)

    def encode_branch(
        self,
        embedding: nn.Embedding,
        encoder: nn.Sequential,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence = embedding(tokens).transpose(1, 2)
        encoded = encoder(sequence)
        return torch.amax(encoded, dim=-1)

    def forward(
        self,
        drug_indices: torch.Tensor,
        protein_indices: torch.Tensor,
        drug_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
    ) -> torch.Tensor:
        del drug_indices, protein_indices
        drug_features = self.encode_branch(self.drug_embedding, self.drug_encoder, drug_tokens)
        protein_features = self.encode_branch(
            self.protein_embedding,
            self.protein_encoder,
            protein_tokens,
        )
        features = torch.cat([drug_features, protein_features], dim=-1)
        return self.regressor(features).squeeze(-1)
