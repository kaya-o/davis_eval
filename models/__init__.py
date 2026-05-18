from __future__ import annotations

from typing import Any

from torch import nn

from .deepdta_cnn import DeepDTASequenceCNN
from .deeppurpose_mlp import DeepPurposeMLP


def build_model(
    config: dict[str, Any],
    num_drugs: int,
    num_proteins: int,
    sequence_feature_info: dict[str, Any] | None = None,
    descriptor_feature_info: dict[str, Any] | None = None,
) -> nn.Module:
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Missing or invalid 'model' section in config")

    model_type = model_config.get("type")
    if model_type == "deepdta_cnn":
        if sequence_feature_info is None:
            raise ValueError("deepdta_cnn requires sequence_feature_info")
        return DeepDTASequenceCNN(
            smiles_vocab_size=int(sequence_feature_info["drug_vocab_size"]),
            protein_vocab_size=int(sequence_feature_info["protein_vocab_size"]),
            sequence_embedding_dim=int(model_config["sequence_embedding_dim"]),
            drug_filters=[int(value) for value in model_config["drug_filters"]],
            protein_filters=[int(value) for value in model_config["protein_filters"]],
            drug_kernel_sizes=[int(value) for value in model_config["drug_kernel_sizes"]],
            protein_kernel_sizes=[int(value) for value in model_config["protein_kernel_sizes"]],
            dense_dims=[int(value) for value in model_config["dense_dims"]],
            activation=str(model_config["activation"]),
            dropout=float(model_config["dropout"]),
            output_dim=int(model_config["output_dim"]),
        )

    if model_type == "deeppurpose_mlp":
        if descriptor_feature_info is None:
            raise ValueError("deeppurpose_mlp requires descriptor_feature_info")
        hidden_dims = model_config.get("hidden_dims")
        if not isinstance(hidden_dims, list) or not hidden_dims:
            raise ValueError("model.hidden_dims must be a non-empty list")
        return DeepPurposeMLP(
            drug_feature_dim=int(descriptor_feature_info["drug_feature_dim"]),
            protein_feature_dim=int(descriptor_feature_info["protein_feature_dim"]),
            hidden_dims=[int(dim) for dim in hidden_dims],
            activation=str(model_config["activation"]),
            dropout=float(model_config["dropout"]),
            output_dim=int(model_config["output_dim"]),
        )

    raise ValueError(
        "This pipeline expects deeppurpose_mlp or deepdta_cnn, "
        f"got model.type={model_type!r}"
    )


__all__ = [
    "DeepDTASequenceCNN",
    "DeepPurposeMLP",
    "build_model",
]
