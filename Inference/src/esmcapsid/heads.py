from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .releases import HEADS_DIRECTORY


def binary_mlp(input_dimension: int, hidden_dimensions: list[int]) -> nn.Sequential:
    layers = []
    previous = input_dimension
    for dimension in hidden_dimensions:
        layers.extend([nn.Linear(previous, dimension), nn.ReLU(), nn.Dropout(0.1)])
        previous = dimension
    layers.append(nn.Linear(previous, 2))
    return nn.Sequential(*layers)


class ScreeningHeads:
    def __init__(self, model_directory: Path):
        root = model_directory / HEADS_DIRECTORY
        config = json.loads((root / "two_stage_config.json").read_text())
        self.models = {}
        self.thresholds = {}
        self.hidden_index = None
        for name in ("normal_head", "hardneg_head"):
            metadata = config[name]
            directory = root / metadata["relative_directory"]
            exported = json.loads((directory / "export_config.json").read_text())
            checkpoint = directory / "layer16_binary_mlp_head.pt"
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {name}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            dimensions = payload.get("hidden_dims") or exported.get("hidden_dims")
            if not dimensions:
                dimensions = [payload.get("hidden_dim") or exported["hidden_dim"]]
            model = binary_mlp(int(payload["input_dim"]), list(map(int, dimensions)))
            state = payload["state_dict"]
            if state and all(key.startswith("net.") for key in state):
                state = {key[4:]: value for key, value in state.items()}
            model.load_state_dict(state)
            model.eval()
            mean = np.load(directory / "feature_mean.npy").astype(np.float32).reshape(-1)
            std = np.load(directory / "feature_std.npy").astype(np.float32).reshape(-1)
            if mean.shape != std.shape or mean.size != int(payload["input_dim"]):
                raise ValueError(f"Invalid normalization dimensions: {name}")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f"Invalid normalization values: {name}")
            # Preserve the published screening script's hidden_states[1 + layer].
            index = 1 + int(exported["layer"])
            if self.hidden_index is not None and self.hidden_index != index:
                raise ValueError("Screening heads request different representations")
            self.hidden_index = index
            self.models[name] = (model, mean, std)
            threshold = float(metadata["positive_threshold"])
            if not 0 <= threshold <= 1:
                raise ValueError(f"Invalid threshold: {name}")
            self.thresholds[name] = threshold

    @torch.inference_mode()
    def predict(self, features: np.ndarray):
        scores = {}
        for name, (model, mean, std) in self.models.items():
            if features.shape[1] != mean.size:
                raise ValueError(f"Embedding dimension does not match {name}")
            normalized = ((features - mean) / std).astype(np.float32)
            probabilities = torch.softmax(model(torch.from_numpy(normalized)), dim=1)[:, 1]
            scores[name] = probabilities.numpy()
            if not np.isfinite(scores[name]).all():
                raise ValueError(f"Invalid scores: {name}")
        passed = ((scores["normal_head"] >= self.thresholds["normal_head"]) &
                  (scores["hardneg_head"] >= self.thresholds["hardneg_head"]))
        return scores["normal_head"], scores["hardneg_head"], passed


class PropertyNetwork(nn.Module):
    def __init__(self, input_dimension: int, class_count: int, hidden_dimension: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dimension, hidden_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_dimension, class_count)

    def forward(self, features):
        return self.layer2(self.dropout(self.relu(self.layer1(F.normalize(features, p=2, dim=1)))))


class PropertyHeads:
    """Optional legacy property heads. Never apply them to the public C layer-35 vector."""

    def __init__(self, root: Path):
        self.heads = []
        seen = set()
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            config_file = directory / "config.json"
            weight_file = directory / "best_model.pth"
            if not config_file.is_file() or not weight_file.is_file():
                raise ValueError(f"Incomplete property head: {directory}")
            config = json.loads(config_file.read_text())
            name = config.get("label_name") or config.get("label_col") or directory.name
            if name in seen:
                raise ValueError(f"Duplicate property task: {name}")
            seen.add(name)
            classes = config["classes"]
            if len(classes) != config["num_classes"]:
                raise ValueError(f"Class mapping mismatch: {name}")
            model = PropertyNetwork(config["input_dim"], config["num_classes"], config["hidden_dim"])
            model.load_state_dict(torch.load(weight_file, map_location="cpu", weights_only=True))
            model.eval()
            threshold = float(config.get("predict_threshold", 0.85))
            if not 0 <= threshold <= 1:
                raise ValueError(f"Invalid property threshold: {name}")
            self.heads.append((name, model, classes, threshold, config.get("task_order", 9999)))
        self.heads.sort(key=lambda item: (item[4], item[0]))
        if not self.heads:
            raise ValueError(f"No property heads found: {root}")

    @torch.inference_mode()
    def predict(self, features):
        result = {}
        for name, model, classes, threshold, _ in self.heads:
            probabilities = torch.softmax(model(torch.from_numpy(features.astype(np.float32))), dim=1)
            confidence, labels = probabilities.max(dim=1)
            if not torch.isfinite(confidence).all():
                raise ValueError(f"Invalid property scores: {name}")
            result[name] = [(str(classes[index]) if score >= threshold else "Unknown", float(score))
                            for index, score in zip(labels.tolist(), confidence.tolist())]
        return result
