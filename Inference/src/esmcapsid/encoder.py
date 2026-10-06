from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from .inputs import SequenceRecord


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu")
    return requested


class Encoder:
    """One released encoder, one requested representation, no token files."""

    def __init__(self, model_directory: Path, device: str):
        from transformers import AutoModel, AutoTokenizer

        # The reviewed public release requires custom modeling code. Never execute
        # code from arbitrary remote repositories supplied by the CLI.
        self.device = device
        self.model = AutoModel.from_pretrained(
            str(model_directory), local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.float32,
        ).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_directory), local_files_only=True, trust_remote_code=True,
        )
        self.dimension = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, records: list[SequenceRecord], *, hidden_index: int,
               max_tokens: int, residues_only: bool = False, property_features: bool = False):
        sequences = [record.sequence for record in records]
        encoded = self.tokenizer(
            sequences, padding=True, truncation=True, max_length=max_tokens,
            add_special_tokens=True, return_tensors="pt", return_special_tokens_mask=True,
        )
        attention = encoded["attention_mask"].bool()
        special = encoded.pop("special_tokens_mask").bool()
        lengths = (attention & ~special).sum(dim=1).tolist()
        arguments = {key: value.to(self.device) for key, value in encoded.items()
                     if key in {"input_ids", "attention_mask"}}
        output = self.model(**arguments, output_hidden_states=True)
        states = output.hidden_states
        if states is None or not 0 <= hidden_index < len(states):
            raise ValueError(f"Requested hidden state {hidden_index}, available: "
                             f"{0 if states is None else len(states)}")
        hidden = states[hidden_index]
        mask = (attention & ~special if residues_only else attention).to(self.device)
        if not mask.any(dim=1).all():
            raise ValueError("Tokenizer produced no usable tokens for an input sequence")
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        features = pooled.float().cpu().numpy()
        if features.shape != (len(records), self.dimension) or not np.isfinite(features).all():
            raise ValueError("Encoder produced invalid pooled embeddings")
        legacy = None
        if property_features:
            # The optional paper heads use the legacy hidden_states[1 + 33]
            # residue-only mean, not the public representation's layer-35 mean.
            residue_weights = (attention & ~special).to(self.device).unsqueeze(-1).to(hidden.dtype)
            legacy = ((states[34] * residue_weights).sum(dim=1) /
                      residue_weights.sum(dim=1).clamp_min(1)).float().cpu().numpy()
            if not np.isfinite(legacy).all():
                raise ValueError("Invalid property-head embeddings")
        return features, lengths, legacy

    def close(self):
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
