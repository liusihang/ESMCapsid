#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ProbingHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.layer2(x)


class InferenceDataset(Dataset):
    def __init__(self, npy_mmap):
        self.X = npy_mmap

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx].copy()).float()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict capsid property heads from pooled layer-33 embeddings."
    )
    parser.add_argument(
        "--npy-file", required=True, help="Input embedding matrix (.npy)"
    )
    parser.add_argument(
        "--mapping-csv",
        required=True,
        help="CSV aligned to the NPY rows; first column is used as sequence ID",
    )
    parser.add_argument(
        "--model-parent-dir",
        required=True,
        help="Directory containing one subdirectory per prediction head",
    )
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional global confidence threshold override",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_heads(model_parent_dir, device, global_threshold):
    head_entries = []
    for subdir in sorted(os.listdir(model_parent_dir)):
        model_dir = os.path.join(model_parent_dir, subdir)
        if not os.path.isdir(model_dir):
            continue
        config_path = os.path.join(model_dir, "config.json")
        weight_path = os.path.join(model_dir, "best_model.pth")
        if not (os.path.exists(config_path) and os.path.exists(weight_path)):
            continue
        with open(config_path) as handle:
            config = json.load(handle)
        model = ProbingHead(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"],
            hidden_dim=config["hidden_dim"],
        ).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        label_name = config.get("label_name") or config.get("label_col") or subdir
        head_entries.append(
            {
                "label_name": label_name,
                "task_order": config.get("task_order", 9999),
                "threshold": global_threshold
                if global_threshold is not None
                else config.get("predict_threshold", 0.85),
                "classes": np.asarray(config["classes"], dtype=object),
                "model": model,
                "preds": [],
                "confs": [],
            }
        )
    head_entries.sort(key=lambda item: (item["task_order"], item["label_name"]))
    if not head_entries:
        raise RuntimeError(f"No valid heads found under {model_parent_dir}")
    return head_entries


def main():
    args = parse_args()
    df_map = pd.read_csv(args.mapping_csv)
    id_col = df_map.columns[0]
    ids = df_map[id_col].tolist()
    X_mmap = np.load(args.npy_file, mmap_mode="r")
    if len(ids) != X_mmap.shape[0]:
        raise ValueError(
            f"Row mismatch between mapping CSV ({len(ids)}) and NPY ({X_mmap.shape[0]}). "
            "These files must be generated from the same embedding export."
        )

    heads = load_heads(args.model_parent_dir, args.device, args.threshold)
    dataset = InferenceDataset(X_mmap)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = batch.to(args.device)
            for head in heads:
                logits = head["model"](batch)
                probs = F.softmax(logits, dim=1)
                confs, indices = torch.max(probs, dim=1)
                confs = confs.cpu().numpy()
                indices = indices.cpu().numpy()
                preds = head["classes"][indices].astype(object)
                preds[confs < head["threshold"]] = "Unknown"
                head["preds"].extend(preds.tolist())
                head["confs"].extend(np.round(confs, 4).tolist())

    result = pd.DataFrame({id_col: ids})
    for head in heads:
        result[f"{head['label_name']}_pred"] = head["preds"]
        result[f"{head['label_name']}_conf"] = head["confs"]
    result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
