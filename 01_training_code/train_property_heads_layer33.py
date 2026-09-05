#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, random_split


class ProbingHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512, dropout=0.2):
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


class EmbeddingDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.from_numpy(features).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one or more capsid property heads from pooled layer-33 embeddings."
    )
    parser.add_argument("--npy-file", required=True, help="Embedding .npy file")
    parser.add_argument(
        "--mapping-csv", required=True, help="Mapping CSV aligned to the NPY rows"
    )
    parser.add_argument("--label-csv", required=True, help="Normalized labels CSV")
    parser.add_argument("--id-col", default="prot_id")
    parser.add_argument(
        "--label-cols",
        nargs="+",
        required=True,
        help="One or more normalized label columns to train",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where per-head subdirectories will be written",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_merged_labels(mapping_csv, label_csv, id_col):
    mapping_df = pd.read_csv(mapping_csv)
    mapping_df["npy_row_idx"] = np.arange(len(mapping_df))
    labels_df = pd.read_csv(label_csv)
    merged = pd.merge(
        mapping_df[[id_col, "npy_row_idx"]], labels_df, on=id_col, how="inner"
    )
    return merged, len(mapping_df)


def train_one_head(args, label_col, merged_df, X_mmap, task_order):
    label_df = merged_df.dropna(subset=[label_col]).copy()
    label_df = label_df[label_df[label_col].astype(str) != ""]
    label_df = label_df[label_df[label_col].astype(str) != "Unknown"]
    if len(label_df) < 10:
        print(f"[WARN] Skip {label_col}: only {len(label_df)} labeled rows.")
        return

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(label_df[label_col].values)
    valid_indices = label_df["npy_row_idx"].to_numpy()
    X_curr = X_mmap[valid_indices]
    dataset = EmbeddingDataset(X_curr, y_all)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ProbingHead(
        X_curr.shape[1], len(label_encoder.classes_), args.hidden_dim
    ).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(args.output_dir, label_col)
    os.makedirs(save_dir, exist_ok=True)
    best_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_acc = correct / total if total else 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    config = {
        "input_dim": int(X_curr.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "num_classes": int(len(label_encoder.classes_)),
        "classes": label_encoder.classes_.tolist(),
        "label_col": label_col,
        "label_name": label_col,
        "source_label_col": label_col,
        "task_order": task_order[label_col],
        "predict_threshold": 0.85,
        "best_val_acc": float(best_acc),
        "seed": int(args.seed),
    }
    with open(os.path.join(save_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(f"[DONE] {label_col}: best_val_acc={best_acc:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    merged_df, total_rows = load_merged_labels(
        args.mapping_csv, args.label_csv, args.id_col
    )
    X_mmap = np.load(args.npy_file, mmap_mode="r")
    if X_mmap.shape[0] != total_rows:
        raise ValueError(
            f"Row mismatch between mapping CSV ({total_rows}) and NPY ({X_mmap.shape[0]})."
        )

    task_order = {
        "realm": 10,
        "kingdom": 20,
        "phylum": 30,
        "class": 40,
        "order": 50,
        "family": 60,
        "genome_label": 70,
        "host_group": 80,
        "fold_label": 90,
    }
    for label_col in args.label_cols:
        if label_col not in merged_df.columns:
            raise KeyError(f"Column not found in label CSV: {label_col}")
        train_one_head(args, label_col, merged_df, X_mmap, task_order)


if __name__ == "__main__":
    main()
