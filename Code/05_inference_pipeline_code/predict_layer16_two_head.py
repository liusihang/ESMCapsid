from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def checkpoint_hidden_dims(payload: dict, export_config: dict) -> list[int]:
    hidden_dims = payload.get("hidden_dims") or export_config.get("hidden_dims")
    if hidden_dims:
        return [int(dim) for dim in hidden_dims]
    hidden_dim = payload.get("hidden_dim") or export_config.get("hidden_dim")
    if hidden_dim is not None:
        return [int(hidden_dim)]
    raise KeyError("checkpoint missing hidden_dims/hidden_dim")


def normalize_state_dict(state_dict: dict) -> dict:
    if state_dict and all(str(key).startswith("net.") for key in state_dict):
        return {str(key)[4:]: value for key, value in state_dict.items()}
    return state_dict


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch

    from experiment_tools.train_dual_stage_low_fpr import build_binary_mlp

    parser = argparse.ArgumentParser(
        description="Predict capsids with the final Layer-16 normal and hard-negative heads."
    )
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--heads-root",
        required=True,
        help="Directory containing two_stage_config.json, normal_head/, and hardneg_head/.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    heads_root = Path(args.heads_root)
    top_config = json.loads(
        (heads_root / "two_stage_config.json").read_text(encoding="utf-8")
    )

    head_specs: list[tuple[str, Path, float, str]] = []
    layers: set[int] = set()
    for head_name in ("normal_head", "hardneg_head"):
        head_meta = top_config[head_name]
        head_dir = heads_root / str(head_meta["relative_directory"])
        export_config = json.loads(
            (head_dir / "export_config.json").read_text(encoding="utf-8")
        )
        layers.add(int(export_config["layer"]))
        head_specs.append(
            (
                head_name,
                head_dir,
                float(head_meta["positive_threshold"]),
                str(head_meta["checkpoint_sha256"]),
            )
        )
    if len(layers) != 1:
        raise ValueError(
            f"final heads must use one shared layer, found {sorted(layers)}"
        )
    layer = layers.pop()

    embeddings = np.load(Path(args.embedding_dir) / f"layer_{layer}.npy", mmap_mode="r")
    metadata_df = pd.read_csv(args.metadata)
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, found {embeddings.shape}")
    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError(
            f"row mismatch: metadata={len(metadata_df)}, embeddings={embeddings.shape[0]}"
        )

    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for head_name, head_dir, threshold, expected_hash in head_specs:
        checkpoint = head_dir / "layer16_binary_mlp_head.pt"
        actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{head_name} checkpoint hash mismatch: {actual_hash} != {expected_hash}"
            )

        export_config = json.loads(
            (head_dir / "export_config.json").read_text(encoding="utf-8")
        )
        payload = torch.load(checkpoint, map_location=args.device)
        model = build_binary_mlp(
            int(payload["input_dim"]),
            checkpoint_hidden_dims(payload, export_config),
        ).to(args.device)
        model.load_state_dict(normalize_state_dict(payload["state_dict"]))
        model.eval()

        mean = (
            np.load(head_dir / "feature_mean.npy")
            .astype(np.float32, copy=False)
            .reshape(-1)
        )
        std = (
            np.load(head_dir / "feature_std.npy")
            .astype(np.float32, copy=False)
            .reshape(-1)
        )
        if embeddings.shape[1] != mean.shape[0] or mean.shape != std.shape:
            raise ValueError(
                f"{head_name} normalization shape mismatch: "
                f"embedding_dim={embeddings.shape[1]}, mean={mean.shape}, std={std.shape}"
            )

        probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, embeddings.shape[0], args.batch_size):
                batch = np.asarray(
                    embeddings[start : start + args.batch_size], dtype=np.float32
                )
                batch = ((batch - mean) / std).astype(np.float32, copy=False)
                logits = model(torch.from_numpy(batch).to(args.device))
                probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        head_scores = np.concatenate(probabilities) if probabilities else np.array([])
        if not np.isfinite(head_scores).all():
            raise ValueError(f"{head_name} produced non-finite probabilities")
        scores[head_name] = head_scores
        thresholds[head_name] = threshold

    output_df = metadata_df.copy()
    output_df["normal_score"] = scores["normal_head"]
    output_df["hardneg_score"] = scores["hardneg_head"]
    output_df["normal_pass"] = (
        output_df["normal_score"] >= thresholds["normal_head"]
    ).astype(int)
    output_df["hardneg_pass"] = (
        output_df["hardneg_score"] >= thresholds["hardneg_head"]
    ).astype(int)
    output_df["capsid_pred"] = (
        (output_df["normal_pass"] == 1) & (output_df["hardneg_pass"] == 1)
    ).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "layer": layer,
                "sequence_count": int(len(output_df)),
                "predicted_capsid_count": int(output_df["capsid_pred"].sum()),
                "normal_threshold": thresholds["normal_head"],
                "hardneg_threshold": thresholds["hardneg_head"],
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
