from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import pandas as pd
    import torch
    from experiment_tools.train_dual_stage_low_fpr import build_binary_mlp

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    export_root = Path(args.export_root)
    config = json.loads(
        (export_root / "export_config.json").read_text(encoding="utf-8")
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    def checkpoint_hidden_dims(payload: dict) -> list[int]:
        hidden_dims = payload.get("hidden_dims")
        if hidden_dims:
            return [int(dim) for dim in hidden_dims]
        hidden_dim = payload.get("hidden_dim")
        if hidden_dim is not None:
            return [int(hidden_dim)]
        raise KeyError("checkpoint missing hidden_dims/hidden_dim")

    def load_head(path: Path):
        payload = torch.load(path, map_location=args.device)
        model = build_binary_mlp(
            int(payload["input_dim"]), checkpoint_hidden_dims(payload)
        ).to(args.device)
        state_dict = payload["state_dict"]
        if state_dict and all(str(key).startswith("net.") for key in state_dict):
            state_dict = {str(key)[4:]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict_prob(model, array):
        with torch.no_grad():
            tensor = torch.from_numpy(array.astype(np.float32, copy=False)).to(
                args.device
            )
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    early_layer = int(config["early_layer"])
    late_layer = int(config["late_layer"])
    early_threshold = float(config["stage1_threshold"])
    late_threshold = float(config["stage2_threshold"])

    early_x = np.load(Path(args.embedding_dir) / f"layer_{early_layer}.npy")
    late_x = np.load(Path(args.embedding_dir) / f"layer_{late_layer}.npy")
    metadata_df = pd.read_csv(args.metadata)

    early_mean = np.load(export_root / "models" / "early_mean.npy")
    early_std = np.load(export_root / "models" / "early_std.npy")
    late_mean = np.load(export_root / "models" / "late_mean.npy")
    late_std = np.load(export_root / "models" / "late_std.npy")

    early_x = ((early_x - early_mean) / early_std).astype(np.float32, copy=False)
    late_x = ((late_x - late_mean) / late_std).astype(np.float32, copy=False)

    early_model = load_head(export_root / "models" / "early_head.pt")
    late_model = load_head(export_root / "models" / "late_head.pt")

    early_prob = predict_prob(early_model, early_x)
    late_prob = predict_prob(late_model, late_x)
    early_pred = (early_prob >= early_threshold).astype(int)
    late_pred = (late_prob >= late_threshold).astype(int)
    cascade_pred = ((early_pred == 1) & (late_pred == 1)).astype(int)

    out_df = metadata_df.copy()
    out_df["early_prob"] = early_prob
    out_df["late_prob"] = late_prob
    out_df["early_pred"] = early_pred
    out_df["late_pred"] = late_pred
    out_df["cascade_pred"] = cascade_pred
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
