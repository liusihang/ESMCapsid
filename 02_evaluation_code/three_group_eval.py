import argparse
import csv
import importlib.util
import json
import statistics
from pathlib import Path

import pandas as pd

GROUPS = [
    ("capsid", "Capsid", {"Capsid"}),
    ("non_capsid_virus", "Non-capsid virus", {"NoCapsidVirus"}),
    (
        "others_cellular",
        "Others / Cellular proteins",
        {"Structure", "Other", "Cellular"},
    ),
]

MODEL_SPECS = [
    (
        "/path/to/project_data/db/Model/Synthyra/ESMplusplus_large",
        "ESMplusplus_large",
        8,
    ),
    ("/path/to/project_data/db/Model/Synthyra/FastESM2_650M", "FastESM2_650M", 8),
    (
        "/path/to/project_data/db/Model/Synthyra/Profluent-E1-600M",
        "Profluent-E1-600M",
        8,
    ),
    ("/path/to/project_data/db/Model/esm3o", "esm3o", 4),
    (
        "/path/to/vicapsid_data/esmc_600m_MLMfinetuned_shuffle",
        "esmc_600m_MLMfinetuned_shuffle",
        8,
    ),
]


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("mlm_evaluation_special_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_group_sequences(
    df: pd.DataFrame, sequence_col: str, seed: int
) -> dict[str, dict]:
    out = {}
    seen_labels = set(df["label"].unique().tolist())
    unmapped = sorted(seen_labels - set().union(*[labels for _, _, labels in GROUPS]))
    if unmapped:
        raise ValueError(f"Unmapped labels found: {unmapped}")
    for group_key, display_name, labels in GROUPS:
        group_df = df[df["label"].isin(labels)].copy()
        n = len(group_df)
        sampled = (
            group_df[sequence_col].sample(n=n, random_state=seed).tolist() if n else []
        )
        out[group_key] = {
            "group_display": display_name,
            "source_labels": sorted(labels),
            "n_sequences": n,
            "sequences": sampled,
            "observed_label_counts": {
                k: int(v) for k, v in group_df["label"].value_counts().to_dict().items()
            },
        }
    return out


def evaluate_one_seed(
    base_module,
    model_path: str,
    model_slug: str,
    model_batch_size: int,
    sampled_csv: Path,
    seed: int,
    output_json: Path,
    device: str,
    mask_ratio: float,
    sequence_col: str,
):
    base_module.set_all_seeds(seed)
    df = pd.read_csv(sampled_csv)
    grouped = sample_group_sequences(df, sequence_col=sequence_col, seed=seed)
    model, tokenizer, model_kind = base_module.load_model_bundle(
        model_path,
        hf_cache=None,
        offline=True,
        device=device,
        torch_dtype=base_module.torch.float16 if device == "cuda" else None,
    )
    result = {
        "model": model_path,
        "model_slug": model_slug,
        "model_kind": model_kind,
        "sampled_csv": str(sampled_csv),
        "seed": seed,
        "mask_ratio": mask_ratio,
        "group_order": [g[0] for g in GROUPS],
        "groups": {},
    }
    for group_key, _, _ in GROUPS:
        payload = grouped[group_key]
        seqs = payload.pop("sequences")
        if model_kind == "e1":
            metrics = base_module.compute_mlm_metrics_e1(
                model, seqs, device, mask_ratio, model_batch_size
            )
        else:
            metrics = base_module.compute_mlm_metrics_hf_or_esm3(
                model, tokenizer, model_kind, seqs, device, mask_ratio, model_batch_size
            )
        result["groups"][group_key] = {**payload, **metrics}
    output_json.write_text(json.dumps(result, indent=2) + "\n")


def aggregate_results(root: Path):
    long_rows = []
    for path in sorted(root.glob("seed_*/results/*.json")):
        data = json.loads(path.read_text())
        for group_key in data["group_order"]:
            group = data["groups"][group_key]
            long_rows.append(
                {
                    "seed": data["seed"],
                    "model": data["model_slug"],
                    "model_kind": data["model_kind"],
                    "model_path": data["model"],
                    "sampled_csv": data["sampled_csv"],
                    "mask_ratio": data["mask_ratio"],
                    "group_key": group_key,
                    "group_display": group["group_display"],
                    "source_labels": ";".join(group["source_labels"]),
                    "n_sequences": group["n_sequences"],
                    "observed_label_counts_json": json.dumps(
                        group["observed_label_counts"], sort_keys=True
                    ),
                    "accuracy": group["accuracy"],
                    "perplexity": group["perplexity"],
                    "avg_loss": group["avg_loss"],
                    "total_masked_tokens": group["total_masked_tokens"],
                    "total_correct": group["total_correct"],
                }
            )
    repeat_long_csv = root / "repeat_long_3category.csv"
    repeat_long_json = root / "repeat_long_3category.json"
    repeat_long_json.write_text(json.dumps(long_rows, indent=2) + "\n")
    if long_rows:
        with repeat_long_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys()))
            writer.writeheader()
            writer.writerows(long_rows)

    summary_rows = []
    grouped = {}
    for row in long_rows:
        grouped.setdefault((row["model"], row["group_key"]), []).append(row)
    for (model, group_key), rows in sorted(grouped.items()):
        acc_vals = [float(r["accuracy"]) for r in rows]
        ppl_vals = [float(r["perplexity"]) for r in rows]
        loss_vals = [float(r["avg_loss"]) for r in rows]
        n_vals = [int(r["n_sequences"]) for r in rows]
        summary_rows.append(
            {
                "model": model,
                "model_kind": rows[0]["model_kind"],
                "model_path": rows[0]["model_path"],
                "group_key": group_key,
                "group_display": rows[0]["group_display"],
                "source_labels": rows[0]["source_labels"],
                "n_repeats": len(rows),
                "n_sequences_mean": sum(n_vals) / len(n_vals),
                "n_sequences_values": ";".join(str(v) for v in n_vals),
                "accuracy_mean": sum(acc_vals) / len(acc_vals),
                "accuracy_std": statistics.stdev(acc_vals)
                if len(acc_vals) > 1
                else 0.0,
                "perplexity_mean": sum(ppl_vals) / len(ppl_vals),
                "perplexity_std": statistics.stdev(ppl_vals)
                if len(ppl_vals) > 1
                else 0.0,
                "avg_loss_mean": sum(loss_vals) / len(loss_vals),
                "avg_loss_std": statistics.stdev(loss_vals)
                if len(loss_vals) > 1
                else 0.0,
            }
        )
    summary_csv = root / "aggregate_summary_3category.csv"
    summary_json = root / "aggregate_summary_3category.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2) + "\n")
    if summary_rows:
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", required=True)
    parser.add_argument("--base-repeat-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--sequence-col", default="seq")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 121, 1220])
    args = parser.parse_args()

    base_module = load_base_module(Path(args.base_script))
    out_root = Path(args.out_root)
    base_repeat_root = Path(args.base_repeat_root)
    for seed in args.seeds:
        seed_root = out_root / f"seed_{seed}"
        result_root = seed_root / "results"
        result_root.mkdir(parents=True, exist_ok=True)
        sampled_csv = (
            base_repeat_root / f"seed_{seed}" / "sampled_capsid2500_other5000.csv"
        )
        for model_path, model_slug, batch_size in MODEL_SPECS:
            output_json = result_root / f"{model_slug}.json"
            evaluate_one_seed(
                base_module=base_module,
                model_path=model_path,
                model_slug=model_slug,
                model_batch_size=batch_size,
                sampled_csv=sampled_csv,
                seed=seed,
                output_json=output_json,
                device=args.device,
                mask_ratio=args.mask_ratio,
                sequence_col=args.sequence_col,
            )
    aggregate_results(out_root)


if __name__ == "__main__":
    main()
