from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _normalize_seq(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def _resolve_csv_column(
    rows: list[dict[str, str]], expected: str, aliases: list[str]
) -> str:
    if not rows:
        raise ValueError("input CSV has no rows")
    columns = list(rows[0].keys())
    lower_map = {col.lower(): col for col in columns}
    if expected in columns:
        return expected
    if expected.lower() in lower_map:
        return lower_map[expected.lower()]
    for alias in aliases:
        if alias in columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise ValueError(
        f"missing required column '{expected}' in CSV. available={columns}"
    )


def load_csv_records(
    csv_path: Path, seq_col: str = "seq", id_col: str | None = None
) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"input CSV is empty: {csv_path}")
    seq_column = _resolve_csv_column(rows, expected=seq_col, aliases=["sequence"])
    if id_col:
        id_column = _resolve_csv_column(rows, expected=id_col, aliases=[])
    else:
        id_column = _resolve_csv_column(
            rows, expected="prot_id", aliases=["id", "name"]
        )
    out: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        prot_id = str(row.get(id_column, "")).strip() or f"row_{idx}"
        seq = _normalize_seq(row.get(seq_column, ""))
        if not seq:
            continue
        out.append({"prot_id": prot_id, "seq": seq})
    if not out:
        raise ValueError(f"no valid sequences found in CSV: {csv_path}")
    return out


def read_fasta_records(fasta_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if current_id is not None:
                    seq = _normalize_seq("".join(chunks))
                    if seq:
                        records.append({"prot_id": current_id, "seq": seq})
                current_id = text[1:].split()[0]
                chunks = []
                continue
            chunks.append(text)
    if current_id is not None:
        seq = _normalize_seq("".join(chunks))
        if seq:
            records.append({"prot_id": current_id, "seq": seq})
    if not records:
        raise ValueError(f"no valid sequences found in FASTA: {fasta_path}")
    return records


def write_fasta(records: list[dict[str, str]], output_fasta: Path) -> int:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(f">{row['prot_id']}\n{row['seq']}\n")
    return len(records)


def write_metadata(records: list[dict[str, str]], output_csv: Path, label: str) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prot_id", "seq", "label"])
        writer.writeheader()
        for row in records:
            writer.writerow(
                {"prot_id": row["prot_id"], "seq": row["seq"], "label": label}
            )
    return len(records)


def build_token_extract_command(
    python_executable: str,
    script_path: Path,
    metadata_csv: Path,
    model_dir: str,
    output_dir: Path,
    layer: int,
    max_len: int,
    batch_size: int,
    device: str = "cuda",
    hf_cache: str | None = None,
    trust_remote_code: bool = True,
    offline: bool = False,
    bucket_sort: bool = True,
    pin_memory: bool = False,
    dataloader_workers: int = 4,
    dataloader_prefetch: int = 2,
) -> list[str]:
    cmd = [
        python_executable,
        str(script_path),
        "--metadata",
        str(metadata_csv),
        "--output",
        str(output_dir),
        "--model",
        model_dir,
        "--layers",
        str(layer),
        "--max-len",
        str(max_len),
        "--bs",
        str(batch_size),
        "--min-samples-per-class",
        "1",
        "--device",
        device,
    ]
    if hf_cache:
        cmd.extend(["--hf-cache", hf_cache])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if offline:
        cmd.append("--offline")
    if not bucket_sort:
        cmd.append("--no-bucket-sort")
    if pin_memory:
        cmd.append("--pin-memory")
    cmd.extend(["--dataloader-workers", str(max(0, int(dataloader_workers)))])
    cmd.extend(["--dataloader-prefetch", str(max(1, int(dataloader_prefetch)))])
    return cmd


def build_single_pass_extract_command(
    python_executable: str,
    script_path: Path,
    metadata_csv: Path,
    model_dir: str,
    output_dir: Path,
    token_layer: int,
    pooled_layer: int,
    max_len: int,
    batch_size: int,
    device: str = "cuda",
    out_dtype: str = "float16",
    pool_only_layers: list[int] | None = None,
    hf_cache: str | None = None,
    trust_remote_code: bool = True,
    offline: bool = False,
    bucket_sort: bool = True,
    pin_memory: bool = False,
    dataloader_workers: int = 4,
    dataloader_prefetch: int = 2,
) -> list[str]:
    layers = ",".join(
        str(layer) for layer in sorted({int(token_layer), int(pooled_layer)})
    )
    cmd = [
        python_executable,
        str(script_path),
        "--metadata",
        str(metadata_csv),
        "--output",
        str(output_dir),
        "--model",
        model_dir,
        "--layers",
        layers,
        "--max-len",
        str(max_len),
        "--bs",
        str(batch_size),
        "--min-samples-per-class",
        "1",
        "--device",
        device,
        "--out-dtype",
        out_dtype,
    ]
    if pool_only_layers:
        pool_only_csv = ",".join(
            str(layer) for layer in sorted({int(x) for x in pool_only_layers})
        )
        cmd.extend(["--pool-only-layers", pool_only_csv])
    if hf_cache:
        cmd.extend(["--hf-cache", hf_cache])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if offline:
        cmd.append("--offline")
    if not bucket_sort:
        cmd.append("--no-bucket-sort")
    if pin_memory:
        cmd.append("--pin-memory")
    cmd.extend(["--dataloader-workers", str(max(0, int(dataloader_workers)))])
    cmd.extend(["--dataloader-prefetch", str(max(1, int(dataloader_prefetch)))])
    return cmd


def pool_token_layer_to_sequence_mean(
    token_npy: Path,
    offsets_npy: Path,
    output_npy: Path,
) -> tuple[int, int]:
    import numpy as np

    token_embeddings = np.load(token_npy, mmap_mode="r")
    offsets = np.load(offsets_npy)
    n_sequences = max(0, int(len(offsets) - 1))
    if n_sequences == 0:
        raise ValueError(f"no sequence offsets found in {offsets_npy}")
    hidden_dim = int(token_embeddings.shape[1])
    pooled = np.zeros((n_sequences, hidden_dim), dtype=np.float32)
    for idx in range(n_sequences):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        if end <= start:
            continue
        pooled[idx] = token_embeddings[start:end].mean(axis=0, dtype=np.float32)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, pooled.astype(np.float32, copy=False))
    return n_sequences, hidden_dim


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline part 2: single-pass ESMCapsid-C inference emitting layer 28 token embeddings and layer 33 pooled embeddings."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file: Potential Capsid .fasta/.fa/.faa or .csv",
    )
    parser.add_argument(
        "--output-root", required=True, help="Output directory for this stage"
    )
    parser.add_argument(
        "--model-dir", required=True, help="ESMCapsid-C model directory"
    )
    parser.add_argument(
        "--token-extract-script",
        required=True,
        help="Path to gpu_token_embed_extract_for_sae.py",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable for child scripts",
    )
    parser.add_argument("--seq-col", default="seq", help="CSV sequence column")
    parser.add_argument(
        "--id-col", default=None, help="CSV id column (default: prot_id/id/name)"
    )
    parser.add_argument(
        "--dummy-label",
        default="PotentialCapsid",
        help="Synthetic label used for token extraction metadata",
    )
    parser.add_argument(
        "--token-layer", type=int, default=28, help="Token layer for SAE"
    )
    parser.add_argument(
        "--pooled-layer",
        type=int,
        default=33,
        help="Sequence pooled layer for downstream",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=786)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--token-out-dtype",
        default="float16",
        help="Token layer output dtype passed to extractor: float32/float16/bfloat16",
    )
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument(
        "--offline", action="store_true", help="Pass --offline to both extractors"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to both extractors",
    )
    parser.add_argument(
        "--bucket-sort", action="store_true", help="Compatibility flag; length bucketing is enabled by default"
    )
    parser.add_argument(
        "--no-bucket-sort",
        action="store_true",
        help="Disable bucket sort in token extractor",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin memory for HF extractor inputs",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="HF extractor DataLoader worker count",
    )
    parser.add_argument(
        "--dataloader-prefetch",
        type=int,
        default=2,
        help="HF extractor DataLoader prefetch factor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    token_extract_script = Path(args.token_extract_script).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    input_cache = output_root / "input_cache"
    input_cache.mkdir(parents=True, exist_ok=True)
    backend_dir = output_root / "single_pass_backend"
    if backend_dir.exists():
        shutil.rmtree(backend_dir)
    backend_dir.mkdir(parents=True, exist_ok=True)
    sae_dir = output_root / "for_sae_layer28"
    downstream_dir = output_root / "for_downstream_layer33"
    sae_dir.mkdir(parents=True, exist_ok=True)
    downstream_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [suffix.lower() for suffix in input_path.suffixes]
    is_csv = ".csv" in suffixes
    if is_csv:
        records = load_csv_records(input_path, seq_col=args.seq_col, id_col=args.id_col)
    else:
        records = read_fasta_records(input_path)

    normalized_fasta = input_cache / "potential_capsid.normalized.fasta"
    normalized_metadata = input_cache / "potential_capsid.metadata.csv"
    write_fasta(records, normalized_fasta)
    write_metadata(records, normalized_metadata, label=args.dummy_label)

    single_pass_cmd = build_single_pass_extract_command(
        python_executable=args.python_executable,
        script_path=token_extract_script,
        metadata_csv=normalized_metadata,
        model_dir=model_dir,
        output_dir=backend_dir,
        token_layer=args.token_layer,
        pooled_layer=args.pooled_layer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        device=args.device,
        out_dtype=args.token_out_dtype,
        pool_only_layers=[args.pooled_layer],
        hf_cache=args.hf_cache,
        trust_remote_code=args.trust_remote_code,
        offline=args.offline,
        bucket_sort=not bool(args.no_bucket_sort),
        pin_memory=bool(args.pin_memory),
        dataloader_workers=int(max(0, args.dataloader_workers)),
        dataloader_prefetch=int(max(1, args.dataloader_prefetch)),
    )
    _run_command(single_pass_cmd)

    shutil.copy2(
        backend_dir / "sample_metadata_filtered.csv",
        sae_dir / "sample_metadata_filtered.csv",
    )
    shutil.copy2(
        backend_dir / "seq_token_offsets.npy", sae_dir / "seq_token_offsets.npy"
    )
    shutil.copy2(backend_dir / "seq_token_len.npy", sae_dir / "seq_token_len.npy")
    sae_token_layer_dir = sae_dir / "token_layers"
    sae_token_layer_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        backend_dir / "token_layers" / f"layer_{args.token_layer}.tokens.npy",
        sae_token_layer_dir / f"layer_{args.token_layer}.tokens.npy",
    )

    shutil.copy2(
        backend_dir / "sample_metadata_filtered.csv",
        downstream_dir / "sample_metadata_filtered.csv",
    )
    pooled_output_npy = downstream_dir / f"layer_{args.pooled_layer}.npy"
    pooled_precomputed = (
        backend_dir / "pooled_layers" / f"layer_{args.pooled_layer}.npy"
    )
    if pooled_precomputed.exists():
        import numpy as np

        shutil.copy2(pooled_precomputed, pooled_output_npy)
        pooled_arr = np.load(pooled_output_npy, mmap_mode="r")
        if pooled_arr.ndim != 2:
            raise ValueError(f"unexpected pooled embedding shape: {pooled_arr.shape}")
        n_sequences = int(pooled_arr.shape[0])
        hidden_dim = int(pooled_arr.shape[1])
    else:
        pooled_token_npy = (
            backend_dir / "token_layers" / f"layer_{args.pooled_layer}.tokens.npy"
        )
        n_sequences, hidden_dim = pool_token_layer_to_sequence_mean(
            token_npy=pooled_token_npy,
            offsets_npy=backend_dir / "seq_token_offsets.npy",
            output_npy=pooled_output_npy,
        )
    shutil.rmtree(backend_dir)

    summary = {
        "input_path": str(input_path),
        "input_mode": "csv" if is_csv else "fasta",
        "n_sequences": int(len(records)),
        "model_dir": model_dir,
        "token_layer": int(args.token_layer),
        "pooled_layer": int(args.pooled_layer),
        "single_pass": True,
        "normalized_fasta": str(normalized_fasta),
        "normalized_metadata": str(normalized_metadata),
        "sae_output_dir": str(sae_dir),
        "sae_outputs": {
            "metadata": str(sae_dir / "sample_metadata_filtered.csv"),
            "token_offsets": str(sae_dir / "seq_token_offsets.npy"),
            "token_lengths": str(sae_dir / "seq_token_len.npy"),
            "token_embedding": str(
                sae_dir / "token_layers" / f"layer_{args.token_layer}.tokens.npy"
            ),
        },
        "downstream_output_dir": str(downstream_dir),
        "downstream_outputs": {
            "metadata": str(downstream_dir / "sample_metadata_filtered.csv"),
            "embedding": str(pooled_output_npy),
        },
        "pooled_summary": {
            "n_sequences": int(n_sequences),
            "hidden_dim": int(hidden_dim),
        },
    }
    (output_root / "stage2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
