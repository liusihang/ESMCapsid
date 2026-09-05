from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_final_head_layer(heads_root: Path) -> int:
    config = json.loads(
        (heads_root / "two_stage_config.json").read_text(encoding="utf-8")
    )
    layers = set()
    for head_name in ("normal_head", "hardneg_head"):
        head_dir = heads_root / str(config[head_name]["relative_directory"])
        export_config = json.loads(
            (head_dir / "export_config.json").read_text(encoding="utf-8")
        )
        layers.add(int(export_config["layer"]))
    if len(layers) != 1:
        raise ValueError(f"final heads must share one layer, found {sorted(layers)}")
    return layers.pop()


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


def convert_csv_to_fasta(
    csv_path: Path, output_fasta: Path, seq_col: str = "seq", id_col: str | None = None
) -> int:
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"input CSV is empty: {csv_path}")
    seq_column = _resolve_csv_column(records, expected=seq_col, aliases=["sequence"])
    if id_col:
        id_column = _resolve_csv_column(records, expected=id_col, aliases=[])
    else:
        id_column = _resolve_csv_column(
            records, expected="prot_id", aliases=["id", "name"]
        )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_fasta.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(records):
            prot_id = str(row.get(id_column, "")).strip()
            if not prot_id:
                prot_id = f"row_{idx}"
            seq = _normalize_seq(row.get(seq_column, ""))
            if not seq:
                continue
            handle.write(f">{prot_id}\n{seq}\n")
            written += 1
    if written == 0:
        raise ValueError(f"no valid sequences found in CSV: {csv_path}")
    return written


def read_fasta_map(path: Path) -> dict[str, str]:
    seq_map: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if current_id is not None:
                    seq_map[current_id] = _normalize_seq("".join(chunks))
                current_id = text[1:].split()[0]
                chunks = []
                continue
            chunks.append(text)
    if current_id is not None:
        seq_map[current_id] = _normalize_seq("".join(chunks))
    return {key: value for key, value in seq_map.items() if value}


def build_extract_command(
    python_executable: str,
    extract_script: Path,
    input_fasta: Path,
    model_dir: str,
    output_dir: Path,
    early_layer: int,
    late_layer: int,
    max_len: int,
    batch_size: int,
    trust_remote_code: bool = True,
    offline: bool = True,
    bucket_sort: bool = True,
) -> list[str]:
    layers = sorted({int(early_layer), int(late_layer)})
    cmd = [
        python_executable,
        str(extract_script),
        "--fasta",
        str(input_fasta),
        "--model",
        model_dir,
        "--layers",
        ",".join(str(layer) for layer in layers),
        "--max-len",
        str(max_len),
        "--bs",
        str(batch_size),
        "--output",
        str(output_dir),
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if offline:
        cmd.append("--offline")
    if bucket_sort:
        cmd.append("--bucket-sort")
    return cmd


def build_predict_command(
    python_executable: str,
    predict_script: Path,
    embedding_dir: Path,
    metadata_csv: Path,
    heads_root: Path,
    output_csv: Path,
    device: str = "cuda",
) -> list[str]:
    return [
        python_executable,
        str(predict_script),
        "--embedding-dir",
        str(embedding_dir),
        "--metadata",
        str(metadata_csv),
        "--heads-root",
        str(heads_root),
        "--output",
        str(output_csv),
        "--device",
        device,
    ]


def _run_command(cmd: list[str]):
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline part 1: FASTA/CSV -> ESMCapsid-S Layer-16 final two-head prediction."
    )
    parser.add_argument(
        "--input", required=True, help="Input file: .fasta/.fa/.faa(.gz) or .csv"
    )
    parser.add_argument(
        "--output-root", required=True, help="Output directory for this stage"
    )
    parser.add_argument(
        "--model-dir", required=True, help="ESMCapsid-S model directory"
    )
    parser.add_argument(
        "--heads-root",
        required=True,
        help="Final classifier package containing two_stage_config.json.",
    )
    parser.add_argument(
        "--extract-script",
        default=str(Path(__file__).with_name("extract_sequence_embeddings.py")),
        help="Path to the sequence-embedding extractor.",
    )
    parser.add_argument(
        "--predict-script",
        default=str(Path(__file__).with_name("predict_layer16_two_head.py")),
        help="Path to predict_layer16_two_head.py",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--capsid-column",
        default="capsid_pred",
        help="Prediction column interpreted as capsid flag",
    )
    parser.add_argument(
        "--capsid-threshold", type=float, default=0.5, help="Threshold on capsid-column"
    )
    parser.add_argument(
        "--no-offline",
        action="store_true",
        help="Do not pass --offline to extract script",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Do not pass --trust-remote-code to extract script",
    )
    parser.add_argument(
        "--no-bucket-sort",
        action="store_true",
        help="Do not pass --bucket-sort to extract script",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    heads_root = Path(args.heads_root).expanduser().resolve()
    extract_script = Path(args.extract_script).expanduser().resolve()
    predict_script = Path(args.predict_script).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    embedding_dir = output_root / "embeddings_dual_stage"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    input_cache_dir = output_root / "input_cache"
    input_cache_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [s.lower() for s in input_path.suffixes]
    is_csv = ".csv" in suffixes
    if is_csv:
        input_fasta = input_cache_dir / "input.from_csv.fasta"
        n_input = convert_csv_to_fasta(
            csv_path=input_path,
            output_fasta=input_fasta,
            seq_col=args.seq_col,
            id_col=args.id_col,
        )
    else:
        input_fasta = input_path
        n_input = len(read_fasta_map(input_fasta))

    layer = load_final_head_layer(heads_root)
    extract_cmd = build_extract_command(
        python_executable=args.python_executable,
        extract_script=extract_script,
        input_fasta=input_fasta,
        model_dir=model_dir,
        output_dir=embedding_dir,
        early_layer=layer,
        late_layer=layer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        trust_remote_code=not args.no_trust_remote_code,
        offline=not args.no_offline,
        bucket_sort=not args.no_bucket_sort,
    )
    _run_command(extract_cmd)

    metadata_csv = embedding_dir / "sample_metadata_filtered.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"missing metadata generated by extractor: {metadata_csv}"
        )

    prediction_csv = output_root / "capsid_predictions_full.csv"
    predict_cmd = build_predict_command(
        python_executable=args.python_executable,
        predict_script=predict_script,
        embedding_dir=embedding_dir,
        metadata_csv=metadata_csv,
        heads_root=heads_root,
        output_csv=prediction_csv,
        device=args.device,
    )
    _run_command(predict_cmd)

    pred_df = pd.read_csv(prediction_csv)
    if args.capsid_column not in pred_df.columns:
        raise ValueError(f"capsid-column not found: {args.capsid_column}")
    capsid_df = pred_df[
        pred_df[args.capsid_column].astype(float) >= args.capsid_threshold
    ].copy()
    capsid_df = capsid_df.sort_values(by=args.capsid_column, ascending=False)
    capsid_csv = output_root / "potential_capsid_predictions.csv"
    capsid_df.to_csv(capsid_csv, index=False)

    seq_map = read_fasta_map(input_fasta)
    capsid_fasta = output_root / "potential_capsid.fasta"
    with capsid_fasta.open("w", encoding="utf-8") as handle:
        for prot_id in capsid_df["prot_id"].astype(str).tolist():
            seq = seq_map.get(prot_id)
            if not seq:
                continue
            handle.write(f">{prot_id}\n{seq}\n")

    summary = {
        "input_path": str(input_path),
        "input_mode": "csv" if is_csv else "fasta",
        "n_input_sequences": int(n_input),
        "model_dir": model_dir,
        "heads_root": str(heads_root),
        "layer": int(layer),
        "capsid_column": args.capsid_column,
        "capsid_threshold": float(args.capsid_threshold),
        "n_capsid_predicted": int(len(capsid_df)),
        "embedding_dir": str(embedding_dir),
        "full_prediction_csv": str(prediction_csv),
        "capsid_prediction_csv": str(capsid_csv),
        "capsid_fasta": str(capsid_fasta),
    }
    summary_path = output_root / "stage1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
