#!/usr/bin/env python3
"""Run the FASTA pipeline end to end with deterministic mock backends."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PIPELINE_ROOT = Path(__file__).resolve().parent
PUBLIC_ENTRYPOINT = PIPELINE_ROOT.parent / "predict_capsid_semantic_motifs.py"

BACKEND_SOURCE = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_embed() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()

    rows = read_rows(args.metadata)
    output = args.output
    token_dir = output / "token_layers"
    token_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    offsets = []
    lengths = []
    token_blocks = []
    offset = 0
    for row in rows:
        sequence = row["seq"]
        length = len(sequence)
        offsets.append(offset)
        lengths.append(length)
        metadata_rows.append(
            {
                **row,
                "tok_start": offset,
                "tok_end": offset + length,
            }
        )
        token_blocks.append(
            np.arange(length * 4, dtype=np.float32).reshape(length, 4) + offset
        )
        offset += length

    write_rows(
        output / "sample_metadata_filtered.csv",
        metadata_rows,
        ["prot_id", "seq", "label", "tok_start", "tok_end"],
    )
    np.save(output / "seq_token_offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(output / "seq_token_len.npy", np.asarray(lengths, dtype=np.int64))
    np.save(
        token_dir / "layer_28.tokens.npy",
        np.concatenate(token_blocks, axis=0),
    )


def run_sae() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--dims", type=int, required=True)
    parser.add_argument("--ks", type=int, required=True)
    args, _ = parser.parse_known_args()

    preprocess = json.loads(
        (args.data_dir / "preprocess_config.json").read_text(encoding="utf-8")
    )
    token_features = np.load(preprocess["input_tokens_path"])
    metadata_rows = read_rows(Path(preprocess["input_meta_path"]))
    sequence_index = np.concatenate(
        [
            np.full(
                int(row["tok_end"]) - int(row["tok_start"]),
                index,
                dtype=np.int64,
            )
            for index, row in enumerate(metadata_rows)
        ]
    )
    output = args.output_root / f"SAE_Tok_D{args.dims}_K{args.ks}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "sae_feats_token_csr.npz",
        features=np.abs(token_features[:, :4]),
    )
    np.savez_compressed(
        output / "token_to_seq_mapping.npz",
        sequence_index=sequence_index,
    )


def run_map() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query_npz", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    args, _ = parser.parse_known_args()

    features = np.load(args.query_npz)["features"]
    labels = np.arange(features.shape[0], dtype=np.int64) % 3
    confidence = np.full(features.shape[0], 0.9, dtype=np.float32)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_prefix.with_name(args.output_prefix.name + "_clusters.npy"),
        labels,
    )
    np.save(
        args.output_prefix.with_name(args.output_prefix.name + "_confidence.npy"),
        confidence,
    )
    args.output_prefix.with_name(
        args.output_prefix.name + "_summary.json"
    ).write_text(
        json.dumps({"query_count": int(features.shape[0])}) + "\n",
        encoding="utf-8",
    )


def run_pool() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--token_npz", type=Path, required=True)
    parser.add_argument("--mapping_npz", type=Path, required=True)
    parser.add_argument("--out_npz", type=Path, required=True)
    args, _ = parser.parse_known_args()

    features = np.load(args.token_npz)["features"]
    sequence_index = np.load(args.mapping_npz)["sequence_index"]
    sequence_features = np.stack(
        [
            features[sequence_index == index].mean(axis=0)
            for index in sorted(set(sequence_index.tolist()))
        ]
    )
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, features=sequence_features)


def run_predict() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query_npz", type=Path, required=True)
    parser.add_argument("--query_ids_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args, _ = parser.parse_known_args()

    rows = read_rows(args.query_ids_csv)
    predictions = [
        {
            "Seq_ID": row["prot_id"],
            "Pred_Label": f"mock_cluster_{index % 2}",
            "Confidence": "0.95",
        }
        for index, row in enumerate(rows)
    ]
    write_rows(
        args.out_dir / f"{args.prefix}_predictions.csv",
        predictions,
        ["Seq_ID", "Pred_Label", "Confidence"],
    )
    (args.out_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps({"query_count": len(predictions)}) + "\n",
        encoding="utf-8",
    )


DISPATCH = {
    "fake_embed": run_embed,
    "fake_sae": run_sae,
    "fake_map": run_map,
    "fake_pool": run_pool,
    "fake_predict": run_predict,
}


if __name__ == "__main__":
    DISPATCH[Path(__file__).stem]()
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def write_config(root: Path, backend_paths: dict[str, Path]) -> Path:
    references = root / "references"
    references.mkdir()
    reference_paths = {
        "semantic_motif_ref_npz": references / "semantic_motif_ref.npz",
        "semantic_motif_labels": references / "semantic_motif_labels.npy",
        "semantic_motif_index": references / "semantic_motif.index",
        "sequence_knn_ref_npz": references / "sequence_knn_ref.npz",
        "sequence_knn_ref_labels_csv": references / "sequence_knn_labels.csv",
    }
    for path in reference_paths.values():
        path.write_bytes(b"smoke\n")

    pretrained_model = references / "sae_model.pt"
    global_stats = references / "global_stats.npz"
    pretrained_model.write_bytes(b"smoke model\n")
    global_stats.write_bytes(b"smoke stats\n")

    config = {
        "pipeline": {
            "max_sequences_per_chunk": 2,
            "max_total_aa_per_chunk": 1000,
            "dummy_label": "__smoke__",
        },
        "runtime": {"python_executable": sys.executable},
        "reference": {
            **{key: str(path) for key, path in reference_paths.items()},
            "sequence_knn_label_col": "Cluster",
        },
        "backends": {
            "embed": {
                "python_executable": sys.executable,
                "script_path": str(backend_paths["fake_embed"]),
                "model": "mock_model",
                "layers": "28",
                "max_len": 786,
                "batch_size": 2,
                "device": "cpu",
            },
            "sae": {
                "python_executable": sys.executable,
                "script_path": str(backend_paths["fake_sae"]),
                "token_file_name": "layer_28.tokens.npy",
                "dims": 4,
                "ks": 2,
                "input_hidden_dim": 4,
                "infer_chunk_size": 16,
                "use_amp": False,
                "use_torch_compile": False,
                "do_mean_center": False,
                "do_rescale": False,
                "clip_extreme_values": False,
                "pretrained_model_path": str(pretrained_model),
                "global_stats_path": str(global_stats),
            },
            "map_semantic_motif": {
                "python_executable": sys.executable,
                "script_path": str(backend_paths["fake_map"]),
                "k": 2,
                "nprobe": 1,
                "gpu_id": 0,
                "use_float16": False,
                "add_batch_size": 100,
                "query_batch_size": 100,
            },
            "pool_seq": {
                "python_executable": sys.executable,
                "script_path": str(backend_paths["fake_pool"]),
            },
            "predict_seq": {
                "python_executable": sys.executable,
                "script_path": str(backend_paths["fake_predict"]),
                "query_id_col": "prot_id",
                "k": 2,
                "backend": "cpu",
                "query_batch": 100,
                "cpu_threads": 1,
                "include_noise_in_vote": True,
            },
        },
    }
    config_path = root / "smoke_config.json"
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def main() -> int:
    result_payload: dict[str, object]
    temp_path: Path
    with tempfile.TemporaryDirectory(prefix="vicapsid_pipeline_smoke_") as temp_dir:
        temp_path = Path(temp_dir)
        backends = temp_path / "backends"
        backends.mkdir()
        backend_paths = {}
        for name in [
            "fake_embed",
            "fake_sae",
            "fake_map",
            "fake_pool",
            "fake_predict",
        ]:
            path = backends / f"{name}.py"
            path.write_text(BACKEND_SOURCE, encoding="utf-8")
            backend_paths[name] = path

        fasta_path = temp_path / "smoke.fasta"
        fasta_path.write_text(
            ">alpha first record\nACD-EF?\n"
            ">duplicate second record\nMNPQ\n"
            ">duplicate third record\nAAAA\n",
            encoding="utf-8",
        )
        config_path = write_config(temp_path, backend_paths)
        run_dir = temp_path / "run"
        command = [
            sys.executable,
            str(PUBLIC_ENTRYPOINT),
            "--fasta",
            str(fasta_path),
            "--run-dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--parts",
            "all",
            "--map-backend",
            "cpu",
        ]
        first_run = run_command(command, PIPELINE_ROOT)

        report_dir = run_dir / "60_report"
        pipeline_report_path = report_dir / "pipeline_report.json"
        final_predictions_path = report_dir / "final_sequence_predictions.csv"
        final_tokens_path = report_dir / "final_token_semantic_motif.parquet"
        pipeline_report = json.loads(pipeline_report_path.read_text(encoding="utf-8"))
        with final_predictions_path.open("r", encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        token_table = pq.read_table(final_tokens_path)

        expected_ids = {"alpha", "duplicate", "duplicate__dup002"}
        actual_ids = {row["Seq_ID"] for row in predictions}
        if actual_ids != expected_ids:
            raise AssertionError(f"Unexpected sequence IDs: {actual_ids}")
        if pipeline_report["chunk_count"] != 2:
            raise AssertionError(pipeline_report)
        if pipeline_report["sequence_count"] != 3:
            raise AssertionError(pipeline_report)
        if pipeline_report["token_annotation_count"] != 15:
            raise AssertionError(pipeline_report)
        if token_table.num_rows != 15:
            raise AssertionError(f"Unexpected Parquet rows: {token_table.num_rows}")

        output_hashes = {
            "pipeline_report": sha256(pipeline_report_path),
            "final_predictions": sha256(final_predictions_path),
            "final_tokens": sha256(final_tokens_path),
        }
        second_run = run_command(command, PIPELINE_ROOT)
        resumed_hashes = {
            "pipeline_report": sha256(pipeline_report_path),
            "final_predictions": sha256(final_predictions_path),
            "final_tokens": sha256(final_tokens_path),
        }
        if resumed_hashes != output_hashes:
            raise AssertionError("Resume run changed completed outputs.")

        result_payload = {
            "status": "PASS",
            "mode": "deterministic mock backends",
            "entrypoint": PUBLIC_ENTRYPOINT.name,
            "pipeline_parts": [
                "prepare",
                "embed",
                "sae",
                "map_semantic_motif",
                "pool_seq",
                "predict_seq",
                "report",
            ],
            "first_run_exit_code": first_run.returncode,
            "resume_run_exit_code": second_run.returncode,
            "chunk_count": pipeline_report["chunk_count"],
            "sequence_count": pipeline_report["sequence_count"],
            "token_annotation_count": pipeline_report["token_annotation_count"],
            "final_parquet_rows": token_table.num_rows,
            "resume_outputs_unchanged": True,
            "temporary_workspace_cleaned": True,
            "real_model_inference_validated": False,
        }

    if temp_path.exists():
        raise AssertionError(f"Temporary smoke directory remains: {temp_path}")
    print(json.dumps(result_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
