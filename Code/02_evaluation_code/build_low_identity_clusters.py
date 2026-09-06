from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path


def write_capsid_fasta(metadata_csv: Path, fasta_path: Path):
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        open(metadata_csv, newline="", encoding="utf-8") as handle,
        open(fasta_path, "w", encoding="utf-8") as out,
    ):
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("label") != "Capsid":
                continue
            prot_id = str(row["prot_id"]).strip()
            seq = str(row["seq"]).strip()
            if not prot_id or not seq:
                continue
            out.write(f">{prot_id}\n{seq}\n")
            count += 1
    return count


def run_mmseqs(
    mmseqs_bin: str,
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        "0.8",
        "--cov-mode",
        "1",
    ]
    subprocess.run(cmd, check=True)


def convert_cluster_tsv(cluster_tsv: Path, mapping_csv: Path, summary_json: Path):
    cluster_sizes = Counter()
    rows = []
    with open(cluster_tsv, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for representative, member in reader:
            cluster_sizes[representative] += 1
            rows.append({"prot_id": member, "cluster_id": representative})

    with open(mapping_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prot_id", "cluster_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    cluster_size_values = list(cluster_sizes.values())
    summary = {
        "n_clusters": len(cluster_sizes),
        "n_members": len(rows),
        "max_cluster_size": max(cluster_size_values) if cluster_size_values else 0,
        "min_cluster_size": min(cluster_size_values) if cluster_size_values else 0,
        "top10_cluster_sizes": cluster_sizes.most_common(10),
    }
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--min-seq-id", type=float, default=0.3)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    fasta_path = output_root / "capsid_sequences.fasta"
    output_prefix = (
        output_root / f"capsid_mmseqs_{str(args.min_seq_id).replace('.', 'p')}"
    )
    tmp_dir = output_root / "tmp"
    mapping_csv = output_root / "capsid_cluster_map.csv"
    summary_json = output_root / "cluster_summary.json"

    n_capsid = write_capsid_fasta(Path(args.metadata_csv), fasta_path)
    run_mmseqs(args.mmseqs_bin, fasta_path, output_prefix, tmp_dir, args.min_seq_id)
    convert_cluster_tsv(Path(f"{output_prefix}_cluster.tsv"), mapping_csv, summary_json)

    print(
        json.dumps(
            {
                "n_capsid": n_capsid,
                "fasta": str(fasta_path),
                "cluster_map": str(mapping_csv),
                "summary": str(summary_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
