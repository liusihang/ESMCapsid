from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import PART_ORDER

CHUNK_STAGE_DIRS = {
    "prepare": "00_input",
    "embed": "10_embed",
    "sae": "20_sae",
    "map_semantic_motif": "30_semantic_motif",
    "pool_seq": "40_seq",
    "predict_seq": "50_seq_pred",
}


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkContext:
    run_dir: Path
    chunk_id: str

    @property
    def chunk_dir(self) -> Path:
        return self.run_dir / "chunks" / self.chunk_id

    def stage_dir(self, part_name: str) -> Path:
        return self.chunk_dir / CHUNK_STAGE_DIRS[part_name]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_relative_config_paths(config: dict[str, Any], config_path: Path) -> None:
    base_dir = config_path.parent
    reference_cfg = config.get("reference", {})
    for key, value in reference_cfg.items():
        if isinstance(value, str) and value.startswith("."):
            reference_cfg[key] = str((base_dir / value).resolve())

    backend_path_keys = {
        "script_path",
        "model",
        "hf_cache",
        "pretrained_model_path",
        "global_stats_path",
    }
    for backend_cfg in config.get("backends", {}).values():
        for key in backend_path_keys:
            value = backend_cfg.get(key)
            if isinstance(value, str) and value.startswith("."):
                backend_cfg[key] = str((base_dir / value).resolve())


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_part_selection(part_arg: str) -> list[str]:
    if part_arg == "all":
        return PART_ORDER.copy()

    selected = []
    seen = set()
    for raw in part_arg.split(","):
        part = raw.strip()
        if not part:
            continue
        if part not in PART_ORDER:
            raise PipelineError(f"Unknown part: {part}")
        if part not in seen:
            seen.add(part)
            selected.append(part)
    if not selected:
        raise PipelineError("No parts selected.")
    return selected


def normalize_seq(seq: str) -> str:
    compact = "".join(str(seq).upper().split())
    return "".join(ch if "A" <= ch <= "Z" else "X" for ch in compact)


def parse_fasta(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise PipelineError(f"FASTA not found: {path}")

    records: list[dict[str, str]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(";") or line.startswith("#"):
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append(
                        {"header": current_header, "sequence": "".join(current_lines)}
                    )
                current_header = line[1:].strip()
                current_lines = []
                continue
            if current_header is None:
                raise PipelineError("Invalid FASTA: sequence line before header.")
            current_lines.append(line)

    if current_header is not None:
        records.append({"header": current_header, "sequence": "".join(current_lines)})

    if not records:
        raise PipelineError(f"No FASTA records found in {path}")
    return records


def safe_seq_id(header: str, index: int) -> str:
    token = (header.split()[0] if header else "").strip()
    token = token or f"seq_{index:06d}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in token)


def dedupe_seq_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: dict[str, int] = {}
    deduped: list[dict[str, Any]] = []
    for item in records:
        base = item["Seq_ID"]
        counter.setdefault(base, 0)
        counter[base] += 1
        seq_id = base if counter[base] == 1 else f"{base}__dup{counter[base]:03d}"
        updated = dict(item)
        updated["Seq_ID"] = seq_id
        updated["Seq_ID_Base"] = base
        deduped.append(updated)
    return deduped


def chunk_records(
    records: list[dict[str, Any]],
    max_sequences_per_chunk: int,
    max_total_aa_per_chunk: int,
) -> list[list[dict[str, Any]]]:
    if max_sequences_per_chunk < 1:
        raise PipelineError("max_sequences_per_chunk must be >= 1")
    if max_total_aa_per_chunk < 1:
        raise PipelineError("max_total_aa_per_chunk must be >= 1")

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_aa = 0
    for record in records:
        aa_len = int(record["aa_len"])
        would_exceed_count = len(current) >= max_sequences_per_chunk
        would_exceed_aa = current and current_aa + aa_len > max_total_aa_per_chunk
        if would_exceed_count or would_exceed_aa:
            chunks.append(current)
            current = []
            current_aa = 0
        current.append(record)
        current_aa += aa_len
    if current:
        chunks.append(current)
    return chunks


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    ensure_dir(path.parent)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {
            fieldname: [row.get(fieldname) for row in rows] for fieldname in fieldnames
        }
        table = pa.table(columns)
        pq.write_table(table, path)
        return
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd

        frame = pd.DataFrame(rows, columns=fieldnames)
        frame.to_parquet(path, index=False)
        return
    except Exception as exc:
        raise PipelineError(
            "Parquet output requires pyarrow or pandas with a parquet engine installed."
        ) from exc


def write_fasta(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record['Seq_ID']}\n")
            sequence = str(record["seq"])
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def resolve_chunk_ids(run_dir: Path, explicit_chunk_id: str | None = None) -> list[str]:
    manifest_path = run_dir / "manifest" / "chunk_manifest.csv"
    if not manifest_path.exists():
        raise PipelineError(f"Chunk manifest not found: {manifest_path}")

    rows: list[dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    all_chunk_ids = [row["chunk_id"] for row in rows]
    if explicit_chunk_id is None:
        return all_chunk_ids

    normalized = explicit_chunk_id.strip()
    if normalized.isdigit():
        normalized = f"chunk_{int(normalized):05d}"
    if normalized not in all_chunk_ids:
        raise PipelineError(f"Unknown chunk_id: {explicit_chunk_id}")
    return [normalized]


def part_command_path(stage_dir: Path, summary_name: str) -> Path:
    if summary_name.endswith(".summary.json"):
        return stage_dir / summary_name.replace(".summary.json", ".command.sh")
    if summary_name.endswith(".json"):
        return stage_dir / (summary_name[:-5] + ".command.sh")
    return stage_dir / f"{summary_name}.command.sh"


def log_dir(stage_dir: Path) -> Path:
    return ensure_dir(stage_dir / "logs")


def is_complete(summary_path: Path, required_outputs: list[Path]) -> bool:
    return summary_path.exists() and all(path.exists() for path in required_outputs)


def write_command(stage_dir: Path, summary_name: str, command: list[str]) -> None:
    command_path = part_command_path(stage_dir, summary_name)
    ensure_dir(command_path.parent)
    shell_line = " ".join(shlex_quote(token) for token in command)
    command_path.write_text(shell_line + "\n", encoding="utf-8")


def shlex_quote(token: str) -> str:
    import shlex

    return shlex.quote(str(token))


def run_command(
    command: list[str],
    stage_dir: Path,
    summary_name: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    ensure_dir(stage_dir)
    write_command(stage_dir, summary_name, command)
    logs = log_dir(stage_dir)
    stdout_path = logs / f"{summary_name[:-5]}.stdout.log"
    stderr_path = logs / f"{summary_name[:-5]}.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    if process.returncode != 0:
        raise PipelineError(
            f"Command failed ({process.returncode}): {' '.join(command)}\n"
            f"See logs: {stdout_path} and {stderr_path}"
        )


def copy_or_symlink(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def canonical_python_executable(
    config: dict[str, Any],
    backend_cfg: dict[str, Any] | None = None,
) -> str:
    if backend_cfg and backend_cfg.get("python_executable"):
        return str(backend_cfg["python_executable"])
    return str(config.get("runtime", {}).get("python_executable", sys.executable))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_part_summary(
    summary_path: Path,
    part_name: str,
    payload: dict[str, Any],
) -> None:
    data = {
        "part": part_name,
        "completed_at": utc_now(),
        **payload,
    }
    save_json(summary_path, data)
