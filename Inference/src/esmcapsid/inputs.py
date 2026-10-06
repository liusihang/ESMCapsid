from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

PROTEIN_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


@dataclass(frozen=True)
class SequenceRecord:
    index: int
    internal_id: str
    original_id: str
    header: str
    sequence: str
    issue: str = ""
    cleaned_terminal_stop: bool = False


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _fasta_rows(path: Path) -> Iterator[tuple[str, str, str]]:
    with _open_text(path) as handle:
        header = None
        parts: list[str] = []
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith((";", "#")):
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0] if header else "", header, "".join(parts)
                header, parts = line[1:].strip(), []
            elif header is None:
                raise ValueError(f"Invalid FASTA: sequence before header at line {line_number}")
            else:
                parts.append(line)
        if header is not None:
            yield header.split()[0] if header else "", header, "".join(parts)


def _csv_rows(path: Path, sequence_column: str, id_column: str):
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if sequence_column not in fields:
            raise ValueError(f"CSV column {sequence_column!r} is missing; found {fields}")
        if id_column not in fields:
            raise ValueError(f"CSV column {id_column!r} is missing; found {fields}")
        for row in reader:
            original_id = (row[id_column] or "").strip()
            yield original_id, original_id, row[sequence_column] or ""


def read_sequences(path: Path, sequence_column="seq", id_column="prot_id") -> list[SequenceRecord]:
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    source = _csv_rows(path, sequence_column, id_column) if ".csv" in suffixes else _fasta_rows(path)
    records = []
    for index, (original_id, header, raw_sequence) in enumerate(source, 1):
        sequence = "".join(raw_sequence.split()).upper()
        terminal_stop = sequence.endswith("*")
        if terminal_stop:
            sequence = sequence[:-1]
        invalid = sorted(set(sequence) - PROTEIN_LETTERS)
        issue = "empty_sequence" if not sequence else ""
        if invalid:
            issue = "invalid_characters:" + "".join(invalid)
        records.append(SequenceRecord(
            index, f"seq_{index:09d}", original_id, header, sequence, issue, terminal_stop,
        ))
    if not records:
        raise ValueError("Input contains no sequence records")
    return records


def batch_records(records: list[SequenceRecord], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]
