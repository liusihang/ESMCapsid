from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .assets import resolve_model
from .encoder import Encoder, choose_device
from .heads import PropertyHeads, ScreeningHeads
from .inputs import batch_records, read_sequences


@dataclass
class RunOptions:
    input: Path
    out: Path
    mode: str = "predict"
    models: Path | None = None
    property_heads: Path | None = None
    offline: bool = False
    device: str = "auto"
    batch_size: int = 8
    screen_only: bool = False
    save_embeddings: bool = False
    sequence_column: str = "seq"
    id_column: str = "prot_id"
    screen_max_tokens: int = 1022
    c_max_tokens: int = 786


BASE_FIELDS = [
    "record_index", "internal_id", "original_id", "original_header", "aa_length",
    "terminal_stop_removed", "input_status", "input_issue", "screen_status",
    "screen_processed_aa", "screen_truncated", "normal_score", "hardneg_score", "capsid_pred",
    "embedding_status", "c_processed_aa", "c_truncated", "property_status", "property_processed_aa",
]


def _save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_predictions(path: Path, rows: list[dict], task_names: list[str]):
    fields = BASE_FIELDS + [field for name in task_names for field in (f"{name}_pred", f"{name}_score")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_candidates(path: Path, records, selected_ids: set[str]):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.internal_id not in selected_ids:
                continue
            # Unique output IDs, original full headers preserved in the result table.
            handle.write(f">{record.internal_id} {record.header}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(record.sequence[start:start + 80] + "\n")


def _embedding_mapping(path: Path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["embedding_row", "internal_id", "original_id", "record_index"])
        for index, record in enumerate(records):
            writer.writerow([index, record.internal_id, record.original_id, record.index])


def run(options: RunOptions, *, encoder_factory=Encoder, screening_factory=ScreeningHeads,
        property_factory=PropertyHeads) -> dict:
    if options.mode not in {"predict", "annotate", "embed"}:
        raise ValueError(f"Unknown mode: {options.mode}")
    if options.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if min(options.screen_max_tokens, options.c_max_tokens) < 3:
        raise ValueError("Token limits must be at least 3")
    if options.screen_only and options.mode != "predict":
        raise ValueError("--screen-only is available only for predict")
    if options.screen_only and options.property_heads:
        raise ValueError("--screen-only cannot be combined with --property-heads")
    if options.mode == "annotate" and options.property_heads is None:
        raise ValueError("annotate requires --property-heads pointing to a local head directory. "
                         "The nine public heads are available from Shuofang127/ESMCapsid-C "
                         "under heads/property_heads; use embed for representations without heads.")
    if options.mode == "embed" and options.property_heads:
        raise ValueError("Use annotate, not embed, to run property heads")
    records = read_sequences(options.input, options.sequence_column, options.id_column)
    valid = [record for record in records if not record.issue]
    if not valid:
        raise ValueError("No valid protein sequences; inspect empty sequences or invalid characters")
    device = choose_device(options.device)
    output = options.out.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"Output directory already exists: {output}; choose a new directory") from exc

    logger = logging.getLogger(f"esmcapsid.run.{id(options)}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    started = time.monotonic()
    summary = {
        "version": __version__, "status": "running", "mode": options.mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input": str(options.input.resolve()), "device": device,
        "batch_size": options.batch_size, "screen_only": options.screen_only,
        "input_count": len(records), "valid_count": len(valid),
        "invalid_count": len(records) - len(valid), "models": {},
        "screen_max_tokens": options.screen_max_tokens, "c_max_tokens": options.c_max_tokens,
        "property_heads": str(options.property_heads.resolve()) if options.property_heads else None,
        "warnings": [],
    }
    _save_json(output / "run.json", summary)
    rows = [{
        **dict.fromkeys(BASE_FIELDS, ""),
        "record_index": record.index, "internal_id": record.internal_id,
        "original_id": record.original_id, "original_header": record.header,
        "aa_length": len(record.sequence), "terminal_stop_removed": int(record.cleaned_terminal_stop),
        "input_status": "invalid" if record.issue else "valid", "input_issue": record.issue,
        "screen_status": "invalid_input" if record.issue else "not_run",
        "embedding_status": "invalid_input" if record.issue else "not_run",
        "property_status": "invalid_input" if record.issue else "not_requested",
    } for record in records]
    by_id = {row["internal_id"]: row for row in rows}
    encoder = None
    tasks = []
    try:
        properties = property_factory(options.property_heads) if options.property_heads else None
        tasks = [head[0] for head in properties.heads] if properties else []
        for row in rows:
            for task in tasks:
                row[f"{task}_pred"] = ""
                row[f"{task}_score"] = ""
        selected = valid
        if options.mode == "predict":
            logger.info("Resolving the public ESMCapsid-S model and screening heads")
            model_path = resolve_model("S", options.models, options.offline)
            summary["models"]["S"] = str(model_path)
            heads = screening_factory(model_path)
            summary["screen_thresholds"] = heads.thresholds
            encoder = encoder_factory(model_path, device)
            saved_s = None
            if options.save_embeddings:
                saved_s = np.lib.format.open_memmap(
                    output / "s_embeddings.npy", mode="w+", dtype=np.float32,
                    shape=(len(valid), encoder.dimension),
                )
                _embedding_mapping(output / "s_embedding_ids.tsv", valid)
            selected = []
            offset = 0
            for batch in batch_records(valid, options.batch_size):
                features, lengths, _ = encoder.encode(
                    batch, hidden_index=heads.hidden_index, max_tokens=options.screen_max_tokens,
                )
                normal, hardneg, passed = heads.predict(features)
                for record, length, normal_score, hardneg_score, candidate in zip(batch, lengths, normal, hardneg, passed):
                    by_id[record.internal_id].update({
                        "screen_status": "completed", "screen_processed_aa": length,
                        "screen_truncated": int(length < len(record.sequence)),
                        "normal_score": float(normal_score), "hardneg_score": float(hardneg_score),
                        "capsid_pred": int(candidate),
                        "embedding_status": "not_requested" if options.screen_only else "not_applicable",
                        "property_status": "not_requested" if not properties else "not_applicable",
                    })
                    if candidate:
                        selected.append(record)
                if saved_s is not None:
                    saved_s[offset:offset + len(batch)] = features
                offset += len(batch)
                logger.info("Screened %d/%d valid sequences", offset, len(valid))
            if saved_s is not None:
                saved_s.flush()
                del saved_s
            encoder.close()
            encoder = None
            summary["candidate_count"] = len(selected)
            _write_candidates(output / "capsid_candidates.faa", records, {r.internal_id for r in selected})
            _write_predictions(output / "predictions.tsv", rows, tasks)
            _save_json(output / "run.json", summary)
            logger.info("S screening completed: %d candidates", len(selected))

        if not options.screen_only:
            summary["embedding_count"] = len(selected)
            if not properties:
                summary["warnings"].append("No property heads supplied: C outputs are representations, not property annotations.")
            _embedding_mapping(output / "c_embedding_ids.tsv", selected)
            if selected:
                logger.info("Resolving the public ESMCapsid-C encoder")
                model_path = resolve_model("C", options.models, options.offline)
                summary["models"]["C"] = str(model_path)
                encoder = encoder_factory(model_path, device)
                saved_c = np.lib.format.open_memmap(
                    output / "c_embeddings.npy", mode="w+", dtype=np.float32,
                    shape=(len(selected), encoder.dimension),
                )
                offset = 0
                for batch in batch_records(selected, options.batch_size):
                    # Released custom code returns 36 block states without an
                    # initial embedding state. Block 35 is index 35, not 36.
                    features, lengths, legacy = encoder.encode(
                        batch, hidden_index=35, max_tokens=options.c_max_tokens,
                        property_features=properties is not None,
                    )
                    predictions = properties.predict(legacy) if properties else {}
                    for index, (record, length) in enumerate(zip(batch, lengths)):
                        row = by_id[record.internal_id]
                        row.update({
                            "embedding_status": "completed", "c_processed_aa": length,
                            "c_truncated": int(length < len(record.sequence)),
                            "property_status": "completed" if properties else "heads_not_supplied",
                            "property_processed_aa": length if properties else "",
                        })
                        for task, values in predictions.items():
                            row[f"{task}_pred"], row[f"{task}_score"] = values[index]
                    saved_c[offset:offset + len(batch)] = features
                    offset += len(batch)
                    logger.info("Encoded %d/%d selected sequences", offset, len(selected))
                saved_c.flush()
                del saved_c
                encoder.close()
                encoder = None
            else:
                # Public encoder dimension, no need to load a 600M model for an empty result.
                np.save(output / "c_embeddings.npy", np.empty((0, 1152), dtype=np.float32))

        _write_predictions(output / "predictions.tsv", rows, tasks)
        summary.update({
            "status": "completed", "elapsed_seconds": round(time.monotonic() - started, 3),
            "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
        })
        truncations = sum(bool(row["screen_truncated"] or row["c_truncated"]) for row in rows)
        if truncations:
            summary["warnings"].append(f"{truncations} sequences were truncated; see predictions.tsv for processed lengths.")
        summary["truncated_count"] = truncations
        _save_json(output / "run.json", summary)
        logger.info("Run completed")
        return summary
    except Exception as exc:
        summary.update({"status": "failed", "error": str(exc),
                        "elapsed_seconds": round(time.monotonic() - started, 3)})
        _save_json(output / "run.json", summary)
        logger.exception("Run failed")
        raise
    finally:
        if encoder is not None:
            encoder.close()
        logger.removeHandler(handler)
        handler.close()
