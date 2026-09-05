from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import PipelineError

PART_REQUIREMENTS = {
    "embed": (
        ("backends", "embed", "python_executable"),
        ("backends", "embed", "script_path"),
        ("backends", "embed", "model"),
    ),
    "sae": (
        ("backends", "sae", "python_executable"),
        ("backends", "sae", "script_path"),
        ("backends", "sae", "pretrained_model_path"),
        ("backends", "sae", "global_stats_path"),
    ),
    "map_semantic_motif": (
        ("backends", "map_semantic_motif", "python_executable"),
        ("backends", "map_semantic_motif", "script_path"),
        ("reference", "semantic_motif_ref_npz"),
        ("reference", "semantic_motif_labels"),
        ("reference", "semantic_motif_index"),
    ),
    "pool_seq": (
        ("backends", "pool_seq", "python_executable"),
        ("backends", "pool_seq", "script_path"),
    ),
    "predict_seq": (
        ("backends", "predict_seq", "python_executable"),
        ("backends", "predict_seq", "script_path"),
        ("reference", "sequence_knn_ref_npz"),
        ("reference", "sequence_knn_ref_labels_csv"),
    ),
}


def _get_value(config: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _is_placeholder(value: str) -> bool:
    return value.startswith("/path/to/") or "<" in value or ">" in value


def _path_status(value: str, executable: bool) -> tuple[bool, str]:
    if _is_placeholder(value):
        return False, "placeholder"
    path = Path(value).expanduser()
    if executable and not path.is_absolute():
        resolved = shutil.which(value)
        return (resolved is not None, resolved or "not found on PATH")
    return (path.exists(), "found" if path.exists() else "not found")


def validate_config(
    config: dict[str, Any], selected_parts: list[str]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    required = {
        requirement
        for part in selected_parts
        for requirement in PART_REQUIREMENTS.get(part, ())
    }
    for keys in sorted(required):
        value = _get_value(config, keys)
        label = ".".join(keys)
        if not isinstance(value, str) or not value:
            checks.append(
                {"key": label, "value": value, "ok": False, "status": "missing"}
            )
            errors.append(f"{label}: missing")
            continue
        if keys[-1] == "model" and not _is_placeholder(value):
            model_path = Path(value).expanduser()
            offline = bool(_get_value(config, keys[:-1] + ("offline",)))
            if model_path.exists():
                checks.append(
                    {"key": label, "value": value, "ok": True, "status": "found"}
                )
                continue
            if not offline and not model_path.is_absolute():
                checks.append(
                    {
                        "key": label,
                        "value": value,
                        "ok": True,
                        "status": "external model identifier",
                    }
                )
                continue
        executable = keys[-1] == "python_executable"
        ok, status = _path_status(value, executable=executable)
        checks.append({"key": label, "value": value, "ok": ok, "status": status})
        if not ok:
            errors.append(f"{label}: {status} ({value})")
    if errors:
        raise PipelineError("Preflight failed:\n- " + "\n- ".join(errors))
    return checks
