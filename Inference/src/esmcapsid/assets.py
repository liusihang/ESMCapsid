from __future__ import annotations

from pathlib import Path

from .releases import RELEASES


def resolve_model(kind: str, models: Path | None, offline: bool) -> Path:
    if models is not None:
        path = models / f"ESMCapsid-{kind}"
        if not (path / "config.json").is_file():
            raise ValueError(f"Model directory missing: {path}; expected config.json")
        return path
    from huggingface_hub import snapshot_download

    repository, revision = RELEASES[kind]
    return Path(snapshot_download(
        repo_id=repository,
        revision=revision,
        local_files_only=offline,
        allow_patterns=[
            "config.json", "model.safetensors", "model.safetensors.index.json",
            "model-*.safetensors", "modeling_esm_plusplus.py", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "LICENSE", "NOTICE.txt",
            "heads/two_stage_layer16_hardneg/*",
        ],
    ))
