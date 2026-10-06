"""Small real-weight regression; reads, but never edits, the published source.

Run on HPC with existing models. This is an engineering check, not a biological
benchmark. Results stay in the separate inference project.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from esmcapsid.assets import resolve_model
from esmcapsid.encoder import Encoder
from esmcapsid.heads import PropertyHeads, ScreeningHeads
from esmcapsid.inputs import read_sequences
from esmcapsid.releases import HEADS_DIRECTORY, RELEASES


def reference_functions(path: Path, names: set[str]) -> dict:
    source = ast.parse(path.read_text())
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError(f"Reference functions missing: {path}")
    namespace = {"torch": torch, "np": np, "tqdm": tqdm}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-code", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="Existing local model parent; no downloads")
    parser.add_argument("--property-heads", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    records = [record for record in read_sequences(args.input) if not record.issue]
    torch.set_num_threads(4)
    result = {"mode": "existing HPC weights, CPU, small engineering regression",
              "sequences": len(records), "models": RELEASES, "checks": {}}
    started = time.monotonic()

    # One-off acceptance check: the existing HPC mirrors must be the same public
    # weights, not an older checkpoint. No hash manifest is used by the runtime.
    expected_weights = {
        "S": "d4eb98695850c766c7275c5600eb4fc538fa30a2895d19b6a513c69ed1ee00ed",
        "C": "0ae042f930cbd0dd796be93debabca6b8c602a97534c608e68fb70f6fd90f125",
    }
    for kind, expected_hash in expected_weights.items():
        digest = hashlib.sha256()
        with (resolve_model(kind, args.models, True) / "model.safetensors").open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Existing {kind} weights differ from the public release; do not substitute silently")
    result["checks"]["existing_weights_match_public_release"] = {"pass": True}

    s_path = resolve_model("S", args.models, True)
    encoder = Encoder(s_path, "cpu")
    heads = ScreeningHeads(s_path)
    actual, _, _ = encoder.encode(records, hidden_index=heads.hidden_index, max_tokens=1022)
    old = reference_functions(args.reference_code / "05_inference_pipeline_code/extract_sequence_embeddings.py", {
        "_batch_iterator_by_length", "_masked_mean_pool", "pool_hidden_states",
        "_infer_hf_layers_and_dim", "layer_embeddings_hf_batched",
    })
    expected = old["layer_embeddings_hf_batched"](
        encoder.model, encoder.model.tokenizer, [record.sequence for record in records], [16],
        device="cpu", max_len=1022, batch_size=2,
    )[16]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    result["checks"]["s_embeddings_match_published_extractor"] = {
        "pass": True, "max_abs_error": float(np.max(np.abs(actual - expected))),
    }
    build = reference_functions(args.reference_code / "experiment_tools/train_dual_stage_low_fpr.py", {
        "normalize_hidden_dims", "build_binary_mlp",
    })["build_binary_mlp"]
    released = json.loads((s_path / HEADS_DIRECTORY / "two_stage_config.json").read_text())
    reference_scores = []
    with torch.inference_mode():
        for name in ("normal_head", "hardneg_head"):
            directory = s_path / HEADS_DIRECTORY / released[name]["relative_directory"]
            payload = torch.load(directory / "layer16_binary_mlp_head.pt", map_location="cpu", weights_only=True)
            model = build(payload["input_dim"], payload["hidden_dims"])
            state = payload["state_dict"]
            if all(key.startswith("net.") for key in state):
                state = {key[4:]: value for key, value in state.items()}
            model.load_state_dict(state)
            model.eval()
            mean = np.load(directory / "feature_mean.npy").astype(np.float32).reshape(-1)
            std = np.load(directory / "feature_std.npy").astype(np.float32).reshape(-1)
            reference_scores.append(torch.softmax(model(torch.from_numpy(((expected - mean) / std).astype(np.float32))), dim=1)[:, 1].numpy())
    normal, hardneg, decisions = heads.predict(actual)
    np.testing.assert_allclose(normal, reference_scores[0], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(hardneg, reference_scores[1], rtol=1e-5, atol=1e-7)
    reference_decisions = ((reference_scores[0] >= released["normal_head"]["positive_threshold"]) &
                           (reference_scores[1] >= released["hardneg_head"]["positive_threshold"]))
    np.testing.assert_array_equal(decisions, reference_decisions)
    result["checks"]["s_scores_and_decisions_match_published_logic"] = {
        "pass": True, "normal_scores": normal.tolist(), "hardneg_scores": hardneg.tolist(),
        "decisions": decisions.astype(int).tolist(),
    }
    encoder.close()

    c_path = resolve_model("C", args.models, True)
    encoder = Encoder(c_path, "cpu")
    actual, lengths, legacy = encoder.encode(records, hidden_index=35, max_tokens=786, property_features=True)
    encoded = encoder.tokenizer(
        [record.sequence for record in records], padding=True, truncation=True, max_length=786,
        add_special_tokens=True, return_tensors="pt", return_special_tokens_mask=True,
    )
    special = encoded.pop("special_tokens_mask").bool()
    with torch.inference_mode():
        output = encoder.model(**encoded, output_hidden_states=True)
        assert len(output.hidden_states) == 36
        mask = encoded["attention_mask"].unsqueeze(-1).to(torch.float32)
        expected = ((output.hidden_states[35] * mask).sum(dim=1) / mask.sum(dim=1)).float().numpy()
        residue_mask = (encoded["attention_mask"].bool() & ~special).unsqueeze(-1).to(torch.float32)
        expected_legacy = ((output.hidden_states[34] * residue_mask).sum(dim=1) /
                           residue_mask.sum(dim=1)).float().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(legacy, expected_legacy, rtol=1e-5, atol=1e-5)
    result["checks"]["c_public_and_legacy_representations"] = {
        "pass": True, "shape": list(actual.shape), "processed_lengths": lengths,
        "returned_block_states": 36, "public_max_abs_error": float(np.max(np.abs(actual - expected))),
        "legacy_max_abs_error": float(np.max(np.abs(legacy - expected_legacy))),
        "real_property_heads_validated": False,
    }
    if args.property_heads:
        properties = PropertyHeads(args.property_heads)
        actual_properties = properties.predict(legacy)
        source = ast.parse((args.reference_code / "05_inference_pipeline_code/predict_property_heads_layer33.py").read_text())
        reference_class = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ProbingHead"]
        if len(reference_class) != 1:
            raise ValueError("Reference property network is missing")
        namespace = {"nn": torch.nn, "F": torch.nn.functional}
        exec(compile(ast.Module(body=reference_class, type_ignores=[]), "reference_property.py", "exec"), namespace)
        for directory in sorted(args.property_heads.iterdir()):
            if not directory.is_dir():
                continue
            config = json.loads((directory / "config.json").read_text())
            name = config.get("label_name") or config.get("label_col") or directory.name
            network = namespace["ProbingHead"](config["input_dim"], config["num_classes"], config["hidden_dim"])
            network.load_state_dict(torch.load(directory / "best_model.pth", map_location="cpu", weights_only=True))
            network.eval()
            with torch.inference_mode():
                confidence, indices = torch.softmax(network(torch.from_numpy(legacy)), dim=1).max(dim=1)
            expected_labels = [str(config["classes"][index]) if score >= config.get("predict_threshold", 0.85) else "Unknown"
                               for index, score in zip(indices.tolist(), confidence.tolist())]
            assert [label for label, score in actual_properties[name]] == expected_labels
            np.testing.assert_allclose([score for label, score in actual_properties[name]], confidence.numpy(), rtol=1e-5, atol=1e-7)
        result["checks"]["property_heads_match_published_network_and_decisions"] = {
            "pass": True, "tasks": list(actual_properties),
        }
        result["checks"]["c_public_and_legacy_representations"]["real_property_heads_validated"] = True
    encoder.close()
    result["status"] = "PASS"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
