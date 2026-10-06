import csv
import json
from pathlib import Path

import numpy as np
import pytest

import esmcapsid.pipeline as pipeline
from esmcapsid.pipeline import RunOptions, run


class FakeEncoder:
    dimension = 4
    live = []
    calls = []

    def __init__(self, directory, device):
        assert not self.live, "The previous encoder must be released before the next one is loaded"
        self.kind = directory.name
        self.live.append(self.kind)

    def encode(self, records, *, hidden_index, max_tokens, property_features=False):
        self.calls.append((self.kind, hidden_index, property_features))
        features = np.array([[record.index, 2, 3, 4] for record in records], dtype=np.float32)
        lengths = [min(len(record.sequence), max_tokens - 2) for record in records]
        legacy = np.full_like(features, -99) if property_features else None
        return features, lengths, legacy

    def close(self):
        self.live.clear()


class FakeScreening:
    hidden_index = 17
    thresholds = {"normal_head": 0.99, "hardneg_head": 0.9675}

    def __init__(self, path):
        pass

    def predict(self, features):
        passed = features[:, 0] % 2 == 1
        return np.where(passed, 0.999, 0.1), np.where(passed, 0.99, 0.1), passed


class NoCandidates(FakeScreening):
    def predict(self, features):
        return np.zeros(len(features)), np.zeros(len(features)), np.zeros(len(features), dtype=bool)


class FakeProperties:
    heads = [("fold_label",)]

    def __init__(self, path):
        pass

    def predict(self, features):
        assert (features == -99).all(), "Property heads must not receive public layer-35 embeddings"
        return {"fold_label": [("Unknown", 0.6)] * len(features)}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    FakeEncoder.live.clear()
    FakeEncoder.calls.clear()
    monkeypatch.setattr(pipeline, "resolve_model", lambda kind, models, offline: Path(f"ESMCapsid-{kind}"))


def input_file(tmp_path):
    path = tmp_path / "input.faa"
    path.write_text(">A original first\nACDEFGHIK\n>A\nMNP\n>A__dup002\nAAA\n>bad\nA-C\n")
    return path


def result_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def execute(options, screening=FakeScreening):
    return run(options, encoder_factory=FakeEncoder, screening_factory=screening,
               property_factory=FakeProperties)


def test_predict_keeps_all_inputs_and_encodes_candidates_only(tmp_path):
    output = tmp_path / "run"
    summary = execute(RunOptions(input_file(tmp_path), output, device="cpu", batch_size=1,
                                 c_max_tokens=5, save_embeddings=True))
    rows = result_rows(output / "predictions.tsv")
    assert summary["candidate_count"] == 2
    assert summary["embedding_count"] == 2
    assert len(rows) == 4
    assert [row["original_id"] for row in rows] == ["A", "A", "A__dup002", "bad"]
    assert rows[1]["capsid_pred"] == "0"
    assert rows[1]["embedding_status"] == "not_applicable"
    assert rows[3]["input_status"] == "invalid"
    assert rows[0]["c_processed_aa"] == "3"
    assert rows[0]["c_truncated"] == "1"
    assert rows[0]["property_status"] == "heads_not_supplied"
    features = np.load(output / "c_embeddings.npy")
    assert features[:, 0].tolist() == [1, 3]
    assert np.load(output / "s_embeddings.npy").shape == (3, 4)
    assert "ACDEFGHIK" in (output / "capsid_candidates.faa").read_text()
    assert summary["truncated_count"] == 1
    assert all(call[1] == 35 for call in FakeEncoder.calls if call[0].endswith("C"))


def test_screen_only_never_loads_c(tmp_path):
    output = tmp_path / "run"
    summary = execute(RunOptions(input_file(tmp_path), output, device="cpu", screen_only=True))
    assert not (output / "c_embeddings.npy").exists()
    assert all(call[0].endswith("S") for call in FakeEncoder.calls)
    assert summary["candidate_count"] == 2


def test_zero_candidates_is_a_success_not_a_model_failure(tmp_path):
    output = tmp_path / "run"
    summary = execute(RunOptions(input_file(tmp_path), output, device="cpu"), NoCandidates)
    assert summary["status"] == "completed"
    assert summary["candidate_count"] == 0
    assert np.load(output / "c_embeddings.npy").shape == (0, 1152)
    assert all(call[0].endswith("S") for call in FakeEncoder.calls)
    assert (output / "capsid_candidates.faa").read_text() == ""


def test_annotate_bypasses_screen_and_uses_distinct_property_features(tmp_path):
    output = tmp_path / "run"
    execute(RunOptions(input_file(tmp_path), output, mode="annotate", device="cpu",
                       property_heads=tmp_path / "heads"))
    rows = result_rows(output / "predictions.tsv")
    assert rows[0]["capsid_pred"] == ""
    assert rows[0]["screen_status"] == "not_run"
    assert rows[0]["fold_label_pred"] == "Unknown"
    assert rows[0]["property_status"] == "completed"
    assert all(call[0].endswith("C") and call[2] for call in FakeEncoder.calls)


def test_embed_is_not_misreported_as_capsid_or_property_prediction(tmp_path):
    output = tmp_path / "run"
    summary = execute(RunOptions(input_file(tmp_path), output, mode="embed", device="cpu"))
    assert summary["embedding_count"] == 3
    assert not (output / "capsid_candidates.faa").exists()
    assert all(not row["capsid_pred"] for row in result_rows(output / "predictions.tsv"))


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user content")
    with pytest.raises(ValueError, match="already exists"):
        execute(RunOptions(input_file(tmp_path), output, device="cpu"))
    assert sentinel.read_text() == "user content"


def test_annotate_without_heads_is_rejected_before_any_output(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="no property heads"):
        execute(RunOptions(input_file(tmp_path), output, mode="annotate", device="cpu"))
    assert not output.exists()


def test_failure_has_failed_receipt_and_releases_encoder(tmp_path):
    class Broken(FakeEncoder):
        def encode(self, *args, **kwargs):
            raise RuntimeError("model failure")

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="model failure"):
        run(RunOptions(input_file(tmp_path), output, device="cpu"),
            encoder_factory=Broken, screening_factory=FakeScreening)
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert not FakeEncoder.live
    assert "model failure" in (output / "run.log").read_text()
