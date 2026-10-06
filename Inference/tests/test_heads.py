import hashlib
import json

import numpy as np
import pytest
import torch

from esmcapsid.heads import PropertyHeads, PropertyNetwork, ScreeningHeads, binary_mlp
from esmcapsid.releases import HEADS_DIRECTORY


def screening_assets(root):
    heads = root / HEADS_DIRECTORY
    config = {}
    for name in ("normal_head", "hardneg_head"):
        directory = heads / name
        directory.mkdir(parents=True)
        network = binary_mlp(4, [3, 2])
        for parameter in network.parameters():
            torch.nn.init.zeros_(parameter)
        network[-1].bias.data[1] = 4
        checkpoint = directory / "layer16_binary_mlp_head.pt"
        torch.save({"input_dim": 4, "hidden_dims": [3, 2], "state_dict": network.state_dict()}, checkpoint)
        np.save(directory / "feature_mean.npy", np.zeros(4, dtype=np.float32))
        np.save(directory / "feature_std.npy", np.ones(4, dtype=np.float32))
        (directory / "export_config.json").write_text(json.dumps({"layer": 16, "hidden_dims": [3, 2]}))
        config[name] = {
            "relative_directory": name, "positive_threshold": 0.97,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    (heads / "two_stage_config.json").write_text(json.dumps(config))
    return heads


def test_actual_binary_head_loading_and_decision(tmp_path):
    screening_assets(tmp_path)
    heads = ScreeningHeads(tmp_path)
    normal, hardneg, passed = heads.predict(np.ones((2, 4), dtype=np.float32))
    np.testing.assert_allclose(normal, 1 / (1 + np.exp(-4)), atol=1e-6)
    np.testing.assert_allclose(hardneg, normal)
    assert passed.tolist() == [True, True]
    assert heads.hidden_index == 17
    heads.thresholds["hardneg_head"] = 0.999
    assert not heads.predict(np.ones((1, 4), dtype=np.float32))[2].any()


def test_release_checkpoint_integrity_is_enforced(tmp_path):
    root = screening_assets(tmp_path)
    (root / "normal_head" / "layer16_binary_mlp_head.pt").write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        ScreeningHeads(tmp_path)


def test_invalid_normalization_is_not_silently_repaired(tmp_path):
    root = screening_assets(tmp_path)
    np.save(root / "normal_head" / "feature_std.npy", np.zeros(4))
    with pytest.raises(ValueError, match="normalization"):
        ScreeningHeads(tmp_path)


def test_property_classes_thresholds_and_unknown_are_preserved(tmp_path):
    directory = tmp_path / "fold_label"
    directory.mkdir()
    network = PropertyNetwork(4, 2, 3)
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    network.layer2.bias.data[1] = 2
    torch.save(network.state_dict(), directory / "best_model.pth")
    config = {"input_dim": 4, "num_classes": 2, "hidden_dim": 3,
              "classes": ["fold_a", "fold_b"], "predict_threshold": 0.99}
    (directory / "config.json").write_text(json.dumps(config))
    heads = PropertyHeads(tmp_path)
    assert heads.predict(np.ones((1, 4), dtype=np.float32))["fold_label"][0][0] == "Unknown"
    config["predict_threshold"] = 0.8
    (directory / "config.json").write_text(json.dumps(config))
    assert PropertyHeads(tmp_path).predict(np.ones((1, 4), dtype=np.float32))["fold_label"][0][0] == "fold_b"
