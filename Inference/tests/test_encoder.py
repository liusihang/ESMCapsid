from types import SimpleNamespace

import numpy as np
import pytest
import torch

from esmcapsid.encoder import Encoder
from esmcapsid.inputs import SequenceRecord


class Tokenizer:
    def __call__(self, *args, **kwargs):
        return {
            "input_ids": torch.tensor([[0, 3, 4, 2, 1], [0, 5, 2, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1], [1, 0, 1, 1, 1]]),
        }


class Model:
    def __call__(self, **kwargs):
        assert set(kwargs) == {"input_ids", "attention_mask", "output_hidden_states"}
        values = torch.tensor([[[10., 10.], [2., 2.], [4., 4.], [20., 20.], [999., 999.]],
                               [[10., 10.], [6., 6.], [20., 20.], [999., 999.], [999., 999.]]])
        return SimpleNamespace(hidden_states=tuple(values + index for index in range(36)))


def encoder():
    result = Encoder.__new__(Encoder)
    result.model = Model()
    result.tokenizer = Tokenizer()
    result.device = "cpu"
    result.dimension = 2
    return result


def test_public_mean_and_legacy_residue_mean_have_different_masks():
    records = [SequenceRecord(1, "1", "a", "a", "AC"), SequenceRecord(2, "2", "b", "b", "M")]
    features, lengths, legacy = encoder().encode(records, hidden_index=35, max_tokens=5, property_features=True)
    assert lengths == [2, 1]
    np.testing.assert_allclose(features[:, 0], [9 + 35, 12 + 35])
    np.testing.assert_allclose(legacy[:, 0], [3 + 34, 6 + 34])


def test_out_of_range_layer_is_not_silently_clamped():
    records = [SequenceRecord(1, "1", "a", "a", "AC"), SequenceRecord(2, "2", "b", "b", "M")]
    with pytest.raises(ValueError, match="available: 36"):
        encoder().encode(records, hidden_index=36, max_tokens=5)
