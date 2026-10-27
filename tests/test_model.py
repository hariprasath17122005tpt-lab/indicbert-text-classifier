"""Tests for model checkpoint loading."""

import torch
from safetensors.torch import save_file

from src.model import _load_safetensors_weights_into_model


def test_load_safetensors_into_meta_model_without_copying_weights(tmp_path):
    reference_model = torch.nn.Linear(3, 2)
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(
        {
            name: tensor.detach().contiguous()
            for name, tensor in reference_model.state_dict().items()
        },
        str(checkpoint_path),
    )

    with torch.device("meta"):
        model = torch.nn.Linear(3, 2)

    loaded_model = _load_safetensors_weights_into_model(model, checkpoint_path)
    input_tensor = torch.tensor([[1.0, 2.0, 3.0]])

    assert torch.allclose(loaded_model(input_tensor), reference_model(input_tensor))
    assert all(not parameter.is_meta for parameter in loaded_model.parameters())
