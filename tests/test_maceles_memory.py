"""Optional MACE integration; run with PYTHONPATH=src:<maceles-mh checkout>."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("mace")
from mace.tools.torch_tools import default_dtype
from tests.golden.build_maceles_anchor import build_model
from tests.extensions.les.les_fixtures import atoms, batch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_mixed_maceles_force_training_matches_original_scripted_solver(dtype, device):
    with default_dtype(dtype):
        model = build_model({"use_atomwise": False}).to(device)
    reference = deepcopy(model)
    # Scripting deliberately retains the original dense Ewald implementation.
    reference.les.ewald = torch.jit.script(reference.les.ewald)
    structures = [atoms(False), atoms(True)]
    structures[1].set_cell([[6.3, 0., 0.], [.4, 6.8, 0.], [.2, .3, 7.1]])
    inputs = batch(model, structures)
    inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
              if torch.is_tensor(value) else value
              for key, value in inputs.items()}
    outputs, gradients = [], []
    for candidate in (model, reference):
        with default_dtype(dtype):
            out = candidate(deepcopy(inputs), training=True, compute_stress=True, compute_node_feats=False)
            loss = out["forces"].square().sum() + .01 * out["energy"].square().sum()
            gradients.append(torch.autograd.grad(loss, tuple(candidate.parameters()), allow_unused=True))
        outputs.append(out)
    tolerance = 4e-5 if dtype == torch.float32 else 2e-10
    for key in ("energy", "forces", "stress", "les_energy"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=tolerance, rtol=tolerance)
    for actual, expected in zip(*gradients):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
