"""PME MACELES force training and solver reconstruction against original LES."""

from copy import deepcopy
import importlib.metadata
import importlib.util
import io
from pathlib import Path

import pytest
import torch

pytest.importorskip("mace")
from les.module.pme import PME
from mace.tools.scripts_utils import extract_model
from mace.tools.torch_tools import default_dtype
from tests.golden.build_maceles_anchor import build_model
from tests.extensions.les.les_fixtures import atoms, batch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_pme_force_training_matches_independent_upstream(dtype, device):
    with default_dtype(dtype):
        model = build_model({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": 32}).to(device)
    assert isinstance(model.les.ewald, PME)
    reference = deepcopy(model)
    path = importlib.metadata.distribution("les").locate_file("les/module/ewald.py")
    assert not path.resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src"), "Original installed LES required"
    spec = importlib.util.spec_from_file_location("original_les_for_maceles_pme", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference.les.ewald = module.Ewald()
    inputs = batch(model, [atoms(False), atoms(True)])
    outputs, gradients = [], []
    for candidate in (model, reference):
        with default_dtype(dtype):
            data = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                    if torch.is_tensor(value) else value
                    for key, value in deepcopy(inputs).items()}
            out = candidate(data, training=True, compute_stress=True, compute_node_feats=False)
            loss = out["forces"].square().sum() + .01 * out["energy"].square().sum()
            gradients.append(torch.autograd.grad(loss, tuple(candidate.parameters()), allow_unused=True))
            outputs.append(out)
    tolerance = 4e-5 if dtype == torch.float32 else 5e-7
    for key in ("energy", "forces", "stress", "les_energy"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=tolerance, rtol=tolerance, check_dtype=False)
    for actual, expected in zip(*gradients):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_pme_copy_checkpoint_and_reconstruction_keep_solver():
    with default_dtype(torch.float64):
        model = build_model({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": [16, 18, 20]})
    assert isinstance(model.les.ewald, PME)
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    for candidate in (deepcopy(model), torch.load(stream, weights_only=False), extract_model(model)):
        assert isinstance(candidate.les.ewald, PME)
        assert candidate.les.ewald.mesh == (16, 18, 20)
        assert candidate.get_les_arguments()["reciprocal_solver"] == "pme"
    model.set_les_reciprocal_mode(reciprocal_solver="direct")
    assert not isinstance(model.les.ewald, PME)


@pytest.mark.parametrize("solver", ["original", "blocked", "pme"])
def test_explicit_solver_precision_survives_copy_checkpoint_and_reconstruction(solver):
    options = {"use_atomwise": False, "reciprocal_solver": solver, "compute_dtype": "float64"}
    if solver == "pme":
        options.update(compute_dtype="float32", pme_mesh=32, pme_cutoff_dtype="float64")
    with default_dtype(torch.float32):
        model = build_model(options)
        inputs = batch(model, [atoms(False), atoms(True)])
        inputs = {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
                  for key, value in inputs.items()}
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        candidates = [model, deepcopy(model), torch.load(stream, weights_only=False), extract_model(model)]
        for candidate in candidates:
            actual = candidate.get_les_arguments()
            for key, expected in options.items():
                assert actual[key] == ((32, 32, 32) if key == "pme_mesh" else expected)
            out = candidate(deepcopy(inputs), training=True, compute_stress=True, compute_node_feats=False)
            assert out["les_energy"].dtype == getattr(torch, options["compute_dtype"])
            loss = (out["forces"] - .01).square().sum() + (out["stress"] - .01).square().sum()
            gradients = torch.autograd.grad(loss, tuple(candidate.les_readouts.parameters()))
            assert all(torch.isfinite(g).all() for g in gradients)


def test_old_pme_checkpoint_without_precision_fields_keeps_fp64_periodic_default():
    with default_dtype(torch.float32):
        model = build_model({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": 32})
    for key in ("compute_dtype", "pme_cutoff_dtype", "reciprocal_solver"):
        delattr(model.les.ewald, key)
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert restored.les.ewald.compute_dtype == "auto"
    assert restored.les.ewald.pme_cutoff_dtype == "float64"
    r = torch.tensor([[.3, .4, .5], [1., 1.2, 1.5]], dtype=torch.float32)
    out = restored.les.ewald.compute_potential_triclinic(r, r[:, 0], torch.eye(3) * 6.3)
    assert out["pot"].dtype == torch.float64
