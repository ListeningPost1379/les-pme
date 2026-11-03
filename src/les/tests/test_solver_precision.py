"""Solver selection and independent LES precision must survive force training."""
import importlib.metadata
import importlib.util

import pytest
import torch

from les import Les
from les.module.ewald import Ewald
from les.module.pme import PME


@pytest.fixture
def upstream():
    path = importlib.metadata.distribution("les").locate_file("les/module/ewald.py")
    spec = importlib.util.spec_from_file_location("precision_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Ewald


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_original_and_blocked_share_input_precision(upstream, dtype):
    r = torch.tensor([[.31, .64, .85], [1.37, 1.96, 2.23]], dtype=dtype)
    q = torch.tensor([.4, -.2], dtype=dtype)
    for cell in (torch.zeros(1, 3, 3, dtype=dtype), torch.eye(3, dtype=dtype)[None] * 6.3):
        actual = Ewald()(q, r, cell)[0]
        expected = upstream()(q, r, cell)[0]
        assert actual.dtype == expected.dtype == dtype
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("solver", ["original", "blocked", "pme"])
@pytest.mark.parametrize("input_dtype,compute_dtype", [(torch.float32, "float64"), (torch.float64, "float32")])
@pytest.mark.parametrize("periodic", [False, True])
def test_solver_precision_cast_keeps_force_and_strain_loss_gradients(solver, input_dtype, compute_dtype, periodic):
    r = torch.tensor([[.31, .64, .85], [1.37, 1.96, 2.23]], dtype=input_dtype, requires_grad=True)
    strain = torch.zeros(3, 3, dtype=input_dtype, requires_grad=True)
    transform = torch.eye(3, dtype=input_dtype) + strain
    cell = torch.eye(3, dtype=input_dtype) * (6.3 if periodic else 0)
    parameter = torch.tensor([.13, .17, -.11], dtype=input_dtype, requires_grad=True)
    positions = r @ transform
    args = {"reciprocal_solver": solver, "compute_dtype": compute_dtype, "use_atomwise": False}
    if solver == "pme":
        args.update(pme_mesh=32, pme_cutoff_dtype="float64")
    energy = Les(args)(positions, (cell @ transform)[None],
                       latent_charges=torch.sin(positions @ parameter))["E_lr"]
    assert energy.dtype == getattr(torch, compute_dtype)
    force, stress = torch.autograd.grad(energy.sum(), (r, strain), create_graph=True)
    gradient = torch.autograd.grad((force - .01).square().sum() + (stress - .02).square().sum(), parameter)[0]
    assert gradient.dtype == input_dtype
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


def test_original_retains_dense_force_graph_but_blocked_recomputes():
    r = torch.rand(24, 3, dtype=torch.float64, requires_grad=True)
    q = torch.randn(24, 1, dtype=r.dtype, requires_grad=True)
    cell = torch.eye(3, dtype=r.dtype)[None] * 8.3
    for solver, saves_dense in (("original", True), ("blocked", False)):
        shapes = []
        def pack(tensor):
            shapes.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            energy = Les({"reciprocal_solver": solver})(r, cell, latent_charges=q)["E_lr"]
            force = torch.autograd.grad(energy.sum(), r, create_graph=True)[0]
        assert bool([s for s in shapes if len(s) == 2 and s[0] == 24 and s[1] > 3]) == saves_dense
        assert torch.isfinite(torch.autograd.grad(force.square().sum(), q)[0]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_original_and_blocked_match_upstream_derivatives(upstream, dtype):
    previous = torch.get_default_dtype()
    try:
        # Explicit precision must also override constants created in another default dtype.
        torch.set_default_dtype(torch.float64 if dtype == torch.float32 else torch.float32)
        r = torch.tensor([[.31, .64, .85], [1.37, 1.96, 2.23]], dtype=torch.float64, requires_grad=True)
        parameter = torch.tensor([.13, .17, -.11], dtype=r.dtype, requires_grad=True)
        q = torch.sin(r @ parameter)
        cell = torch.eye(3, dtype=r.dtype) * 6.3
        results = []
        for solver in (upstream(), Ewald(reciprocal_solver='original', compute_dtype=str(dtype).split('.')[-1]),
                       Ewald(reciprocal_solver='blocked', compute_dtype=str(dtype).split('.')[-1])):
            energy = solver.compute_potential_triclinic(r.to(dtype), q.to(dtype), cell.to(dtype))['pot']
            force = -torch.autograd.grad(energy.sum(), r, create_graph=True, retain_graph=True)[0]
            gradient = torch.autograd.grad((force - .01).square().sum(), parameter, retain_graph=True)[0]
            results.append((energy, force, gradient))
        assert results[1][0].dtype == results[2][0].dtype == dtype
        tolerance = 3e-5 if dtype == torch.float32 else 2e-11
        for result in results[1:]:
            for actual, expected in zip(result, results[0]):
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance, check_dtype=False)
    finally:
        torch.set_default_dtype(previous)


def test_fp32_pme_fp64_cutoff_preserves_boundary_modes(monkeypatch):
    r = torch.tensor([[.31, .64, .85], [1.37, 1.96, 2.23]], dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=r.dtype) * 6
    outputs = []
    fft_dtypes = []
    original_fft = torch.fft.rfftn
    def observe(tensor, *args, **kwargs):
        fft_dtypes.append(tensor.dtype)
        return original_fft(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.fft, "rfftn", observe)
    for dtype, cutoff in (("float64", "float64"), ("float32", "float64"), ("float32", "float32")):
        parameter = torch.tensor([.13, .17, -.11], dtype=r.dtype, requires_grad=True)
        solver = PME(mesh=64, compute_dtype=dtype, pme_cutoff_dtype=cutoff)
        energy = solver.compute_potential_triclinic(r, torch.sin(r @ parameter), cell)["pot"]
        force = -torch.autograd.grad(energy.sum(), r, create_graph=True)[0]
        gradient = torch.autograd.grad((force - .01).square().sum(), parameter)[0]
        outputs.append((energy, force, gradient))
    assert outputs[1][0].dtype == torch.float32
    assert fft_dtypes == [torch.float64, torch.float32, torch.float32]
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-5)
    assert (outputs[2][1] - outputs[0][1]).abs().max() > 1e-4


@pytest.mark.parametrize("arguments", [
    {"compute_dtype": "float16"}, {"reciprocal_solver": "unknown"},
    {"reciprocal_solver": "pme", "pme_cutoff_dtype": "float16"},
    {"reciprocal_solver": "original", "pme_mesh": 32},
    {"is_periodic": True, "compute_dtype": "float32"},
    {"reciprocal_solver": "blocked", "reciprocal_mode": "smooth"},
])
def test_invalid_or_unsupported_solver_options_raise(arguments):
    with pytest.raises(ValueError):
        Les(arguments)
