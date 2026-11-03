"""Keep the original reciprocal operator through force-loss backward."""

import pytest
import torch

from les.module.ewald import Ewald
from les.module.reciprocal import reciprocal_energy


def dense_energy(solver, r, q, cell):
    # Full +/- sphere and complex exponentials, independent of the half-space path.
    bounds = [max(1, int(n / solver.dl)) for n in cell.detach().norm(dim=1)]
    axes = [torch.arange(-n, n + 1, device=r.device) for n in bounds]
    modes = torch.cartesian_prod(*axes).to(r.dtype)
    k = modes @ (2 * torch.pi * cell.inverse().T)
    k2 = k.square().sum(1)
    keep = (k2 > 0) & (k2 <= solver.k_sq_max)
    k, k2 = k[keep], k2[keep]
    spectrum = q.to(torch.complex128 if r.dtype == torch.float64 else torch.complex64).T @ torch.exp(1j * (r @ k.T))
    energy = (spectrum.abs().square() * (-solver.sigma_sq_half * k2).exp() / k2).sum()
    energy = energy * solver.norm_factor / cell.det()
    if solver.remove_self_interaction:
        energy = energy - q.square().sum() * solver.norm_factor / (solver.sigma * solver.twopi**1.5)
    return energy


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("channels,remove_self", [(1, True), (3, False)])
def test_energy_force_strain_and_force_loss_match_full_sphere(dtype, channels, remove_self):
    torch.manual_seed(1379)
    r0 = torch.rand(5, 3, dtype=dtype, requires_grad=True)
    q0 = torch.randn(5, channels, dtype=dtype, requires_grad=True)
    cell0 = torch.tensor([[6.3, 0., 0.], [.4, 6.8, 0.], [.2, .3, 7.1]], dtype=dtype)
    strain = torch.zeros(3, 3, dtype=dtype, requires_grad=True)
    transform = torch.eye(3, dtype=dtype) + (strain + strain.T) / 2
    r, cell = r0 @ transform, cell0 @ transform
    q = q0 + .03 * r.square().sum(1, keepdim=True)
    solver = Ewald(sigma=1.2, dl=2.1, remove_self_interaction=remove_self)
    actual = solver.compute_potential_triclinic(r, q, cell)["pot"].sum()
    expected = dense_energy(solver, r, q, cell)
    first, second = [], []
    for energy in (actual, expected):
        gradients = torch.autograd.grad(energy, (r0, q0, strain), create_graph=True, retain_graph=True)
        first.append(gradients)
        second.append(torch.autograd.grad(gradients[0].square().sum(), (r0, q0, strain), retain_graph=True))
    tolerance = 3e-4 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for levels in (first, second):
        for candidate, reference in zip(*levels):
            torch.testing.assert_close(candidate, reference, atol=tolerance, rtol=tolerance)


def test_force_graph_does_not_save_atom_by_mode_tensors():
    torch.manual_seed(42)
    r = torch.rand(128, 3, dtype=torch.float64, requires_grad=True)
    q = torch.randn(128, 1, dtype=r.dtype, requires_grad=True)
    cell = torch.eye(3, dtype=r.dtype) * 8.3
    shapes = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        energy = Ewald().compute_potential_triclinic(r, q, cell)["pot"].sum()
        force = -torch.autograd.grad(energy, r, create_graph=True)[0]
    # Original Ewald retains [N,K] phase/trigonometric arrays even after dE/dR.
    assert not [shape for shape in shapes if len(shape) == 2 and shape[0] == len(r) and shape[1] > 3]
    gradient = torch.autograd.grad(force.square().sum(), q)[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("modes", [0, 1, 1031])
def test_recomputed_sum_handles_empty_and_partial_blocks(modes):
    torch.manual_seed(42)
    r = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
    q = torch.randn(3, 2, dtype=r.dtype, requires_grad=True)
    k = torch.randn(modes, 3, dtype=r.dtype, requires_grad=True)
    weight = torch.rand(modes, dtype=r.dtype, requires_grad=True)
    actual = reciprocal_energy(r, q, k, weight)
    spectrum = q.to(torch.complex128).T @ torch.exp(1j * (r @ k.T))
    expected = (spectrum.abs().square() * weight).sum(1)
    torch.testing.assert_close(actual, expected)
    derivatives = []
    for energy in (actual, expected):
        derivatives.append(torch.autograd.grad(energy.sum(), (r, q, k, weight), create_graph=True))
    for a, b in zip(*derivatives):
        torch.testing.assert_close(a, b)
    for a, b in zip(*[
        torch.autograd.grad(sum(t.square().sum() for t in gradient), (r, q, k, weight))
        for gradient in derivatives
    ]):
        torch.testing.assert_close(a, b)


def test_recomputed_sum_gradcheck_and_gradgradcheck():
    torch.manual_seed(1379)
    args = tuple(torch.randn(*shape, dtype=torch.float64, requires_grad=True)
                 for shape in ((2, 3), (2, 2), (4, 3), (4,)))
    assert torch.autograd.gradcheck(reciprocal_energy, args)
    assert torch.autograd.gradgradcheck(reciprocal_energy, args)


def test_autocast_keeps_original_reduction_precision():
    torch.manual_seed(42)
    r = torch.rand(8, 3, requires_grad=True)
    q = torch.randn(8, 1, requires_grad=True)
    cell = torch.eye(3) * 8.3
    solver = Ewald()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = solver.compute_potential_triclinic(r, q, cell)["pot"]
        # Field requests retain the original broadcast reduction.
        expected = solver.compute_potential_triclinic(r, q, cell, compute_field=True)["pot"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tf32_keeps_original_reduction_precision():
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(42)
        r = torch.rand(128, 3, device="cuda", requires_grad=True)
        q = torch.randn(128, 3, device="cuda", requires_grad=True)
        cell = torch.eye(3, device="cuda") * 8.3
        solver = Ewald()
        actual = solver.compute_potential_triclinic(r, q, cell)["pot"]
        expected = solver.compute_potential_triclinic(r, q, cell, compute_field=True)["pot"]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
