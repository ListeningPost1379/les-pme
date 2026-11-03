"""PME training acceptance against the independently installed upstream LES."""

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def upstream():
    path = Path(importlib.metadata.distribution("les").locate_file("les/module/ewald.py"))
    if not path.is_file() or path.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        pytest.skip("An independently installed upstream LES is required")
    spec = importlib.util.spec_from_file_location("upstream_les_pme_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Ewald


def case(channels=1):
    r = torch.tensor([[.4, .5, .7], [1.3, .6, .8], [.2, 1.5, .6], [2.1, 1.2, 1.4]],
                     dtype=torch.float64, requires_grad=True)
    q = torch.tensor([.4, -.2, .15, .1], dtype=r.dtype)[:, None].repeat(1, channels).requires_grad_()
    cell = torch.tensor([[6.2, 0., 0.], [.6, 5.8, 0.], [.3, .4, 6.4]], dtype=r.dtype, requires_grad=True)
    return r, q, cell


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("remove_self", [False, True])
@pytest.mark.parametrize("neutral", [False, True])
def test_pme_matches_upstream_through_force_loss(upstream, channels, remove_self, neutral):
    from les.module.pme import PME

    r, q0, cell = case(channels)
    q = q0 + .03 * r.square().sum(1, keepdim=True)
    if neutral:
        q = q - q.mean(0)
    parameters = dict(sigma=.85, dl=1.3, remove_self_interaction=remove_self)
    results, firsts, seconds = [], [], []
    for solver in (PME(mesh=64, **parameters), upstream(**parameters)):
        energy = solver.compute_potential_triclinic(r, q, cell)["pot"].sum()
        first = torch.autograd.grad(energy, (r, q0, cell), create_graph=True, retain_graph=True)
        second = torch.autograd.grad(first[0].square().sum(), (r, q0, cell), retain_graph=True)
        results.append(energy)
        firsts.append(first)
        seconds.append(second)
    torch.testing.assert_close(*results, atol=2e-8, rtol=0)
    for levels in (firsts, seconds):
        for actual, expected in zip(*levels):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_mesh_refinement_reduces_upstream_errors(upstream):
    from les.module.pme import PME

    r, q, cell = case()
    parameters = dict(sigma=.85, dl=1.3)
    energies, gradients = [], []
    for solver in (upstream(**parameters), PME(mesh=16, **parameters), PME(mesh=32, **parameters)):
        energy = solver.compute_potential_triclinic(r, q, cell)["pot"].sum()
        energies.append(energy)
        gradients.append(torch.autograd.grad(energy, (r, q, cell), retain_graph=True))
    assert abs(energies[2] - energies[0]) < abs(energies[1] - energies[0]) / 5
    for reference, coarse, fine in zip(*gradients):
        assert (fine - reference).abs().max() < (coarse - reference).abs().max() / 5


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))])
def test_zero_charge_hessian_and_force_target_match_upstream(upstream, device):
    from les.module.pme import PME

    r, _, cell = case()
    r = r[:2].to(device).detach().requires_grad_()
    cell = cell.to(device)
    q0 = torch.zeros(2, 1, dtype=r.dtype, device=device, requires_grad=True)
    q = q0 + .03 * (r.square().sum(1, keepdim=True)
                     - r.detach().square().sum(1, keepdim=True))
    results = []
    for solver in (PME(mesh=32, sigma=.85, dl=1.3), upstream(sigma=.85, dl=1.3)):
        energy = solver.compute_potential_triclinic(r, q, cell)["pot"].sum()
        dr, dq = torch.autograd.grad(energy, (r, q0), create_graph=True, retain_graph=True)
        hessian_vector = torch.autograd.grad(dq.sum(), q0, retain_graph=True)[0]
        force_gradient = torch.autograd.grad((-dr - 1).square().sum(), q0, retain_graph=True)[0]
        results.append((hessian_vector, force_gradient))
    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_pme_gradcheck_and_gradgradcheck():
    from les.module.pme import PME

    r, q, cell = case()
    r, q = r[:2].detach().requires_grad_(), q[:2].detach().requires_grad_()
    cell = cell.detach()
    solver = PME(mesh=12)

    def energy(positions, charges):
        return solver.compute_potential_triclinic(positions, charges, cell)["pot"].sum()

    assert torch.autograd.gradcheck(energy, (r, q))
    assert torch.autograd.gradgradcheck(energy, (r, q))


def test_periodic_translation_and_scalar_charge_shape():
    from les.module.pme import PME

    r, q, cell = case()
    solver = PME(mesh=(16, 18, 20))
    actual = solver.compute_potential_triclinic(r, q, cell)["pot"]
    shifted = solver.compute_potential_triclinic(r + 3 * cell[0] - 2 * cell[1], q[:, 0], cell)["pot"]
    torch.testing.assert_close(actual, shifted, atol=1e-12, rtol=1e-12)


def test_pme_rejects_invalid_geometry_and_accepts_zero_dipoles():
    from les.module.pme import PME

    r, q, cell = case()
    with pytest.raises(ValueError):
        PME(mesh=5)
    with pytest.raises(ValueError):
        PME(mesh=16).compute_potential_triclinic(r, q, torch.zeros_like(cell))
    with pytest.raises(ValueError):
        PME(mesh=6, dl=.1).compute_potential_triclinic(r, q, cell)
    solver = PME(mesh=16)
    torch.testing.assert_close(solver.compute_potential_triclinic(r, q, cell)['pot'],
        solver.compute_potential_triclinic(r, q, cell, u=torch.zeros(4, 1, 3))['pot'])


@pytest.mark.parametrize("mesh", [(32, 32, 32), (31, 33, 35)])
def test_potential_matches_upstream_and_energy_charge_derivative(upstream, mesh):
    from les.module.pme import PME

    r, q, cell = case()
    parameters = dict(sigma=.85, dl=1.3)
    actual = PME(mesh=mesh, **parameters).compute_potential_triclinic(r, q, cell, compute_potential=True)
    expected = upstream(**parameters).compute_potential_triclinic(r, q, cell, compute_potential=True, compute_field=True)
    torch.testing.assert_close(actual["phi"], expected["phi"], atol=2e-6, rtol=0)
    derivative = torch.autograd.grad(actual["pot"].sum(), q)[0]
    torch.testing.assert_close(actual["phi"], derivative, atol=2e-12, rtol=2e-12)


def test_les_constructor_selects_pme_explicitly():
    from les import Les
    from les.module.pme import PME

    solver = Les({"reciprocal_solver": "pme", "pme_mesh": [16, 18, 20]})
    assert isinstance(solver.ewald, PME)
    assert solver.ewald.mesh == (16, 18, 20)
    with pytest.raises(ValueError):
        Les({"reciprocal_solver": "unknown"})


def test_pme_auto_mesh_sizing_and_large_cell_scaling():
    from les import Les
    from les.module.pme import (
        PME,
        fft_friendly_dimension,
        recommend_pme_mesh,
        recommend_pme_mesh_dimension,
    )

    assert fft_friendly_dimension(64) == 64
    assert fft_friendly_dimension(65) == 80
    assert recommend_pme_mesh_dimension(30.0, dl=2.0) == 64
    assert recommend_pme_mesh_dimension(40.0, dl=2.0) == 80
    assert recommend_pme_mesh_dimension(50.0, dl=2.0) == 96
    assert recommend_pme_mesh_dimension(70.0, dl=2.0) == 128
    assert recommend_pme_mesh_dimension(99.0, dl=2.0) == 192

    # Default PME() and Les(reciprocal_solver="pme") use auto_mesh=True (64^3 placeholder until a cell is seen)
    les_pme = Les({"use_atomwise": False, "reciprocal_solver": "pme", "dl": 2.0, "sigma": 1.0})
    assert les_pme.ewald.auto_mesh is True
    assert les_pme.ewald.mesh == (64, 64, 64)

    # Small box (10 A) resolves to (64, 64, 64)
    r_small = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.5, 0.2]], dtype=torch.float64)
    q_small = torch.tensor([0.5, -0.5], dtype=torch.float64)
    cell_small = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0
    out_small = les_pme(positions=r_small, cell=cell_small, latent_charges=q_small)
    assert torch.isfinite(out_small["E_lr"]).all()
    assert les_pme.ewald.last_resolved_mesh == (64, 64, 64)

    # Large box (70x70x99 A, where 2*bound >= 64 would fail a fixed 64^3 mesh) automatically scales to (128, 128, 192)
    cell_large = torch.diag(torch.tensor([70.0, 70.0, 99.0], dtype=torch.float64)).unsqueeze(0)
    out_large = les_pme(positions=r_small, cell=cell_large, latent_charges=q_small)
    assert torch.isfinite(out_large["E_lr"]).all()
    assert les_pme.ewald.last_resolved_mesh == (128, 128, 192)
    assert recommend_pme_mesh(cell_large, dl=2.0) == (128, 128, 192)

    # An explicit 64 is a fixed 64^3 grid even on a large cell; only "auto" scales
    les_64 = Les({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": 64, "dl": 2.0, "sigma": 1.0})
    assert les_64.ewald.auto_mesh is False
    assert les_64.ewald.resolve_mesh(cell_large[0]) == (64, 64, 64)
    for mesh in (64, [64, 64, 64], (64, 64, 64)):
        assert PME(mesh=mesh).auto_mesh is False
    for mesh in ("auto", "AUTO", None):
        assert PME(mesh=mesh).auto_mesh is True
    assert PME().resolve_mesh(cell_large[0]) == (128, 128, 192)

    # Explicit small mesh=16 remains fixed and raises ValueError on a 70 A cell
    les_fixed = Les({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": 16, "dl": 2.0, "sigma": 1.0})
    assert les_fixed.ewald.auto_mesh is False
    with pytest.raises(ValueError, match="PME mesh is too small"):
        les_fixed(positions=r_small, cell=cell_large, latent_charges=q_small)


def _restore_pme(state):
    from les.module.pme import PME

    restored = PME.__new__(PME)
    restored.__setstate__(state)
    return restored


def test_pme_mesh_pickle_compatibility():
    from les.module.pme import PME

    cell_large = torch.diag(torch.tensor([70.0, 70.0, 99.0], dtype=torch.float64))

    # Saved before automatic meshes existed (pre-0cc6774): plain `mesh` attribute,
    # always a fixed grid, so a stored 64 stays a fixed 64^3 grid.
    for stored in ((64, 64, 64), (16, 18, 20)):
        state = dict(PME(mesh=stored, dl=2.0, sigma=1.0).__dict__)
        state["mesh"] = state.pop("_mesh")
        state.pop("auto_mesh")
        state.pop("last_resolved_mesh")
        restored = _restore_pme(state)
        assert restored.auto_mesh is False
        assert restored.mesh == stored
        assert restored.resolve_mesh(cell_large) == stored

    # Saved between 0cc6774 and the explicit-64 fix: auto_mesh is stored (and
    # _explicit_mesh); auto_mesh=True ran as "auto" then and stays "auto".
    state = dict(PME(dl=2.0, sigma=1.0).__dict__)
    state["_explicit_mesh"] = False
    restored = _restore_pme(state)
    assert restored.auto_mesh is True
    assert "_explicit_mesh" not in restored.__dict__
    assert restored.resolve_mesh(cell_large) == (128, 128, 192)

    state = dict(PME(mesh=32, dl=2.0, sigma=1.0).__dict__)
    state["_explicit_mesh"] = True
    restored = _restore_pme(state)
    assert restored.auto_mesh is False
    assert restored.mesh == (32, 32, 32)

    # A full pickle round trip keeps the new semantics.
    import pickle

    for mesh, auto in ((64, False), ("auto", True)):
        clone = pickle.loads(pickle.dumps(PME(mesh=mesh, dl=2.0, sigma=1.0)))
        assert clone.auto_mesh is auto
        assert clone.resolve_mesh(cell_large) == ((128, 128, 192) if auto else (64, 64, 64))
