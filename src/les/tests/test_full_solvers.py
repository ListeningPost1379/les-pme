"""Multipole/response acceptance against the independently installed original LES."""
import importlib.metadata
import importlib.util
from pathlib import Path

import pytest
import torch

from les.module.ewald import Ewald
from les.module.pme import PME
from les import Les


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def original():
    root = Path(importlib.metadata.distribution("les").locate_file("les/module"))
    assert not root.resolve().is_relative_to(Path(__file__).resolve().parents[3])
    module = _load("original_full_ewald", root / "ewald.py")
    module.make_kernels = _load("original_full_kernels", root / "make_kernels.py").make_kernels
    return module.Ewald


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable'))]


def inputs(dtype=torch.float64, channels=2, device='cpu'):
    torch.manual_seed(20261024)
    r = torch.tensor([[.41, .63, .82], [1.67, .94, .53], [.73, 2.11, 1.42]], dtype=dtype, device=device, requires_grad=True)
    cell = torch.tensor([[6.2, 0., 0.], [.3, 5.9, 0.], [.2, .4, 6.7]], dtype=dtype, device=device, requires_grad=True)
    p = torch.randn(3, channels, dtype=dtype, device=device, requires_grad=True)
    return r, cell, p, values_from(r, p)


def values_from(r, p):
    # All permanent and response inputs depend on positions and shared parameters.
    q = .17 * torch.sin(r @ p)
    u = .06 * torch.cos(r[:, None, :] + p.T[None, :, :])
    raw = .03 * torch.sin(r[:, None, :, None] + p.T[None, :, None, :])
    quad = (raw + raw.transpose(-1, -2)) / 2
    alpha = .002 * (torch.eye(3, dtype=r.dtype, device=r.device)[None, None] + .03 * quad)
    kappa = .001 * (1 + q.square())
    return dict(q=q, u=u, quad=quad, alpha=alpha, kappa=kappa,
                e_ext=r.new_tensor([.01, -.02, .03]))


def evaluate(solver, periodic, r, cell, values):
    if periodic:
        # Original has a legacy potential-only prefactor bug: request both outputs.
        return solver.compute_potential_triclinic(r, cell_now=cell, compute_potential=True,
                                                 compute_field=True, **values)
    return solver.compute_potential_realspace(r, compute_field=True, **values)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("remove_self", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize('device', DEVICES)
def test_blocked_all_features_matches_original_through_training(original, periodic, remove_self, dtype, device):
    r, cell, parameter, values = inputs(dtype, device=device)
    results = []
    options = dict(sigma=.9, dl=1.7, remove_self_interaction=remove_self)
    for solver in (original(**options), Ewald(reciprocal_solver="blocked", **options)):
        out = evaluate(solver, periodic, r, cell, values)
        variables = (r, parameter, cell) if periodic else (r, parameter)
        first = torch.autograd.grad(out["pot"].sum(), variables, create_graph=True, retain_graph=True)
        loss = sum((g - .017).square().sum() for g in first)
        second = torch.autograd.grad(loss, variables, retain_graph=True) if periodic or len(results) else ()
        results.append((out, first, second))
    tolerance = 2e-4 if dtype == torch.float32 else 2e-10
    for key in ("pot", "phi", "field", "q_induced", "u_induced"):
        torch.testing.assert_close(results[1][0][key], results[0][0][key], atol=tolerance, rtol=tolerance)
    for level in ((1, 2) if periodic else (1,)):
        for actual, expected in zip(results[1][level], results[0][level]):
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert all(torch.isfinite(g).all() for g in results[1][2])


@pytest.mark.parametrize("periodic", [False, True])
def test_blocked_force_graph_does_not_retain_pair_or_mode_intermediates(periodic):
    r, cell, _, values = inputs()
    shapes = []
    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = evaluate(Ewald(reciprocal_solver="blocked", dense_pairs=0), periodic, r, cell, values)
        force = torch.autograd.grad(out["pot"].sum(), r, create_graph=True)[0]
    # Saved multipoles themselves [N,C,3,3] are allowed; kernels and phases are not.
    assert not any(len(s) >= 5 and s[:2] == (3, 3) for s in shapes)
    assert not any(len(s) == 2 and s[0] == 3 and s[1] > 3 for s in shapes)
    assert torch.isfinite(torch.autograd.grad(force.square().sum(), r)[0]).all()


# Periodic PME uses the mesh, not the blocked multipole sums.
@pytest.mark.parametrize("method,periodic", [("blocked", False), ("blocked", True), ("pme", False)])
def test_small_dense_fast_path_matches_streaming_through_force_loss(method, periodic):
    r, cell, p, _ = inputs()
    make = (lambda pairs: Ewald(reciprocal_solver="blocked", dense_pairs=pairs)) if method == "blocked" else (
        lambda pairs: PME(mesh=(31, 33, 35), dense_pairs=pairs))
    results, shapes = [], []
    for pairs in (0, 9 * 10**6):
        def pack(tensor):
            shapes.append((pairs, tuple(tensor.shape)))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            out = evaluate(make(pairs), periodic, r, cell, values_from(r, p))
            variables = (r, p)
            first = torch.autograd.grad(out["pot"].sum(), variables, create_graph=True)
        second = torch.autograd.grad(sum((g - .017).square().sum() for g in first), variables)
        results.append((out, first, second))
    for key in ("pot", "phi", "field", "q_induced", "u_induced"):
        torch.testing.assert_close(results[1][0][key], results[0][0][key], atol=1e-12, rtol=1e-12)
    for level in (1, 2):
        for actual, expected in zip(results[1][level], results[0][level]):
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    # The dense path keeps ordinary pair/mode intermediates; streaming does not.
    def keeps(pairs):
        return any(q == pairs and ((len(s) >= 5 and s[:2] == (3, 3)) or (len(s) == 2 and s[0] == 3 and s[1] > 3))
                   for q, s in shapes)
    assert keeps(9 * 10**6) and not keeps(0)


def test_dense_pairs_threshold_and_validation():
    from les.module.blocked import DEFAULT_DENSE_PAIRS, real_multipoles
    assert Ewald(reciprocal_solver="blocked").dense_pairs == DEFAULT_DENSE_PAIRS
    for bad in (-1, 1.5, True, None):
        with pytest.raises(ValueError):
            Ewald(reciprocal_solver="blocked", dense_pairs=bad)
    assert Les({"reciprocal_solver": "pme", "dense_pairs": 7}).ewald.dense_pairs == 7
    r, _, _, values = inputs()
    for pairs, dense in ((len(r) ** 2, True), (len(r) ** 2 - 1, False)):
        shapes = []
        def pack(tensor):
            shapes.append(tuple(tensor.shape))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            real_multipoles(r, values["q"], values["u"], None, 1., 1., True, block_size=2, dense_pairs=pairs)
        assert any(len(s) >= 3 and s[:2] == (3, 3) for s in shapes) == dense
    legacy = Ewald(reciprocal_solver="blocked")
    state = legacy.__dict__.copy()
    state.pop("dense_pairs")
    restored = Ewald.__new__(Ewald)
    restored.__setstate__(state)
    assert restored.dense_pairs == DEFAULT_DENSE_PAIRS


def test_nonperiodic_second_derivative_matches_original_force_finite_difference(original):
    r, cell, p, _ = inputs()
    directions = (torch.sin(r.detach()), torch.cos(p.detach()))
    def force_loss(solver, rr, pp):
        out = evaluate(solver, False, rr, cell, values_from(rr, pp))
        force = -torch.autograd.grad(out['pot'].sum(), rr, create_graph=True)[0]
        return (force - .017).square().sum()
    actual = torch.autograd.grad(force_loss(Ewald(reciprocal_solver='blocked'), r, p), (r, p))
    projected = sum((g * d).sum() for g, d in zip(actual, directions))
    eps = 2e-5
    losses = [force_loss(original(), (r + sign * eps * directions[0]).detach().requires_grad_(),
                         (p + sign * eps * directions[1]).detach().requires_grad_()).detach()
              for sign in (1, -1)]
    torch.testing.assert_close(projected, (losses[0] - losses[1]) / (2 * eps), atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('features', ['q', 'u', 'Q', 'iq', 'iu', 'all'])
@pytest.mark.parametrize('method', ['blocked', 'pme'])
def test_scalar_optional_features_and_outputs(original, periodic, features, method):
    r, cell, _, values = inputs(channels=1)
    enabled = {'q': (), 'u': ('u',), 'Q': ('quad',), 'iq': ('kappa',),
               'iu': ('alpha',), 'all': ('u', 'quad', 'alpha', 'kappa', 'e_ext')}[features]
    values = {k: v.squeeze(1) if k != 'e_ext' else v for k, v in values.items() if k == 'q' or k in enabled}
    for self_interaction in (False, True):
        solver = Ewald(reciprocal_solver='blocked', remove_self_interaction=self_interaction) if method == 'blocked' else (
            PME(mesh=(31, 33, 35), remove_self_interaction=self_interaction))
        outs = [evaluate(candidate, periodic, r, cell, values) for candidate in
                (original(remove_self_interaction=self_interaction), solver)]
        tolerance = 2e-11 if method == 'blocked' or not periodic else 8e-6
        for key in ('pot', 'phi', 'field', 'q_induced', 'u_induced'):
            torch.testing.assert_close(outs[1][key], outs[0][key], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('modes', [0, 1, 7])
def test_reciprocal_block_tails_and_second_derivatives(modes):
    from les.module.blocked import reciprocal_multipoles
    torch.manual_seed(12)
    shapes = ((2, 3), (2, 1), (2, 1, 3), (2, 1, 3, 3), (modes, 3), (modes,))
    args = tuple(torch.randn(shape, dtype=torch.float64, requires_grad=True) * .1 for shape in shapes)
    def outputs(*tensors):
        return reciprocal_multipoles(*tensors, True, True, block_size=3)
    assert torch.autograd.gradcheck(outputs, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(outputs, args, fast_mode=True)


def test_real_block_tails_and_second_derivatives(original):
    from les.module.blocked import real_multipoles
    r, _, _, values = inputs()
    args = tuple(t.detach().requires_grad_() for t in (r, values['q'], values['u'], values['quad']))
    def outputs(*tensors):
        return real_multipoles(*tensors, 1., 90.4756 / (2 * torch.pi), True, block_size=2)
    actual = outputs(*args)
    expected = original().compute_potential_realspace(r, values['q'], u=values['u'], quad=values['quad'], compute_field=True)
    for a, b in zip(actual, (expected['pot'], expected['phi'], expected['field'])):
        torch.testing.assert_close(a.sum() if a.ndim == 1 else a, b.sum() if b.ndim == 1 else b)
    assert torch.autograd.gradcheck(outputs, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(outputs, args, fast_mode=True)


@pytest.mark.parametrize('remove_self', [False, True])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('device', DEVICES)
def test_pme_all_features_matches_original_through_training(original, remove_self, precision, device):
    r, cell, p, values = inputs(device=device)
    options = dict(sigma=.9, dl=1.7, remove_self_interaction=remove_self, use_epsilon_r_scaling=True)
    results = []
    for solver in (original(**options), PME(mesh=48, compute_dtype=precision, **options)):
        out = evaluate(solver, True, r, cell, values)
        first = torch.autograd.grad(out['pot'].sum(), (r, p, cell), create_graph=True, retain_graph=True)
        second = torch.autograd.grad(sum((g - .017).square().sum() for g in first), (r, p, cell), retain_graph=True)
        results.append((out, first, second))
    tolerance = 2e-5 if precision == 'float32' else 3e-6
    for key in ('pot', 'phi', 'field', 'q_induced', 'u_induced', 'epsilon_r'):
        torch.testing.assert_close(results[1][0][key], results[0][0][key], atol=tolerance, rtol=tolerance, check_dtype=False)
    for level in (1, 2):
        for a, b in zip(results[1][level], results[0][level]):
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)


def test_pme_multipole_mesh_refinement(original):
    r, cell, _, values = inputs()
    data = []
    for solver in (original(), PME(mesh=16), PME(mesh=32)):
        out = evaluate(solver, True, r, cell, values)
        force = torch.autograd.grad(out['pot'].sum(), r, retain_graph=True)[0]
        data.append((out['pot'], out['phi'], out['field'], force))
    for reference, coarse, fine in zip(*data):
        assert (fine - reference).abs().max() < (coarse - reference).abs().max() / 4


def test_pme_multipole_response_gradcheck_and_gradgradcheck():
    r, cell, _, values = inputs(channels=1)
    names = ('q', 'u', 'quad', 'alpha', 'kappa', 'e_ext')
    args = tuple(t.detach().requires_grad_() for t in (r, cell, *(values[n] for n in names)))
    def energy(rr, cc, *features):
        return evaluate(PME(mesh=12), True, rr, cc, dict(zip(names, features)))['pot']
    assert torch.autograd.gradcheck(energy, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(energy, args, fast_mode=True)


@pytest.mark.parametrize('solver', ['blocked', 'pme'])
def test_unrequested_nonperiodic_field_is_zero_with_self_terms(solver):
    r, _, _, values = inputs()
    out = (Ewald(reciprocal_solver=solver) if solver == 'blocked' else PME(mesh=16)).compute_potential_realspace(
        r, values['q'], u=values['u'])
    assert torch.count_nonzero(out['field']) == 0
    out = (Ewald(reciprocal_solver='blocked', remove_self_interaction=False) if solver == 'blocked'
           else PME(mesh=16, remove_self_interaction=False)).compute_potential_realspace(r, values['q'], u=values['u'])
    assert torch.count_nonzero(out['field']) == 0


@pytest.mark.parametrize('method', ['blocked', 'pme'])
@pytest.mark.parametrize('device', DEVICES)
def test_zero_multipole_force_target_and_hessian_match_original(original, method, device):
    r, cell, _, values = inputs(channels=1, device=device)
    base = [torch.zeros_like(values[name], requires_grad=True) for name in ('q', 'u', 'quad')]
    q, u, Q = base
    delta = r - r.detach()
    features = dict(q=q + .03 * delta.sum(-1, keepdim=True), u=u + .02 * delta[:, None],
                    quad=Q + .01 * (delta[:, None, :, None] + delta[:, None, None, :]),
                    kappa=values['kappa'], alpha=values['alpha'])
    solver = Ewald(reciprocal_solver='blocked') if method == 'blocked' else PME(mesh=32)
    results = []
    for candidate in (original(), solver):
        out = evaluate(candidate, True, r, cell, features)
        force = -torch.autograd.grad(out['pot'].sum(), r, create_graph=True, retain_graph=True)[0]
        gradients = torch.autograd.grad((force - .1).square().sum(), base, retain_graph=True)
        results.append(gradients)
    for actual, expected in zip(*results):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('method', ['blocked', 'pme'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batched_field', [False, True])
def test_les_mixed_boundaries_roundtrip_and_response_training(original, method, dtype, batched_field):
    import io
    from copy import deepcopy
    from les import Les
    r, cell, p, values = inputs(dtype)
    positions = torch.cat((r, r + .21), dim=0)
    features = {name: torch.cat((value, value * 1.07), dim=0) if name != 'e_ext' else value
                for name, value in values.items()}
    if batched_field:
        features['e_ext'] = torch.stack((values['e_ext'], -2 * values['e_ext']))
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    cells = torch.stack((cell, torch.zeros_like(cell)))
    kwargs = dict(positions=positions, cell=cells, batch=batch, e_ext=features['e_ext'],
                  latent_charges=features['q'], latent_dipoles=features['u'], latent_quads=features['quad'],
                  latent_kappas=features['kappa'], latent_alphas=features['alpha'])
    options = dict(reciprocal_solver=method, compute_dtype=str(dtype).split('.')[-1])
    if method == 'pme':
        options['pme_mesh'] = 32
    model = Les(options)
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    outputs = [candidate(**kwargs) for candidate in (model, deepcopy(model), torch.load(stream, weights_only=False))]
    ref = original()
    reference = [evaluate(ref, periodic, r + offset, cell,
                          {k: v * factor if k != 'e_ext' else
                           (features['e_ext'][index] if batched_field else v)
                           for k, v in values.items()})
                 for index, (periodic, offset, factor) in enumerate(((True, 0., 1.), (False, .21, 1.07)))]
    tolerance = 3e-4 if dtype == torch.float32 else 5e-6
    for out in outputs:
        torch.testing.assert_close(out['E_lr'], torch.cat([d['pot'] for d in reference]), atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(out['latent_charges'], features['q'] + torch.cat([d['q_induced'] for d in reference]),
                                   atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(out['latent_dipoles'], features['u'] + torch.cat([d['u_induced'] for d in reference]),
                                   atol=tolerance, rtol=tolerance)
        force, stress = torch.autograd.grad(out['E_lr'].sum(), (r, cell), create_graph=True, retain_graph=True)
        gradient = torch.autograd.grad((force - .01).square().sum() + (stress - .02).square().sum(), p, retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0


@pytest.mark.parametrize('method', ['blocked', 'pme'])
@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('device', DEVICES)
def test_self_only_system_has_graph_connected_zero_force(method, periodic, device):
    r = torch.tensor([[.1, .2, .3]], dtype=torch.float64, device=device, requires_grad=True)
    q = torch.tensor([[.2]], dtype=r.dtype, device=device, requires_grad=True)
    u = torch.full((1, 1, 3), .1, dtype=r.dtype, device=device, requires_grad=True)
    quad = torch.full((1, 1, 3, 3), .01, dtype=r.dtype, device=device, requires_grad=True)
    solver = Ewald(reciprocal_solver='blocked') if method == 'blocked' else PME(mesh=12)
    if periodic:
        out = solver.compute_potential_triclinic(r, q, torch.eye(3, dtype=r.dtype, device=device) * .5,
                                                 u=u, quad=quad, compute_field=True)
    else:
        out = solver.compute_potential_realspace(r, q, u=u, quad=quad, compute_field=True)
    force = -torch.autograd.grad(out['pot'].sum(), r, create_graph=True)[0]
    assert force.requires_grad
    gradients = torch.autograd.grad((force - .1).square().sum(), (q, u, quad))
    assert all(torch.count_nonzero(g) == 0 for g in gradients)


@pytest.mark.parametrize('shape', [(2,), (3, 3), (2, 2), (1, 1, 3)])
def test_batched_external_field_shape_is_validated(shape):
    solver = Ewald(reciprocal_solver='blocked')
    with pytest.raises(ValueError, match='e_ext must have shape'):
        solver(q=torch.ones(2), r=torch.zeros(2, 3), cell=torch.zeros(2, 3, 3),
               batch=torch.tensor([0, 1]), e_ext=torch.zeros(shape))


@pytest.mark.parametrize('method', ['blocked', 'pme'])
def test_batched_external_fields_allow_nonperiodic_cell_none(method):
    r, _, _, values = inputs()
    solver = PME(mesh=32) if method == 'pme' else Ewald(reciprocal_solver=method)
    fields = torch.stack((values['e_ext'], -2 * values['e_ext']))
    sources = {name: torch.cat((value, value), dim=0) for name, value in values.items() if name != 'e_ext'}
    actual = solver(r=torch.cat((r, r + .3)), cell=None,
                    batch=torch.tensor([0, 0, 0, 1, 1, 1]), e_ext=fields, **sources)
    single = [solver(r=r + offset, cell=None, e_ext=field,
                     **{name: value for name, value in values.items() if name != 'e_ext'})
              for offset, field in zip((0., .3), fields)]
    for index, value in enumerate(actual):
        torch.testing.assert_close(value, torch.cat([result[index] for result in single]))
