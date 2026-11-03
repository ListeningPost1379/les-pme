"""Full q/u/Q/iq/iu chain: MACE readouts, solver, force loss, optimizer and reload."""
from copy import deepcopy
import io

import pytest
import torch

pytest.importorskip('mace')
from les.tests.test_full_solvers import original  # Independent original, including its own kernels.
from mace.tools.scripts_utils import extract_model, get_optimizer, get_params_options
from mace.tools.torch_tools import default_dtype
from tests.golden.build_maceles_anchor import build_model
from tests.golden.build_maceles_anchor import ANCHOR_CONFIG
from tests.extensions.les.les_fixtures import atoms, batch
from tests.unit.test_optimizer_param_groups import _training_args


@pytest.mark.parametrize('method', ['blocked', 'pme'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('anisotropic', [False, True])
@pytest.mark.parametrize('ell', [1, 2])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))])
def test_all_readouts_train_and_reload_against_original(original, method, dtype, anisotropic, ell, device, monkeypatch):
    if ell == 2:
        monkeypatch.setitem(ANCHOR_CONFIG, 'hidden_irreps', '8x0e + 8x1o + 8x2e')
    options = dict(use_atomwise=False, use_dipole=True, use_quad=True,
                   use_induced_charge=True, use_induced_dipole=True,
                   use_anisotropic_polarizability=anisotropic,
                   make_alpha_positive=True, make_kappa_positive=True,
                   output_scale=.2, alpha_scale=.2, kappa_scale=.2,
                   reciprocal_solver=method, compute_dtype=str(dtype).split('.')[-1])
    if method == 'pme':
        options['pme_mesh'] = 32
    with default_dtype(dtype):
        model = build_model(options).to(device=device, dtype=dtype)
        data = batch(model, [atoms(False), atoms(True)])
        data = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                if torch.is_tensor(v) else v for k, v in data.items()}
        reference = deepcopy(model)
        reference.les.ewald = original()
        opt = get_optimizer(_training_args(), get_params_options(_training_args(), model))
        outputs, gradients = [], []
        for candidate in (model, reference):
            out = candidate(deepcopy(data), training=True, compute_stress=True)
            loss = ((out['forces'] - .01).square().sum() + (out['stress'] - .02).square().sum()
                    + .001 * out['energy'].square().sum())
            # Original real-space position Hessians contain NaNs, but parameter gradients are finite.
            grad = torch.autograd.grad(loss, tuple(candidate.parameters()), allow_unused=True)
            outputs.append(out)
            gradients.append(grad)
        tolerance = 4e-4 if dtype == torch.float32 else 1e-5
        for key in ('energy', 'forces', 'stress', 'latent_charges', 'latent_dipoles', 'latent_quads', 'latent_alphas'):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=tolerance, rtol=tolerance)
        for actual, expected in zip(*gradients):
            if actual is None or expected is None:
                assert actual is expected
            else:
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        before = {name: p.detach().clone() for name, p in model.named_parameters() if name.startswith('les_')}
        for p, g in zip(model.parameters(), gradients[0]):
            p.grad = g
        opt.step()
        for group in ('les_readouts', 'les_u_readouts', 'les_quad_1o_readouts', 'les_quad_2e_readouts',
                      'les_kappa_readouts', 'les_alpha_readouts', 'les_alpha_1o_readouts', 'les_alpha_2e_readouts'):
            names = [name for name in before if name.startswith(group + '.')]
            if names:
                assert any(not torch.equal(before[name], dict(model.named_parameters())[name]) for name in names), group
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        expected = model(deepcopy(data), training=True, compute_stress=True)
        for candidate in (torch.load(buffer, weights_only=False), extract_model(model, map_location=device)):
            assert candidate.get_les_arguments() == model.get_les_arguments()
            out = candidate(deepcopy(data), training=True, compute_stress=True)
            for key in ('energy', 'forces', 'stress'):
                torch.testing.assert_close(out[key], expected[key])


@pytest.mark.parametrize('method', ['blocked', 'pme'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))])
def test_standard_cli_trains_full_multipoles(tmp_path, method, device):
    import json
    import os
    import subprocess
    import sys
    import ase.io
    import numpy as np
    import yaml

    frames = []
    for index in range(6):
        structure = atoms(index % 2 == 0)
        structure.positions[1, 0] += .03 * index
        structure.info['REF_energy'] = -.1 + .01 * index
        structure.info['REF_stress'] = np.eye(3) * .001
        structure.arrays['REF_forces'] = np.full((2, 3), .01)
        frames.append(structure)
    ase.io.write(tmp_path/'train.xyz', frames[:4])
    ase.io.write(tmp_path/'valid.xyz', frames[4:])
    options = dict(use_atomwise=False, use_dipole=True, use_quad=True,
                   use_induced_charge=True, use_induced_dipole=True,
                   use_anisotropic_polarizability=True, make_alpha_positive=True,
                   make_kappa_positive=True, reciprocal_solver=method, compute_dtype='float64')
    if method == 'pme':
        options['pme_mesh'] = 32
    les_config = tmp_path/'les.yaml'
    les_config.write_text(yaml.safe_dump(options))
    config = dict(name='full', model='MACELES', train_file=str(tmp_path/'train.xyz'),
                  valid_file=str(tmp_path/'valid.xyz'), E0s='{1:0.,8:0.}',
                  r_max=3.5, hidden_irreps='4x0e+4x1o+4x2e', num_interactions=2,
                  num_radial_basis=4, correlation=1, radial_MLP='[8]', MLP_irreps='4x0e',
                  loss='stress', energy_key='REF_energy', forces_key='REF_forces', stress_key='REF_stress',
                  batch_size=2, valid_batch_size=2, max_num_epochs=1, eval_interval=1,
                  default_dtype='float32', device=device, seed=1379, num_workers=0,
                  lr=1e-4, save_cpu=True, plot=False, use_reduced_cg=False, gpu_memory_limit_gib=3.0,
                  les_arguments=str(les_config), **{key: str(tmp_path) for key in
                      ('work_dir','model_dir','checkpoints_dir','results_dir','log_dir')})
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    env = dict(os.environ, OMP_NUM_THREADS='4')
    completed = subprocess.run([sys.executable, '-m', 'mace.cli.run_train', '--config', str(path)],
                               env=env, capture_output=True, text=True, timeout=180)
    (tmp_path/'cli.stdout').write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-12000:]
    model = torch.load(tmp_path/'full.model', map_location='cpu', weights_only=False)
    assert model.get_les_arguments()['reciprocal_solver'] == method
    for flag in ('use_dipoles','use_quads','use_induced_charges','use_induced_dipoles'):
        assert getattr(model, flag)
    records = [json.loads(line) for line in (tmp_path/'full_run-1379_train.txt').read_text().splitlines()]
    assert any(row['mode'] == 'opt' for row in records)
    data = batch(model, [atoms(False), atoms(True)])
    with default_dtype(torch.float32):
        data = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in data.items()}
        out = model(data, training=True, compute_stress=True)
    assert torch.isfinite(out['energy']).all() and torch.isfinite(out['forces']).all()


@pytest.mark.parametrize('method', ['blocked', 'pme'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is unavailable'))])
def test_full_multipole_ase_energy_force_stress(method, device):
    import numpy as np
    from mace.calculators import MACECalculator
    options = dict(use_atomwise=False, use_dipole=True, use_quad=True,
                   use_induced_charge=True, use_induced_dipole=True,
                   make_alpha_positive=True, make_kappa_positive=True,
                   reciprocal_solver=method, compute_dtype='float64')
    if method == 'pme':
        options['pme_mesh'] = 32
    with default_dtype(torch.float64):
        model = build_model(options).to(device)
        calculator = MACECalculator(models=deepcopy(model), device=device, default_dtype='float64', require_les=True)
        for periodic in (False, True):
            structure = atoms(periodic)
            data = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch(model, [structure]).items()}
            reference = model(data, compute_stress=True)
            structure.calc = calculator
            np.testing.assert_allclose(structure.get_potential_energy(), reference['energy'].item(), rtol=2e-10, atol=2e-10)
            np.testing.assert_allclose(structure.get_forces(), reference['forces'].detach().cpu().numpy(), rtol=2e-10, atol=2e-10)
            if periodic:
                np.testing.assert_allclose(structure.get_stress(voigt=False), reference['stress'][0].detach().cpu().numpy(),
                                           rtol=2e-10, atol=2e-10)
