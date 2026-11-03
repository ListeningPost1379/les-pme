"""Regression tests for audited correctness fixes, edge cases, and physics consistency."""
import pytest
import torch

from les import Les
from les.module.atomicalpha import AtomicAlpha
from les.module.atomwise import Atomwise
from les.module.bec import BEC
from les.module.ewald import Ewald
from les.module.ewald_vectorized import Ewald_vectorized
from les.module.fixedcharges import FixedCharges
from les.module.make_kernels import bare_multipole_tensors
from les.module.pme import PME
from les.util.grad import grad


def test_ewald_vectorized_batched_e_ext_and_1d_shape_preservation():
    """Bug 1: Ewald_vectorized supports 2D [B, 3] e_ext and preserves 1D/2D shapes for q_induced/u_induced."""
    torch.manual_seed(101)
    r = torch.rand(6, 3, dtype=torch.float64) * 5.0
    q_1d = torch.tensor([0.3, -0.3, 0.2, -0.2, 0.4, -0.4], dtype=torch.float64)
    u_2d = torch.randn(6, 3, dtype=torch.float64) * 0.1
    kappa_1d = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.1, 0.3], dtype=torch.float64)
    alpha_1d = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.4, 0.9], dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    cell = torch.stack([torch.eye(3, dtype=torch.float64) * 8.0, torch.eye(3, dtype=torch.float64) * 9.0], dim=0)
    e_ext_2d = torch.tensor([[0.05, -0.02, 0.03], [-0.04, 0.06, -0.01]], dtype=torch.float64)

    for is_periodic in (True, False):
        cell_in = cell if is_periodic else torch.zeros_like(cell)
        ewald_ref = Ewald(dl=2.0, sigma=1.0, remove_self_interaction=True)
        ewald_vec = Ewald_vectorized(
            dl=2.0, sigma=1.0, remove_self_interaction=True,
            is_periodic=is_periodic, N_max=5
        )
        pot_ref, q_ind_ref, u_ind_ref = ewald_ref(
            q=q_1d, r=r, cell=cell_in if is_periodic else None, batch=batch,
            u=u_2d, kappa=kappa_1d, alpha=alpha_1d, e_ext=e_ext_2d
        )
        pot_vec, q_ind_vec, u_ind_vec = ewald_vec(
            q=q_1d, r=r, cell=cell_in, batch=batch,
            u=u_2d, kappa=kappa_1d, alpha=alpha_1d, e_ext=e_ext_2d
        )
        assert q_ind_vec.shape == (6, 1)
        assert u_ind_vec.shape == (6, 1, 3)
        torch.testing.assert_close(q_ind_vec.squeeze(1), q_ind_ref, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(u_ind_vec.squeeze(1), u_ind_ref, atol=1e-10, rtol=1e-10)

        # Also verify Les.forward with use_ewald_vectorization=True does not broadcast [N] + [N, 1] -> [N, N]
        les_vec = Les({
            "use_atomwise": False,
            "dl": 2.0,
            "sigma": 1.0,
            "remove_self_interaction": True,
            "use_ewald_vectorization": True,
            "is_periodic": is_periodic,
            "N_max": 5,
        })
        out = les_vec(
            positions=r, cell=cell_in, batch=batch,
            latent_charges=q_1d, latent_dipoles=u_2d,
            latent_kappas=kappa_1d, latent_alphas=alpha_1d, e_ext=e_ext_2d
        )
        assert out["latent_charges"].shape == (6,)
        assert out["latent_dipoles"].shape == (6, 3)


def test_mixed_1d_q_and_3d_u_across_solvers():
    """Bug 2: Split q_one_dim and u_two_dim so mixed 1D/2D q and 2D/3D u work in Ewald, PME, and Ewald_vectorized."""
    torch.manual_seed(102)
    r = torch.rand(4, 3, dtype=torch.float64) * 4.0
    q_1d = torch.tensor([0.2, -0.2, 0.1, -0.1], dtype=torch.float64)
    u_3d = torch.randn(4, 1, 3, dtype=torch.float64) * 0.1
    kappa_1d = torch.tensor([0.1, 0.2, 0.15, 0.25], dtype=torch.float64)
    alpha_2d = torch.tensor([[0.5], [0.6], [0.7], [0.8]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 8.0

    for solver in (
        Ewald(dl=2.0, sigma=1.0, reciprocal_solver="original"),
        Ewald(dl=2.0, sigma=1.0, reciprocal_solver="blocked"),
        PME(mesh=24, dl=2.0, sigma=1.0),
    ):
        res_pbc = solver.compute_potential_triclinic(r, q_1d, cell, u=u_3d, kappa=kappa_1d, alpha=alpha_2d)
        assert res_pbc["q_induced"].shape == (4,)
        assert res_pbc["u_induced"].shape == (4, 1, 3)

        res_rs = solver.compute_potential_realspace(r, q_1d, u=u_3d, kappa=kappa_1d, alpha=alpha_2d)
        assert res_rs["q_induced"].shape == (4,)
        assert res_rs["u_induced"].shape == (4, 1, 3)


def test_original_solver_compute_potential_only():
    """Bug 3: Ewald(reciprocal_solver='original') computes non-zero reciprocal phi when compute_potential=True alone."""
    torch.manual_seed(103)
    r = torch.rand(5, 3, dtype=torch.float64) * 5.0
    q = torch.tensor([0.5, -0.3, 0.2, -0.6, 0.2], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 7.5

    orig = Ewald(dl=2.0, sigma=1.0, reciprocal_solver="original")
    blk = Ewald(dl=2.0, sigma=1.0, reciprocal_solver="blocked")

    out_orig = orig.compute_potential_triclinic(r, q, cell, compute_potential=True, compute_field=False)
    out_blk = blk.compute_potential_triclinic(r, q, cell, compute_potential=True, compute_field=False)

    torch.testing.assert_close(out_orig["phi"], out_blk["phi"], atol=1e-12, rtol=1e-12)
    # Also verify phi == dE/dq
    q_req = q.clone().requires_grad_(True)
    pot = orig.compute_potential_triclinic(r, q_req, cell)["pot"]
    dE_dq = torch.autograd.grad(pot.sum(), q_req)[0]
    torch.testing.assert_close(out_orig["phi"].squeeze(-1), dE_dq, atol=1e-12, rtol=1e-12)


def test_non_monotonic_and_empty_graph_batch_in_ewald_pme_and_bec():
    """Bug 4: Non-monotonic batch ordering and empty graphs in batch preserve atom order and graph count."""
    torch.manual_seed(104)
    r = torch.rand(6, 3, dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.4, -0.1, -0.4, 0.1, 0.2, -0.2], dtype=torch.float64)
    kappa = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.15, 0.25], dtype=torch.float64)
    alpha = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.4, 0.9], dtype=torch.float64)

    # Interleaved batch: atoms [0, 2, 4] belong to graph 0, atoms [1, 3, 5] belong to graph 2, graph 1 is empty!
    batch_interleaved = torch.tensor([0, 2, 0, 2, 0, 2], dtype=torch.long)
    cell = torch.stack([
        torch.eye(3, dtype=torch.float64) * 8.0,
        torch.eye(3, dtype=torch.float64) * 8.5,
        torch.eye(3, dtype=torch.float64) * 9.0,
    ], dim=0)

    # Sorted reference for graph 0 and graph 2
    idx0 = torch.tensor([0, 2, 4])
    idx2 = torch.tensor([1, 3, 5])

    for solver in (
        Ewald(dl=2.0, sigma=1.0, reciprocal_solver="blocked"),
        PME(mesh=24, dl=2.0, sigma=1.0),
    ):
        pot, q_ind, u_ind = solver(q=q, r=r, cell=cell, batch=batch_interleaved, kappa=kappa, alpha=alpha)
        assert pot.shape == (3,)
        assert pot[1].item() == 0.0
        assert q_ind.shape == (6,)
        assert u_ind.shape == (6, 3)

        pot0, q0, u0 = solver(q=q[idx0], r=r[idx0], cell=cell[0:1], kappa=kappa[idx0], alpha=alpha[idx0])
        pot2, q2, u2 = solver(q=q[idx2], r=r[idx2], cell=cell[2:3], kappa=kappa[idx2], alpha=alpha[idx2])

        torch.testing.assert_close(pot[0], pot0[0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(pot[2], pot2[0], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(q_ind[idx0], q0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(q_ind[idx2], q2, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(u_ind[idx0], u0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(u_ind[idx2], u2, atol=1e-12, rtol=1e-12)

    # Test BEC with non-monotonic + empty graph batch
    bec_mod = BEC(remove_mean=True)
    q_dep = (r ** 2).sum(dim=-1)
    bec_interleaved = bec_mod(q=q_dep, r=r, cell=cell, batch=batch_interleaved)
    r0 = r[idx0].detach().clone().requires_grad_(True)
    r2 = r[idx2].detach().clone().requires_grad_(True)
    bec0 = bec_mod(q=(r0 ** 2).sum(dim=-1), r=r0, cell=cell[0:1])
    bec2 = bec_mod(q=(r2 ** 2).sum(dim=-1), r=r2, cell=cell[2:3])
    torch.testing.assert_close(bec_interleaved[idx0], bec0, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(bec_interleaved[idx2], bec2, atol=1e-12, rtol=1e-12)


def test_left_handed_periodic_cell_orientation_invariance():
    """Bug 5: Left-handed periodic cells (det(cell) < 0) are treated as periodic with V = |det(cell)|."""
    torch.manual_seed(105)
    frac = torch.rand(5, 3, dtype=torch.float64)
    q = torch.tensor([0.3, -0.2, 0.4, -0.1, -0.4], dtype=torch.float64)
    cell_rh = torch.diag(torch.tensor([8.0, 8.5, 9.0], dtype=torch.float64))
    # Swap first two lattice vectors -> det(cell_lh) = -det(cell_rh) < 0, same physical lattice & atom coords!
    cell_lh = cell_rh[[1, 0, 2], :]
    assert torch.linalg.det(cell_lh) < 0

    r = frac @ cell_rh

    for solver in (
        Ewald(dl=2.0, sigma=1.0, reciprocal_solver="original"),
        Ewald(dl=2.0, sigma=1.0, reciprocal_solver="blocked"),
        PME(mesh=24, dl=2.0, sigma=1.0),
    ):
        pot_rh, _, _ = solver(q=q, r=r, cell=cell_rh.unsqueeze(0))
        pot_lh, _, _ = solver(q=q, r=r, cell=cell_lh.unsqueeze(0))
        torch.testing.assert_close(pot_lh, pot_rh, atol=1e-10, rtol=1e-10)

    ewald_vec = Ewald_vectorized(dl=2.0, sigma=1.0, is_periodic=True, N_max=5)
    pot_vrh, _, _ = ewald_vec(q=q, r=r, cell=cell_rh.unsqueeze(0))
    pot_vlh, _, _ = ewald_vec(q=q, r=r, cell=cell_lh.unsqueeze(0))
    torch.testing.assert_close(pot_vlh, pot_vrh, atol=1e-10, rtol=1e-10)


def test_pme_auto_and_explicit_dtype_behavior():
    """Bug 6: PME(compute_dtype='auto') defaults periodic to float64 and handles mixed periodic/nonperiodic float32 batches without dtype mismatch; compute_dtype='float32' stays float32."""
    torch.manual_seed(106)
    r = torch.rand(4, 3, dtype=torch.float32) * 5.0
    q = torch.tensor([0.25, -0.25, 0.15, -0.15], dtype=torch.float32)
    kappa = torch.tensor([0.1, 0.1, 0.2, 0.2], dtype=torch.float32)
    alpha = torch.tensor([0.5, 0.5, 0.6, 0.6], dtype=torch.float32)
    cell = torch.eye(3, dtype=torch.float32) * 8.0

    pme_auto = PME(mesh=16, dl=2.0, sigma=1.0, compute_dtype="auto")
    out_auto = pme_auto.compute_potential_triclinic(r, q, cell, kappa=kappa, alpha=alpha, compute_potential=True, compute_field=True)
    assert out_auto["pot"].dtype == torch.float64

    # Mixed nonperiodic (float32) + periodic (float64) batch under auto must not fail on q_induced/u_induced assignment
    r_mix = torch.cat([r, r], dim=0)
    q_mix = torch.cat([q, q], dim=0)
    k_mix = torch.cat([kappa, kappa], dim=0)
    a_mix = torch.cat([alpha, alpha], dim=0)
    batch_mix = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    cells_mix = torch.stack([torch.zeros_like(cell), cell], dim=0)
    pot_mix, q_ind_mix, u_ind_mix = pme_auto(q=q_mix, r=r_mix, cell=cells_mix, batch=batch_mix, kappa=k_mix, alpha=a_mix)
    assert pot_mix.shape == (2,) and q_ind_mix.shape == (8,) and u_ind_mix.shape == (8, 3)

    pme_fp32 = PME(mesh=16, dl=2.0, sigma=1.0, compute_dtype="float32")
    out_fp32 = pme_fp32.compute_potential_triclinic(r, q, cell, kappa=kappa, alpha=alpha, compute_potential=True, compute_field=True)
    assert out_fp32["pot"].dtype == torch.float32
    assert out_fp32["q_induced"].dtype == torch.float32
    assert out_fp32["u_induced"].dtype == torch.float32
    assert out_fp32["phi"].dtype == torch.float32
    assert out_fp32["field"].dtype == torch.float32


def test_auxiliary_modules_regressions():
    """Bug 7: bare_multipole_tensors, FixedCharges 2D, AtomicAlpha 2D/4D + actinides, Atomwise FP64, BEC constant u."""
    torch.manual_seed(107)
    # 7a. bare_multipole_tensors shape, value, and finite backward on diagonal
    r = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    T1, T2, T3, T4 = bare_multipole_tensors(r)
    assert T1.shape == (2, 2, 3)
    assert T2.shape == (2, 2, 3, 3)
    assert T3.shape == (2, 2, 3, 3, 3)
    assert T4.shape == (2, 2, 3, 3, 3, 3)
    # At d = 2 along x: T2_xx = (3*1 - 1) / 2^3 = 2/8 = 0.25, T2_yy = -1 / 8 = -0.125
    torch.testing.assert_close(T2[0, 1, 0, 0], torch.tensor(0.25, dtype=torch.float64))
    torch.testing.assert_close(T2[0, 1, 1, 1], torch.tensor(-0.125, dtype=torch.float64))
    loss = T1.sum() + T2.sum() + T3.sum() + T4.sum()
    grad_r = torch.autograd.grad(loss, r)[0]
    assert torch.isfinite(grad_r).all()

    # 7b. FixedCharges with 2D latent_charges [N, 1] in Les
    les_fc = Les({"use_atomwise": False, "use_fixed_atomic_charges": True, "dl": 2.0, "sigma": 1.0})
    r4 = torch.rand(4, 3, dtype=torch.float64) * 5.0
    cell4 = (torch.eye(3, dtype=torch.float64) * 8.0).unsqueeze(0)
    z4 = torch.tensor([3, 9, 11, 17], dtype=torch.long)
    out_fc = les_fc(positions=r4, cell=cell4, latent_charges=torch.zeros(4, 1, dtype=torch.float64), atomic_numbers=z4)
    assert out_fc["latent_charges"].shape == (4, 1)

    # 7c. AtomicAlpha with 2D/4D latent_alphas and actinide Z=92 (Uranium)
    aa = AtomicAlpha()
    z_act = torch.tensor([8, 92, 94], dtype=torch.long)
    alpha_act = aa(z_act)
    assert alpha_act.shape == (3,) and (alpha_act > 0).all()

    les_aa = Les({"use_atomwise": False, "use_atomic_alpha": True, "dl": 2.0, "sigma": 1.0})
    for shape in ((3,), (3, 1), (3, 3, 3), (3, 1, 3, 3)):
        r3 = torch.rand(3, 3, dtype=torch.float64) * 5.0
        out_aa = les_aa(
            positions=r3, cell=cell4,
            latent_charges=torch.tensor([0.2, -0.4, 0.2], dtype=torch.float64),
            latent_alphas=torch.ones(shape, dtype=torch.float64) * 0.1,
            atomic_numbers=z_act,
        )
        assert torch.isfinite(out_aa["E_lr"]).all()

    # 7d. Atomwise with float64 descriptors
    aw = Atomwise(n_out=1, n_hidden=[8, 8])
    desc64 = torch.randn(5, 16, dtype=torch.float64)
    q64 = aw(desc64)
    assert q64.dtype == torch.float64 and q64.shape == (5, 1)

    # 7e. BEC with constant q / constant u (does not depend on r)
    r_bec = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
    q_const = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64)
    u_const = torch.zeros(3, 3, dtype=torch.float64)
    bec_out = BEC(remove_mean=False)(q=q_const, r=r_bec, cell=cell4, u=u_const)
    assert bec_out.shape == (3, 2, 3, 3) and torch.isfinite(bec_out).all()


def test_quadrupole_symmetry_and_blocked_zero_multipole_parity():
    """M-2 & P-1/P-4: Unsymmetrized quad gives identical results across original, blocked, and PME."""
    torch.manual_seed(108)
    r = torch.rand(5, 3, dtype=torch.float64, requires_grad=True) * 4.0
    q = torch.randn(5, 1, dtype=torch.float64, requires_grad=True)
    quad_unsym = torch.randn(5, 1, 3, 3, dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=torch.float64) * 7.5

    s_orig = Ewald(dl=2.0, sigma=1.0, reciprocal_solver="original")
    s_blk = Ewald(dl=2.0, sigma=1.0, reciprocal_solver="blocked")
    s_pme = PME(mesh=36, dl=2.0, sigma=1.0)

    out_orig = s_orig.compute_potential_triclinic(r, q, cell, quad=quad_unsym, compute_potential=True, compute_field=True)
    out_blk = s_blk.compute_potential_triclinic(r, q, cell, quad=quad_unsym, compute_potential=True, compute_field=True)
    out_pme = s_pme.compute_potential_triclinic(r, q, cell, quad=quad_unsym, compute_potential=True, compute_field=True)

    torch.testing.assert_close(out_blk["pot"], out_orig["pot"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out_blk["phi"], out_orig["phi"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out_blk["field"], out_orig["field"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out_pme["pot"], out_orig["pot"], atol=1e-4, rtol=1e-4)

    # Also verify real-space blocked with quad only (has_u=False, has_quad=True)
    rs_orig = s_orig.compute_potential_realspace(r, q, quad=quad_unsym, compute_field=True)
    rs_blk = s_blk.compute_potential_realspace(r, q, quad=quad_unsym, compute_field=True)
    torch.testing.assert_close(rs_blk["pot"], rs_orig["pot"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rs_blk["phi"], rs_orig["phi"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(rs_blk["field"], rs_orig["field"], atol=1e-12, rtol=1e-12)


def test_bec_eval_detaches_second_order_graph():
    """P1.4: BEC in eval() mode does not retain a second-order autograd graph, while train() mode does."""
    torch.manual_seed(109)
    r = torch.rand(4, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 1, dtype=torch.float64, requires_grad=True)
    cell = (torch.eye(3, dtype=torch.float64) * 8.0).unsqueeze(0)

    bec_mod = BEC(remove_mean=True)
    bec_mod.eval()
    q_eval = (r @ w).squeeze(-1)
    u_eval = r * w[0, 0]
    out_eval = bec_mod(q=q_eval, r=r, cell=cell, u=u_eval)
    assert not out_eval.requires_grad
    assert out_eval.grad_fn is None

    bec_mod.train()
    q_train = (r @ w).squeeze(-1)
    u_train = r * w[0, 0]
    out_train = bec_mod(q=q_train, r=r, cell=cell, u=u_train)
    assert out_train.requires_grad
    grad_w = torch.autograd.grad(out_train.square().sum(), w)[0]
    assert torch.isfinite(grad_w).all() and grad_w.abs().max() > 0


def test_torchscript_compilation_of_les_modules():
    """Verify Les (Ewald original/blocked, BEC) compiles cleanly with torch.jit.script without .tolist() type errors."""
    for solver in ("original", "blocked"):
        les_mod = Les({"use_atomwise": False, "reciprocal_solver": solver})
        assert les_mod.ewald.reciprocal_solver == solver
        scripted = torch.jit.script(les_mod)
        assert isinstance(scripted, torch.jit.ScriptModule)
