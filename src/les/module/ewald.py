import torch
import torch.nn as nn
from itertools import product
from typing import Dict, Optional, Tuple, List

from les.module.make_kernels import make_kernels
from les.module.reciprocal import reciprocal_energy
from les.module.blocked import DEFAULT_DENSE_PAIRS, reciprocal_multipoles, real_multipoles

__all__ = ['Ewald']

class Ewald(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 remove_self_interaction=True,
                 norm_factor=90.4756,
                 use_epsilon_r_scaling=False,
                 reciprocal_solver='direct',
                 compute_dtype='auto',
                 dense_pairs=DEFAULT_DENSE_PAIRS,
                 ):
        super().__init__()
        if reciprocal_solver not in ('direct', 'original', 'blocked'):
            raise ValueError("Ewald reciprocal_solver must be 'direct', 'original' or 'blocked'")
        if compute_dtype not in ('auto', 'float32', 'float64'):
            raise ValueError("compute_dtype must be 'auto', 'float32' or 'float64'")
        if isinstance(dense_pairs, bool) or not isinstance(dense_pairs, int) or dense_pairs < 0:
            raise ValueError("dense_pairs must be a non-negative integer")
        self.reciprocal_solver = reciprocal_solver
        self.compute_dtype = compute_dtype
        # blocked/pme: graphs with at most this many pairs (or atom-mode pairs)
        # skip recomputation; 0 always streams with bounded storage.
        self.dense_pairs = dense_pairs
        self.dl = dl
        self.sigma = sigma
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.52635*10^{-3} e^2 eV^{-1} A^{-1}
        self.norm_factor = norm_factor
        self.k_sq_max = (self.twopi / self.dl) ** 2
        self.use_epsilon_r_scaling = use_epsilon_r_scaling

    def __setstate__(self, state):
        state.setdefault('reciprocal_solver', getattr(type(self), 'reciprocal_solver', 'direct'))
        state.setdefault('compute_dtype', 'auto')
        state.setdefault('dense_pairs', DEFAULT_DENSE_PAIRS)
        super().__setstate__(state)

    def _cast(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.compute_dtype == 'float32':
            return tensor.float()
        if self.compute_dtype == 'float64':
            return tensor.double()
        return tensor

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q] or [n_atoms]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                batch: Optional[torch.Tensor] = None,
                u: Optional[torch.Tensor] = None, # [n_atoms, n_q, 3] or [natoms, 3]
                quad: Optional[torch.Tensor] = None, # [natoms,3,3]
                kappa: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms]
                alpha: Optional[torch.Tensor] = None, # [n_atoms, n_q] or [n_atoms, n_q, 3, 3] or [n_atoms] or [n_atoms, 3, 3]
                e_ext: Optional[torch.Tensor] = None,
                compute_field: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'
        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        if cell is not None and cell.dim() == 3:
            n_graphs = int(cell.shape[0])
        elif e_ext is not None and e_ext.dim() == 2:
            n_graphs = int(e_ext.shape[0])
        elif batch.numel() > 0:
            n_graphs = int(batch.max().item()) + 1
        else:
            n_graphs = 0

        if e_ext is not None:
            if e_ext.dim() == 1:
                if e_ext.shape[0] != 3:
                    raise ValueError('e_ext must have shape [3] or [batch_size, 3]')
            elif e_ext.dim() == 2:
                if e_ext.shape[0] != n_graphs or e_ext.shape[1] != 3:
                    raise ValueError('e_ext must have shape [3] or [batch_size, 3]')
            else:
                raise ValueError('e_ext must have shape [3] or [batch_size, 3]')

        if cell is None:
            is_periodic_cpu = torch.zeros(n_graphs, dtype=torch.bool, device="cpu")
        elif cell.dim() == 3:
            is_periodic_cpu = (torch.abs(torch.linalg.det(cell.detach())) >= 1e-6).to(
                device="cpu", dtype=torch.bool
            )
        else:
            is_periodic_cpu = (
                (torch.abs(torch.linalg.det(cell.detach())) >= 1e-6)
                .to(device="cpu", dtype=torch.bool)
                .expand(n_graphs)
            )

        results: List[torch.Tensor] = []
        q_induced_all: Optional[torch.Tensor] = None
        u_induced_all: Optional[torch.Tensor] = None
        for i in range(n_graphs):
            mask = batch == i  # Create a mask for the i-th configuration
            if not mask.any():
                results.append(torch.zeros(1, device=r.device, dtype=self._cast(r).dtype))
                continue
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now = r[mask], q[mask]

            u_now = u[mask] if u is not None else None
            quad_now = quad[mask] if quad is not None else None
            kappa_now = kappa[mask] if kappa is not None else None
            alpha_now = alpha[mask] if alpha is not None else None
            box_now = (cell[i] if cell.dim() == 3 else cell) if cell is not None else None
            field_now = e_ext
            if e_ext is not None and e_ext.dim() == 2:
                field_now = e_ext[i]

            # check if the box is periodic or not
            if box_now is None or not bool(is_periodic_cpu[i].item()):
                # the box is not periodic, we use the direct sum
                result = self.compute_potential_realspace(r_raw=r_raw_now, q=q_now, u=u_now,
                                                          quad=quad_now,
                                                          kappa=kappa_now,
                                                          alpha=alpha_now,
                                                          compute_field=compute_field,
                                                          e_ext=field_now,
                                                          )
            else:
                # the box is periodic, we use the reciprocal sum
                result = self.compute_potential_triclinic(r_raw=r_raw_now, q=q_now,
                                                          cell_now=box_now, u=u_now, quad=quad_now,
                                                          kappa=kappa_now, alpha=alpha_now,
                                                          compute_field=compute_field,
                                                          e_ext=field_now,
                                                          )
            results.append(result['pot'])
            q_i = result['q_induced']
            if q_induced_all is None:
                q_induced_all = torch.zeros([n] + list(q_i.shape[1:]), device=q_i.device, dtype=q_i.dtype)
            q_induced_all[mask] = q_i.to(q_induced_all.dtype)
            u_i = result['u_induced']
            if u_induced_all is None:
                u_induced_all = torch.zeros([n] + list(u_i.shape[1:]), device=u_i.device, dtype=u_i.dtype)
            u_induced_all[mask] = u_i.to(u_induced_all.dtype)

        if q_induced_all is None:
            q_induced_all = torch.empty((0,), device=r.device, dtype=r.dtype)
        if u_induced_all is None:
            u_induced_all = torch.empty((0,), device=r.device, dtype=r.dtype)
        pot_out = torch.cat(results) if len(results) > 0 else torch.zeros(0, device=r.device, dtype=r.dtype)
        return pot_out, q_induced_all, u_induced_all

    def compute_potential_realspace(self, r_raw, q,
                                    u: Optional[torch.Tensor]=None,
                                    quad: Optional[torch.Tensor]=None,
                                    kappa: Optional[torch.Tensor]=None,
                                    alpha: Optional[torch.Tensor]=None,
                                    compute_field: bool=False,
                                    e_ext: Optional[torch.Tensor]=None
                                    ):

        # this is 1/(4pi epsilon_0)
        norm_const = self.norm_factor / self.twopi

        r_raw, q = self._cast(r_raw), self._cast(q)
        u = self._cast(u) if u is not None else None
        quad = self._cast(quad) if quad is not None else None
        kappa = self._cast(kappa) if kappa is not None else None
        alpha = self._cast(alpha) if alpha is not None else None
        e_ext = self._cast(e_ext) if e_ext is not None else None

        q_one_dim = (q.dim() == 1)
        u_two_dim = (u.dim() == 2 and u.shape[1] == 3) if u is not None else (
            q_one_dim and (alpha is None or alpha.dim() in (1, 3))
        )
        if q_one_dim:
            q = q.unsqueeze(1)
        q = q.to(r_raw.dtype)
        n_node, n_q = q.shape
        device = r_raw.device

        e_field = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)
        q_induced = torch.zeros((n_node, n_q), device=device, dtype=r_raw.dtype)
        u_induced = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        if u is not None:
            u = u.to(r_raw.dtype)
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        if quad is not None:
            quad = quad.to(r_raw.dtype)
            if quad.dim() == 3 and quad.shape[1] == 3:
                quad = quad.unsqueeze(1)
            assert quad.shape == (n_node, n_q, 3, 3), 'quad dimension error'
            quad = 0.5 * (quad + quad.transpose(-1, -2))

        if not torch.jit.is_scripting():
            if self.reciprocal_solver in ('blocked', 'pme'):
                if torch.is_autocast_enabled(r_raw.device.type):
                    raise ValueError('blocked real-space sum requires autocast disabled')
                pot, phi, field = real_multipoles(r_raw, q, u, quad, self.sigma, norm_const,
                                                 compute_field or alpha is not None,
                                                 dense_pairs=self.dense_pairs)
                out = self._finish_multipoles(pot, phi, field, q, u, quad, kappa, alpha,
                                             e_ext, q_one_dim, False, u_two_dim=u_two_dim)
                if not compute_field and alpha is None:
                    out['field'] = torch.zeros_like(field)
                return out

        # f_qu/f_uu also feed the charge-induced e_field when compute_field
        # or alpha is set, so compute_u covers those cases too.
        compute_u = (u is not None) or compute_field or (alpha is not None)
        compute_Q = quad is not None
        f_qq, f_qu, f_uu, f_Qu, f_QQ = make_kernels(
            r_raw, self.sigma, norm_const,
            compute_u=compute_u,
            compute_Q=compute_Q,
        )

        # electric potential at r_j due to q at r_i, sum over i
        e_phi = torch.einsum('iq,ij->jq', q, f_qq)
        pot = 0.5 * torch.einsum('iq,iq->q', e_phi, q)

        if u is not None:
            assert f_qu is not None
            e_phi_u = torch.einsum('iqc,ijc->jq', u, f_qu)
            e_phi = e_phi + e_phi_u
            pot_qu = torch.einsum('iq,iq->q', e_phi_u, q)
            pot += pot_qu

            # field at j induced by dipoles at all i (kernel excludes i=j via mask_off)
            assert f_uu is not None
            E_u = torch.einsum('ijcd,iqc->jqd', f_uu, u)
            pot_uu = -0.5 * torch.einsum('iqc,iqc->q', u, E_u)
            pot += pot_uu
        else:  # for torchscript compatibility
            E_u = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        if quad is not None:
            # potential and field at j induced by quadrupoles at all i
            # (sign convention matches the triclinic structure factor S_Q = -(1/2)(k·Q·k))
            assert f_uu is not None
            e_phi_Q = 0.5 * torch.einsum('iqab,ijab->jq', quad, f_uu)
            assert f_Qu is not None
            E_Q = 0.5 * torch.einsum('iqab,ijabc->jqc', quad, f_Qu)
            e_phi = e_phi + e_phi_Q

            pot_Qq = torch.einsum('iq,iq->q', q, e_phi_Q)
            assert f_QQ is not None
            pot_QQ = 0.125 * torch.einsum('iqab,ijabcd,jqcd->q', quad, f_QQ, quad)
            pot = pot + pot_Qq + pot_QQ
            if u is not None:
                pot_Qu = -torch.einsum('iqc,iqc->q', u, E_Q)
                pot = pot + pot_Qu
        else:  # for torchscript compatibility
            E_Q = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        # note this is the *opposite* behavior of triclinic -- triclinic gets the self-interaction automatically
        if not self.remove_self_interaction:
            pot += (q ** 2).sum(dim=0) / (self.sigma * self.twopi**1.5) * self.norm_factor
            e_phi = e_phi + q * (2 / (self.sigma * self.twopi**1.5)) * self.norm_factor
            if u is not None:
                pot += (u**2).sum(dim=(0,2)) / ( 3 * self.sigma**3. * self.twopi**1.5) * self.norm_factor
                a = 1.0 / (self.sigma * (2.0 ** 0.5))
                c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) * norm_const
                E_u = E_u - c_self * u
            if quad is not None:
                pot += (quad**2).sum(dim=(0,2,3)) / (10 * self.sigma**5. * self.twopi**1.5) * self.norm_factor

        # for computing induced charges
        if kappa is not None:
            q_induced = self._get_induced_q(e_phi, kappa)
            pot_induced = 0.5 * (e_phi * q_induced).sum(dim=0)
            pot += pot_induced

        # for computing electric field
        if compute_field or alpha is not None:
            assert f_qu is not None
            e_field = torch.einsum('iq,ijc->jqc', q, f_qu)

            if u is not None:
                e_field = e_field + E_u

            if quad is not None:
                e_field = e_field + E_Q

            if alpha is not None:
                u_induced = self._get_induced_u(e_field, alpha, e_ext)
                pot_u_induced = - 0.5 * (e_field * u_induced).sum(dim=(0,2)) # [n_q]
                pot += pot_u_induced

        output = {
                 'pot': pot.sum().view(-1),
                 'q_induced': q_induced.squeeze(dim=1) if q_one_dim else q_induced,
                 'u_induced': u_induced.squeeze(dim=1) if u_two_dim else u_induced,
                 'phi': e_phi,
                 'field': e_field,
                 }
        return output

    def _get_induced_q(self, e_phi, kappa):
        if kappa.dim() == 1:
            kappa = kappa.unsqueeze(1)
        assert kappa.dim() == 2, 'kappa dimension error'
        q_induced = - kappa * e_phi # [n, n_q]
        return q_induced

    @torch.jit.unused
    def _finish_multipoles(self, pot, phi, field, q, u, quad, kappa, alpha,
                          e_ext, one_dim, periodic, volume=None, u_two_dim: Optional[bool] = None):
        """Apply original self conventions and the response to the complete source sum."""
        if u_two_dim is None:
            u_two_dim = one_dim
        sign = -1 if periodic and self.remove_self_interaction else (
            1 if not periodic and not self.remove_self_interaction else 0)
        charge_self = self.norm_factor / (self.sigma * self.twopi**1.5)
        if sign:
            pot = pot + sign * charge_self * q.square().sum(0)
            phi = phi + sign * 2 * charge_self * q
            if u is not None:
                dipole_self = charge_self / (3 * self.sigma**2)
                pot = pot + sign * dipole_self * u.square().sum((0, 2))
                field = field - sign * 2 * dipole_self * u
            if quad is not None:
                pot = pot + sign * charge_self / (10 * self.sigma**4) * quad.square().sum((0, 2, 3))
        qi, ui = torch.zeros_like(q), q.new_zeros(*q.shape, 3)
        if kappa is not None:
            qi = self._get_induced_q(phi, kappa)
            pot = pot + .5 * (phi * qi).sum(0)
        if alpha is not None:
            ui = self._get_induced_u(field, alpha, e_ext)
            pot = pot - .5 * (field * ui).sum((0, 2))
        output = {'pot': pot.sum().view(-1), 'phi': phi, 'field': field,
                  'q_induced': qi.squeeze(1) if one_dim else qi,
                  'u_induced': ui.squeeze(1) if u_two_dim else ui}
        if periodic:
            output['epsilon_r'] = self._get_epsilon_r(alpha, volume) if (
                alpha is not None and self.use_epsilon_r_scaling) else q.new_ones(q.shape[1])
        return output

    def _get_induced_u(self, e_field, alpha, e_ext: Optional[torch.Tensor]=None):
        if e_ext is not None:
            e_field = e_field + e_ext[None,None,:]
        if alpha.dim() == 1 or (alpha.dim() == 3 and alpha.shape[1:3] == (3,3)):
            alpha = alpha.unsqueeze(1)
        if alpha.dim() == 2:
            u_induced = e_field * alpha.unsqueeze(2) # [n, n_q, 3]
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3,3):
            # e_field: [n, n_q, 3], alpha: [n, n_q, 3, 3]
            u_induced = torch.einsum('iqc,iqcd->iqd', e_field, alpha)
        else:
            raise ValueError('alpha dimension error')
        return u_induced

    def _get_epsilon_r(self, alpha, volume):
        epsilon_0 = 0.00552635  # e^2 eV^{-1} A^{-1}
        if alpha.dim() == 1 or (alpha.dim() == 3 and alpha.shape[1:3] == (3,3)):
            alpha = alpha.unsqueeze(1)
        if alpha.dim() == 2: # isotropic alpha
            epsilon_r = alpha.sum(dim=0) / volume / epsilon_0 + 1.
        elif alpha.dim() == 4 and alpha.shape[2:4] == (3,3): # anisotropic alpha
            epsilon_r = torch.einsum('iqcc->q', alpha) / 3. / volume / epsilon_0 + 1.
        else:
            raise ValueError('alpha dimension error')
        return epsilon_r

    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now,
                                    u: Optional[torch.Tensor]=None,
                                    quad: Optional[torch.Tensor]=None,
                                    kappa:Optional[torch.Tensor]=None,
                                    alpha:Optional[torch.Tensor]=None,
                                    compute_potential:bool =False,
                                    compute_field: bool=False,
                                    e_ext: Optional[torch.Tensor]=None):
        r_raw, q, cell_now = self._cast(r_raw), self._cast(q), self._cast(cell_now)
        u = self._cast(u) if u is not None else None
        quad = self._cast(quad) if quad is not None else None
        kappa = self._cast(kappa) if kappa is not None else None
        alpha = self._cast(alpha) if alpha is not None else None
        e_ext = self._cast(e_ext) if e_ext is not None else None
        device = r_raw.device
        q_one_dim = (q.dim() == 1)
        u_two_dim = (u.dim() == 2 and u.shape[1] == 3) if u is not None else (
            q_one_dim and (alpha is None or alpha.dim() in (1, 3))
        )
        if q_one_dim:
            q = q.unsqueeze(1)
        n_node, n_q = q.shape

        # pre-allocate tensors for torchscript compatibility
        e_phi = torch.zeros((n_node, n_q), device=device, dtype=r_raw.dtype)
        e_field = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)
        q_induced = torch.zeros((n_node, n_q), device=device, dtype=r_raw.dtype)
        u_induced = torch.zeros((n_node, n_q, 3), device=device, dtype=r_raw.dtype)

        if u is not None:
            u = u.to(r_raw.dtype)
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        if quad is not None:
            if quad.dim() == 3 and quad.shape[1] == 3:
                quad = quad.unsqueeze(1)
            assert quad.shape == (n_node, n_q, 3, 3), 'quad dimension error'
            quad = 0.5 * (quad + quad.transpose(-1, -2))

        volume = torch.abs(torch.det(cell_now))
        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        if alpha is not None and hasattr(self, 'use_epsilon_r_scaling') and self.use_epsilon_r_scaling:
            epsilon_r = self._get_epsilon_r(alpha, volume) #[n_q]
        else:
            epsilon_r = torch.ones(n_q, device=device, dtype=r_raw.dtype)

        # max Nk for each axis
        norms_cpu = (torch.norm(cell_now.detach(), dim=1) / self.dl).to(
            device="cpu", dtype=torch.int64
        )
        Nk = [
            max(1, int(norms_cpu[0].item())),
            max(1, int(norms_cpu[1].item())),
            max(1, int(norms_cpu[2].item())),
        ]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3).to(G.dtype)
        kvec = nvec @ G  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]

        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)
        if self.compute_dtype != 'auto':
            factors = factors.to(r_raw.dtype)

        if not torch.jit.is_scripting():
            if self.reciprocal_solver == 'blocked':
                if (torch.is_autocast_enabled(r_raw.device.type)
                    or (r_raw.is_cuda and r_raw.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32)):
                    raise ValueError('blocked requires autocast and FP32 TF32 disabled')
                weight = factors * torch.exp(-self.sigma_sq_half * k_sq) / k_sq / volume * self.norm_factor
                potential, field = compute_potential or kappa is not None, compute_field or alpha is not None
                if u is None and quad is None and not potential and not field:
                    pot = reciprocal_energy(r_raw, q, kvec, weight)
                else:
                    pot, e_phi, e_field = reciprocal_multipoles(r_raw, q, u, quad, kvec, weight, potential, field,
                                                               dense_pairs=self.dense_pairs)
                out = self._finish_multipoles(pot, e_phi, e_field, q, u, quad, kappa, alpha,
                                             e_ext, q_one_dim, True, volume, u_two_dim=u_two_dim)
                if not potential:
                    out['phi'] = torch.zeros_like(e_phi)
                if not field:
                    out['field'] = torch.zeros_like(e_field)
                return out
            eligible = (u is None and quad is None and kappa is None and alpha is None
                    and not compute_potential and not compute_field and q.dtype == r_raw.dtype
                    and not torch.is_autocast_enabled(r_raw.device.type)
                    and not (r_raw.is_cuda and r_raw.dtype == torch.float32
                             and torch.backends.cuda.matmul.allow_tf32))
            if self.reciprocal_solver != 'original' and eligible:
                weight = factors * torch.exp(-self.sigma_sq_half * k_sq) / k_sq / volume * self.norm_factor
                pot = reciprocal_energy(r_raw, q, kvec, weight)
                if self.remove_self_interaction:
                    pot = pot - q.square().sum(0) / (self.sigma * self.twopi**1.5) * self.norm_factor
                return {'pot': pot.sum().view(-1), 'q_induced': q_induced.squeeze(1) if q_one_dim else q_induced,
                        'u_induced': u_induced.squeeze(1) if u_two_dim else u_induced,
                        'phi': e_phi, 'field': e_field, 'epsilon_r': epsilon_r}
        elif self.reciprocal_solver == 'blocked':
            raise ValueError('blocked requires eager execution; select original for TorchScript')

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        # exp_ikr = torch.exp(1j * k_dot_r) # [n, M]
        # S_k = (q.unsqueeze(2) * exp_ikr.unsqueeze(1)).sum(dim=0) # [n_q, M]
        cos_kr = torch.cos(k_dot_r) # [n, M]
        sin_kr = torch.sin(k_dot_r) # [n, M]
        S_k_real = (q.unsqueeze(2) * cos_kr.unsqueeze(1)).sum(dim=0) # [n_q, M]
        S_k_imag = (q.unsqueeze(2) * sin_kr.unsqueeze(1)).sum(dim=0) # [n_q, M]

        if u is not None:
            uk = u @ kvec.T # [n, n_q, 3] @ [M, 3] -> [n_node, n_q, M]
            S_k_real_u = - (uk * sin_kr.unsqueeze(1)).sum(dim=0) # [n_q, M]
            S_k_real = S_k_real + S_k_real_u
            S_k_imag_u = (uk * cos_kr.unsqueeze(1)).sum(dim=0)
            S_k_imag = S_k_imag + S_k_imag_u

        if quad is not None:
            kk = (kvec[:, :, None] * kvec[:, None, :]).reshape(kvec.shape[0], 9)
            qk2 = (quad.reshape(n_node * n_q, 9) @ kk.T).reshape(n_node, n_q, kvec.shape[0])
            S_k_real_Q = -0.5 * (qk2 * cos_kr.unsqueeze(1)).sum(dim=0)
            S_k_real = S_k_real + S_k_real_Q
            S_k_imag_Q = -0.5 * (qk2 * sin_kr.unsqueeze(1)).sum(dim=0)
            S_k_imag = S_k_imag + S_k_imag_Q

        S_k_sq = S_k_real**2 + S_k_imag**2  # [n_q, M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq

        # Compute potential energy, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume * self.norm_factor # [n_q]

        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            pot -= torch.sum(q**2, dim=0) / (self.sigma * self.twopi**1.5) * self.norm_factor
            if u is not None:
                pot -= torch.sum(u**2, dim=(0,2)) / ( 3 * self.sigma**3. * self.twopi**1.5) * self.norm_factor
            if quad is not None:
                pot -= torch.sum(quad**2, dim=(0,2,3)) / (10 * self.sigma**5. * self.twopi**1.5) * self.norm_factor

        # for computing electric field or potential
        if compute_potential or compute_field or kappa is not None or alpha is not None:
            #S_k = S_k_real + 1j * S_k_imag
            #exp_ikr = cos_k_dot_r + 1j * sin_k_dot_r
            # sk_field = 2 * kfac * torch.conj(S_k)   # [n_q, M]
            prefactor = (factors * 2.0 * kfac) / volume * self.norm_factor # [M]
        else:
            prefactor = torch.zeros(kvec.shape[0], device=device, dtype=kvec.dtype)

        # for computing electric potential
        if compute_potential or kappa is not None:
            # real part of exp(-ik*r) * S(k) is the contribution to the potential,
            # Real part -> cos(k*r)*S_real + sin(k*r)*S_imag
            term_real = S_k_real.unsqueeze(0) * cos_kr.unsqueeze(1) + S_k_imag.unsqueeze(0) * sin_kr.unsqueeze(1) # [n, n_q, M]
            e_phi = (prefactor.unsqueeze(0) * term_real).sum(dim=2) # [n, n_q]

            if self.remove_self_interaction:
                e_phi -= q * (2 / (self.sigma * self.twopi**1.5)) * self.norm_factor # [n, n_q]

            if kappa is not None: # compute induced charges
                q_induced = self._get_induced_q(e_phi, kappa)
                pot_induced = 0.5 * (e_phi * q_induced).sum(dim=0) # [n_q]
                pot += pot_induced

        # for computing electric field
        if compute_field or alpha is not None:
            # imaginary part of exp(-ik*r) * S(k) contributes to the field
            # Imaginary part -> cos(k*r)*S_imag - sin(k*r)*S_real
            term_imag = S_k_real.unsqueeze(0) * sin_kr.unsqueeze(1) - S_k_imag.unsqueeze(0) * cos_kr.unsqueeze(1) # [n, n_q, M]
            e_field = (prefactor.unsqueeze(0).unsqueeze(0).unsqueeze(3)
                       * term_imag.unsqueeze(3) * kvec.unsqueeze(0).unsqueeze(0)).sum(dim=2) # [n, n_q, 3]

            if self.remove_self_interaction and u is not None:
                a = 1.0 / (self.sigma * (2.0 ** 0.5))
                c_self = (4.0 / (3.0 * torch.pi**0.5)) * (a**3) / self.twopi * self.norm_factor
                e_field += c_self * u

            # compute induced dipoles
            if alpha is not None:
                u_induced = self._get_induced_u(e_field, alpha, e_ext)
                pot_induced = - 0.5 * (e_field * u_induced).sum(dim=(0,2)) # [n_q]
                pot += pot_induced

        output = {
                 'pot': pot.sum().view(-1), # sum over the energy contributions from different nq channels
                 'q_induced': q_induced.squeeze(dim=1) if q_one_dim else q_induced,
                 'u_induced': u_induced.squeeze(dim=1) if u_two_dim else u_induced,
                 'phi': e_phi,
                 'field': e_field,
                 'epsilon_r': epsilon_r,
                 }
        return output

    def __repr__(self):
        return f"Ewald(dl={self.dl}, sigma={self.sigma}, remove_self_interaction={self.remove_self_interaction})"
