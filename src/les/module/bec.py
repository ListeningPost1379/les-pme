import torch
import torch.nn as nn
from typing import Dict, Optional

from ..util import grad

__all__ = ['BEC']

class BEC(nn.Module):
    def __init__(self,
                 remove_mean: bool = True,
                 epsilon_factor: float = 1., # \epsilon_infty
                 ):
        super().__init__()
        self.remove_mean = remove_mean
        self.epsilon_factor = epsilon_factor
        self.normalization_factor = epsilon_factor ** 0.5

    def forward(self,
                q: torch.Tensor,  # [n_atoms, n_q]
                r: torch.Tensor, # [n_atoms, 3]
                cell: torch.Tensor, # [batch_size, 3, 3]
                u: Optional[torch.Tensor] = None, # [n_atoms, 3]
                batch: Optional[torch.Tensor] = None,
                output_index: Optional[int] = None, # 0, 1, 2 to select only one component
                ) -> torch.Tensor:

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)

        n_node, n_q = q.shape

        if u is not None:
            if u.dim() == 2 and u.shape[1] == 3:
                u = u.unsqueeze(1)
            assert u.shape == (n_node, n_q, 3), 'u dimension error'

        if n_q > 1:
            q = q.sum(dim=1, keepdim=True)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        if batch is None:
            batch = torch.zeros(n, dtype=torch.int64, device=r.device)

        if cell is not None and cell.dim() == 3:
            n_graphs = int(cell.shape[0])
        elif batch.numel() > 0:
            n_graphs = int(batch.max().item()) + 1
        else:
            n_graphs = 0

        complex_dtype = self._complex_dtype(r.dtype)
        phases = torch.ones((n, 3), dtype=complex_dtype, device=r.device)

        # compute the polarization for each batch
        all_P = []
        all_P_u = []
        all_projections = []
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

        for i in range(n_graphs):
            mask = batch == i  # Create a mask for the i-th configuration
            if not mask.any():
                all_P.append(torch.zeros(3, dtype=complex_dtype, device=r.device))
                all_projections.append(torch.eye(3, device=r.device, dtype=r.dtype))
                if u is not None:
                    all_P_u.append(torch.zeros(3, dtype=r.dtype, device=r.device))
                continue

            r_now, q_now = r[mask], q[mask]

            if self.remove_mean:
                q_now = q_now - torch.mean(q_now, dim=0, keepdim=True)

            box_now = (cell[i] if cell.dim() == 3 else cell) if cell is not None else None

            # check if the box is periodic or not
            if box_now is None or not bool(is_periodic_cpu[i].item()):
                # the box is not periodic, we use the direct sum
                polarization = torch.sum(q_now * r_now, dim=0)
                phase = torch.ones_like(r_now, dtype=complex_dtype)
                projection = torch.eye(3, device=r_now.device, dtype=r_now.dtype)
            else:
                polarization, phase, projection = self.compute_pol_pbc(r_now, q_now, box_now)

            all_P.append(polarization * self.normalization_factor)
            phases[mask] = phase
            all_projections.append(projection)

            if u is not None:
                u_now = u[mask]
                polarization_u = u_now.sum((0, 1))
                all_P_u.append(polarization_u * self.normalization_factor)

        P = torch.stack(all_P, dim=0)

        # take the gradient of the polarization w.r.t. the positions to get the complex BEC
        # Grad returns P on 2nd index, BEC is defined with P on first index
        bec_basis_complex = grad(
            y=P, x=r, training=self.training, retain_graph=True
        ).transpose(1, 2).contiguous()
        if not self.training:
            phases = phases.detach()
        # dephase in the same reciprocal-lattice basis used to define each phase
        result_basis = (bec_basis_complex * phases.unsqueeze(2).conj()).real

        result = torch.zeros_like(result_basis)
        for i, projection in enumerate(all_projections):
            mask = batch == i
            if not mask.any():
                continue
            if not self.training:
                projection = projection.detach()
            projection = projection.to(device=result_basis.device, dtype=result_basis.dtype)
            result[mask] = torch.matmul(projection.unsqueeze(0), result_basis[mask])

        if output_index is not None:
            result = result[:, output_index, :]

        if u is not None:
            P_u = torch.stack(all_P_u, dim=0)
            result_u = grad(
                y=P_u, x=r, training=self.training, retain_graph=True
            ).transpose(1, 2).contiguous()
            if output_index is not None:
                result_u = result_u[:, output_index, :]
            return torch.stack([result, result_u], dim=1)
        return result

    def compute_pol_pbc(self, r_now, q_now, box_now):
        r_frac = torch.matmul(r_now, torch.linalg.inv(box_now))
        phase = torch.exp(1j * 2. * torch.pi * r_frac)
        S = torch.sum(q_now * phase, dim=0)
        polarization = S / (1j * 2. * torch.pi)
        projection = box_now.T
        return polarization.reshape(-1), phase, projection

    @staticmethod
    def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
        return torch.complex128 if dtype == torch.float64 else torch.complex64

    def __repr__(self):
        return f'BEC(remove_mean={self.remove_mean}, epsilon_factor={self.epsilon_factor})'
