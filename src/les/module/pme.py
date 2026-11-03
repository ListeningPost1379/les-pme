"""Differentiable order-6 PME for the original LES multipole operator."""

import math
from numbers import Integral
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .ewald import Ewald


def fft_friendly_dimension(min_dim: int) -> int:
    """Round up to the FFT-friendly ladder {4, 5, 6, 8} * 2^k (64, 80, 96, 128, 160, 192, 256, ...)."""
    min_n = int(min_dim)
    if min_n <= 6:
        return 6
    small_dims = (6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)
    for d in small_dims:
        if d >= min_n:
            return d
    base = 16
    while 8 * base < min_n:
        base *= 2
    if 4 * base >= min_n:
        return 4 * base
    if 5 * base >= min_n:
        return 5 * base
    if 6 * base >= min_n:
        return 6 * base
    return 8 * base


PME_AUTO_MIN_DIM = 64
"""Per-axis floor of the automatic (``mesh='auto'``) PME grid."""


def recommend_pme_mesh_dimension(
    length: float,
    dl: float = 2.0,
    min_dim: int = PME_AUTO_MIN_DIM,
) -> int:
    """Choose an FFT-friendly centered order-6 B-spline PME grid dimension for a cell axis."""
    length_f = float(length)
    if not math.isfinite(length_f) or length_f < 0.0:
        raise ValueError("Cell length must be finite and non-negative")
    dl_f = float(dl)
    if not math.isfinite(dl_f) or dl_f <= 0.0:
        raise ValueError("PME dl must be finite and positive")
    min_dim_i = max(6, int(min_dim))
    if length_f == 0.0:
        return fft_friendly_dimension(min_dim_i)
    cutoff_modes = length_f / dl_f
    bound = max(1, int(math.floor(cutoff_modes)))
    min_oversampled = int(math.ceil(3.5 * cutoff_modes))
    target = max(min_dim_i, 2 * bound + 2, min_oversampled)
    return fft_friendly_dimension(target)


def recommend_pme_mesh(
    cell: Union[torch.Tensor, np.ndarray, Sequence[Sequence[float]], Sequence[float]],
    dl: float = 2.0,
    min_dim: Union[int, Sequence[int]] = PME_AUTO_MIN_DIM,
    bounds: Optional[Sequence[int]] = None,
) -> Tuple[int, int, int]:
    """Recommend a 3D PME mesh shape (nx, ny, nz) for a periodic cell."""
    if isinstance(cell, torch.Tensor):
        c = cell.detach().cpu().to(dtype=torch.float64)
        if c.ndim == 3 and c.shape[0] == 1:
            c = c[0]
        if c.shape == (3, 3):
            lengths = torch.linalg.norm(c, dim=-1).tolist()
        elif c.numel() == 3:
            lengths = [float(x) for x in c.reshape(3).tolist()]
        else:
            raise ValueError(f"Expected cell of shape (3, 3) or (3,), got {tuple(c.shape)}")
    else:
        arr = np.asarray(cell, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape == (3, 3):
            lengths = np.linalg.norm(arr, axis=-1).tolist()
        elif arr.size == 3:
            lengths = [float(x) for x in arr.reshape(3).tolist()]
        else:
            raise ValueError(f"Expected cell of shape (3, 3) or (3,), got {arr.shape}")

    if isinstance(min_dim, Integral):
        min_dims = (int(min_dim), int(min_dim), int(min_dim))
    else:
        min_dims_tuple = tuple(int(x) for x in min_dim)
        if len(min_dims_tuple) != 3:
            raise ValueError("min_dim must be an integer or a 3-element sequence")
        min_dims = (min_dims_tuple[0], min_dims_tuple[1], min_dims_tuple[2])

    if bounds is not None:
        if len(bounds) != 3:
            raise ValueError("bounds must be a 3-element sequence")
        min_dims = tuple(max(m, 2 * int(b) + 2) for m, b in zip(min_dims, bounds))  # type: ignore[assignment]

    return (
        recommend_pme_mesh_dimension(lengths[0], dl=dl, min_dim=min_dims[0]),
        recommend_pme_mesh_dimension(lengths[1], dl=dl, min_dim=min_dims[1]),
        recommend_pme_mesh_dimension(lengths[2], dl=dl, min_dim=min_dims[2]),
    )


def _spline(x, order):
    y = x + .5 * order
    values = [((y - j >= 0) & (y - j < 1)).to(x.dtype) for j in range(order)]
    for degree in range(2, order + 1):
        values = [((y - j) * values[j] + (degree - y + j) * values[j + 1]) / (degree - 1)
                  for j in range(len(values) - 1)]
    return values[0]


class PME(Ewald):
    """Legacy LES kernel on an order-6 B-spline PME mesh, including force-loss backward.

    Nonperiodic graphs use the blocked direct sum. Mesh size controls
    approximation error, separately from dl, for q/u/Q and single-response iq/iu.
    ``mesh="auto"`` (the default, also ``None``) chooses the dimensions from the
    periodic cell lengths and ``dl`` on every call (>= ``PME_AUTO_MIN_DIM`` = 64
    along each axis, rounded up to an FFT-friendly 5-smooth even integer). An
    integer or three integers always select exactly that fixed mesh; ``64`` means
    a fixed 64^3 grid that never grows.
    """

    reciprocal_solver = "pme"

    def __init__(self, mesh="auto", pme_cutoff_dtype='float64', **kwargs):
        super().__init__(**kwargs)
        if pme_cutoff_dtype not in ('float32', 'float64'):
            raise ValueError("pme_cutoff_dtype must be 'float32' or 'float64'")
        self.reciprocal_solver = 'pme'
        self.pme_cutoff_dtype = pme_cutoff_dtype
        if any(not math.isfinite(value) or value <= 0 for value in (self.sigma, self.dl, self.norm_factor)):
            raise ValueError("PME sigma, dl and norm_factor must be finite and positive")
        self._set_mesh(mesh)
        self.last_resolved_mesh = self._mesh
        self._mesh_cache = {}

    def _set_mesh(self, mesh):
        if mesh is None or (isinstance(mesh, str) and mesh.strip().lower() == "auto"):
            self.auto_mesh = True
            # Placeholder until a cell is seen; resolve_mesh() picks the real grid.
            self._mesh = (PME_AUTO_MIN_DIM,) * 3
            return
        if isinstance(mesh, str):
            raise ValueError("PME mesh requires 'auto' or three integer dimensions of at least 6")
        shape = (mesh,) * 3 if isinstance(mesh, Integral) else tuple(mesh)
        if len(shape) != 3 or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 6 for n in shape):
            raise ValueError("PME mesh requires three integer dimensions of at least 6")
        self._mesh = tuple(int(n) for n in shape)
        self.auto_mesh = False

    @property
    def mesh(self) -> Tuple[int, int, int]:
        """Fixed mesh shape; with ``auto_mesh`` the 64^3 placeholder (see ``last_resolved_mesh``)."""
        return self._mesh

    @mesh.setter
    def mesh(self, value) -> None:
        self._set_mesh(value)

    def resolve_mesh(
        self,
        cell_or_lengths,
        bounds: Optional[Sequence[int]] = None,
    ) -> Tuple[int, int, int]:
        if getattr(self, "auto_mesh", False):
            resolved = recommend_pme_mesh(
                cell_or_lengths,
                dl=self.dl,
                min_dim=PME_AUTO_MIN_DIM,
                bounds=bounds,
            )
        else:
            resolved = self._mesh
        self.last_resolved_mesh = resolved
        return resolved

    def __setstate__(self, state):
        state.setdefault('pme_cutoff_dtype', 'float64')
        state.setdefault('_mesh_cache', {})
        if "_mesh" not in state and "mesh" in state:
            # Saved before automatic meshes existed (pre-0cc6774): every stored
            # mesh, including 64, was a fixed grid and stays fixed.
            state["_mesh"] = tuple(int(n) for n in state.pop("mesh"))
        # From 0cc6774 on, auto_mesh is stored; True there always meant the
        # automatic grid (an explicit 64 was folded into auto then), so keep it.
        state["auto_mesh"] = bool(state.get("auto_mesh", False))
        state.pop("_explicit_mesh", None)
        if state["auto_mesh"]:
            state["_mesh"] = (PME_AUTO_MIN_DIM,) * 3
        state.setdefault("last_resolved_mesh", state["_mesh"])
        super().__setstate__(state)

    def _get_mesh_tables(self, device, dtype, mesh_shape: Optional[Tuple[int, int, int]] = None):
        cache = getattr(self, '_mesh_cache', None)
        if cache is None:
            cache = {}
            self._mesh_cache = cache
        resolved_mesh = self.mesh if mesh_shape is None else tuple(int(n) for n in mesh_shape)
        key = (resolved_mesh, device, dtype)
        cached = cache.get(key)
        if cached is not None:
            return cached
        nx, ny, nz = resolved_mesh
        shape = torch.tensor(resolved_mesh, device=device, dtype=dtype)
        axes = [torch.arange(n, device=device, dtype=dtype) for n in (nx, ny)]
        axes = [torch.where(axis > n // 2, axis - n, axis) for axis, n in zip(axes, (nx, ny))]
        axes.append(torch.arange(nz // 2 + 1, device=device, dtype=dtype))
        modes = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        deconvolution = torch.sinc(modes / shape).pow(12).prod(-1)
        multiplicity = torch.full((nz // 2 + 1,), 2., device=device, dtype=dtype)
        multiplicity[0] = 1
        if nz % 2 == 0:
            multiplicity[-1] = 1
        shape_int = torch.tensor(resolved_mesh, device=device)
        arange6 = torch.arange(6, device=device)
        cached = (shape_int, arange6, modes, deconvolution, multiplicity)
        cache[key] = cached
        return cached

    def compute_potential_triclinic(self, r_raw, q, cell_now, u=None, quad=None,
                                    kappa=None, alpha=None, compute_potential=False,
                                    compute_field=False, e_ext=None):
        q_one_dim = q.dim() == 1
        u_two_dim = (u.dim() == 2 and u.shape[1] == 3) if u is not None else (
            q_one_dim and (alpha is None or alpha.dim() in (1, 3))
        )
        if q_one_dim:
            q = q.unsqueeze(1)
        if r_raw.ndim != 2 or r_raw.shape[1] != 3 or q.ndim != 2 or q.shape[0] != r_raw.shape[0] or q.shape[1] < 1:
            raise ValueError("PME requires positions [N,3] and charges [N] or [N,C]")
        if cell_now.shape != (3, 3):
            raise ValueError("PME requires a finite 3x3 cell")
        with torch.autocast(device_type=r_raw.device.type, enabled=False):
            if self.compute_dtype == 'float32':
                r, q, cell = r_raw.float(), q.float(), cell_now.float()
            else:
                r, q, cell = r_raw.double(), q.double(), cell_now.double()
            u = u.to(q) if u is not None else None
            quad = quad.to(q) if quad is not None else None
            kappa = kappa.to(q) if kappa is not None else None
            alpha = alpha.to(q) if alpha is not None else None
            e_ext = e_ext.to(q) if e_ext is not None else None
            if u is not None:
                u = u.unsqueeze(1) if u.ndim == 2 else u
                if u.shape != (*q.shape, 3):
                    raise ValueError('u dimension error')
            if quad is not None:
                quad = quad.unsqueeze(1) if quad.ndim == 3 else quad
                if quad.shape != (*q.shape, 3, 3):
                    raise ValueError('quad dimension error')
                quad = 0.5 * (quad + quad.transpose(-1, -2))
            volume = torch.abs(cell.det())
            cutoff_cell = cell_now.detach().double() if self.pme_cutoff_dtype == 'float64' else cell_now.detach().float()
            axis_lengths = cutoff_cell.norm(dim=1)
            bounds = (axis_lengths / self.dl).floor().clamp(min=1)
            cell_finite = torch.isfinite(cell_now.detach()).all().to(volume.dtype)
            checks_cpu = torch.cat(
                [
                    cell_finite.reshape(1),
                    volume.detach().reshape(1),
                    axis_lengths.to(volume.dtype).reshape(3),
                    bounds.to(volume.dtype).reshape(3),
                ]
            ).to(device="cpu")
            cell_ok = float(checks_cpu[0].item())
            volume_value = float(checks_cpu[1].item())
            if cell_ok < 0.5:
                raise ValueError("PME requires a finite 3x3 cell")
            if not math.isfinite(volume_value) or volume_value < 1e-6:
                raise ValueError("PME requires a nonsingular periodic cell")
            lengths_cpu = (
                float(checks_cpu[2].item()),
                float(checks_cpu[3].item()),
                float(checks_cpu[4].item()),
            )
            bounds_cpu = (
                int(checks_cpu[5].item()),
                int(checks_cpu[6].item()),
                int(checks_cpu[7].item()),
            )
            mesh_shape = self.resolve_mesh(lengths_cpu, bounds=bounds_cpu)
            if any(2 * b >= n for b, n in zip(bounds_cpu, mesh_shape)):
                raise ValueError("PME mesh is too small for the cell/dl reciprocal support")
            shape, arange6, modes, deconvolution, multiplicity = self._get_mesh_tables(
                r.device, r.dtype, mesh_shape=mesh_shape
            )
            inverse = cell.inverse()
            fractional = (r @ inverse).remainder(1)
            coordinates = fractional * shape
            unwrapped = coordinates.floor().long()[:, :, None] - 2 + arange6
            nodes = unwrapped.remainder(shape[None, :, None])
            weights = _spline(coordinates[:, :, None] - unwrapped, 6)
            nx, ny, nz = mesh_shape
            indices = ((nodes[:, 0, :, None, None] * ny + nodes[:, 1, None, :, None]) * nz
                       + nodes[:, 2, None, None, :]).reshape(len(r), 216)
            assignment = (weights[:, 0, :, None, None] * weights[:, 1, None, :, None]
                          * weights[:, 2, None, None, :]).reshape(len(r), 216)
            points = nx * ny * nz
            channels = q.shape[1]
            components = [q]
            if u is not None:
                components.extend(u.unbind(-1))
            pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
            if quad is not None:
                components.extend(quad[..., i, j] if i == j else quad[..., i, j] + quad[..., j, i]
                                  for i, j in pairs)
            sources = torch.cat(components, dim=1).T
            grids = len(components) * channels
            density = q.new_zeros(grids, points).scatter_add(
                1, indices.reshape(1, -1).expand(grids, -1),
                (sources[:, :, None] * assignment).reshape(grids, -1))
            spectra = torch.fft.rfftn(density.reshape(grids, nx, ny, nz), dim=(-3, -2, -1))
            k = modes @ (self.twopi * inverse.T)
            # FFT uses exp(-ikr), the conjugate of the original structure factor.
            spectrum = spectra[:channels]
            offset = channels
            if u is not None:
                for axis in range(3):
                    spectrum = spectrum - 1j * k[..., axis] * spectra[offset:offset + channels]
                    offset += channels
            if quad is not None:
                for i, j in pairs:
                    spectrum = spectrum - .5 * k[..., i] * k[..., j] * spectra[offset:offset + channels]
                    offset += channels
            k2 = k.square().sum(-1)
            if cutoff_cell.dtype == r.dtype:
                cutoff_k2 = k2.detach()
            else:
                cutoff_k = modes.to(cutoff_cell.dtype) @ (self.twopi * cutoff_cell.inverse().T)
                cutoff_k2 = cutoff_k.square().sum(-1)
            keep = (cutoff_k2 > 0) & (cutoff_k2 <= self.k_sq_max) & (modes.abs() <= bounds).all(-1)
            influence = torch.where(keep,
                self.norm_factor / volume * (-self.sigma_sq_half * k2).exp()
                / torch.where(k2 > 0, k2, torch.ones_like(k2)) / deconvolution, 0.)
            # abs() has the wrong Hessian at zero complex coefficients.
            power = spectrum.real.square() + spectrum.imag.square()
            pot = (power * influence * multiplicity).sum(dim=(-3, -2, -1))
            phi = torch.zeros_like(q)
            field = q.new_zeros(len(r), channels, 3)
            potential_needed, field_needed = compute_potential or kappa is not None, compute_field or alpha is not None
            if potential_needed or field_needed:
                spec_inf = spectrum * influence
                irfft_inputs = []
                if potential_needed:
                    irfft_inputs.append(spec_inf)
                if field_needed:
                    for axis in range(3):
                        irfft_inputs.append(-1j * k[..., axis] * spec_inf)
                stacked_spec = torch.cat(irfft_inputs, dim=0)
                grids_real = 2 * points * torch.fft.irfftn(stacked_spec, s=mesh_shape, dim=(-3, -2, -1))
                n_comp = Stacked_n = grids_real.shape[0]
                gathered = (grids_real.reshape(n_comp, points)[:, indices.reshape(-1)]
                            .reshape(n_comp, len(r), 216) * assignment).sum(-1).T  # [len(r), n_comp]
                off = 0
                if potential_needed:
                    phi = gathered[:, :channels]
                    off = channels
                if field_needed:
                    field = torch.stack([gathered[:, off + axis * channels:off + (axis + 1) * channels]
                                         for axis in range(3)], dim=-1)
            out = self._finish_multipoles(pot, phi, field, q, u, quad, kappa, alpha,
                                         e_ext, q_one_dim, True, volume, u_two_dim=u_two_dim)
            if not potential_needed:
                out['phi'] = torch.zeros_like(phi)
            if not field_needed:
                out['field'] = torch.zeros_like(field)
            return out
