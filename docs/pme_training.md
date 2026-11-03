# PyTorch PME training

The packaged implementation is `src/les/module/pme.py`.

The [solver overview](../README.md)
compares original LES, exact blocking/recomputation and this mesh approximation.

## Operator and reference

The reference is the independently installed original
`ChengUCB/les@35b971a650cf30adb84fef7138761e81aa47b3a2`, loaded directly from
the distribution's `les/module/ewald.py`. Tests do not use the locally optimized
Ewald as their numerical oracle. Symmetrix inference source was read for its
order-6 assignment and FFT conventions; its inference results are not used as
validation data.

PME retains the original LES Gaussian kernel, per-axis integer bounds
`max(1, floor(||cell_i||/dl))`, spherical `2*pi/dl` cutoff, omitted zero mode,
and optional self term. Coordinates use row-vector cells: fractional positions
are `r @ inverse(cell)`, and reciprocal vectors use `2*pi*inverse(cell).T`.
Sources are assigned with order-6 cardinal B-splines and the squared assignment
window is compensated by `prod(sinc(n_i/mesh_i)^12)`. The real FFT's conjugate
multiplicities are included. Mesh/FFT arithmetic defaults to FP64 even with
FP32 MACE features; `compute_dtype: float32` selects FP32 arithmetic. Gradients
propagate back through the casts. `pme_cutoff_dtype` independently controls the
discrete support test and defaults to FP64 for stable boundary selection.

Periodic mesh operations use differentiable PyTorch primitives. Positions, learned multipoles/responses and cell
remain connected through force construction and force-loss backward. The
potential gather is also checked against dE/dq without induction. There is no
periodic custom backward, native kernel, persistent mesh cache, or additional
dependency. Nonperiodic graphs use the exact recomputed blocked sum.

PME approximates the original sum. Converge mesh error separately from dl;
increasing the mesh does not remove the legacy hard-cutoff discontinuity.

## Opt in

With LES-PME installed (and the companion maceles-mh for MACE use), pass these LES arguments
to `Les` or MACELES (including the existing training CLI's LES YAML):

```yaml
use_atomwise: false
reciprocal_mode: legacy
reciprocal_solver: pme
pme_mesh: auto              # default; or a fixed mesh such as [96, 96, 96]
compute_dtype: float64
pme_cutoff_dtype: float64
```

`pme_mesh` defaults to `auto`: each axis uses
`max(64, 2*floor(L/dl)+2, ceil(3.5*L/dl))`, rounded up to an FFT-friendly size
(64, 80, 96, 128, 160, 192, 256, ...). Any explicit integer or three integers,
including `64`, is exactly that fixed mesh and never grows. Older checkpoints retain their stored fixed/automatic mesh choice. The default remains `reciprocal_solver: direct`.
For an existing MACELES model:

```python
model.set_les_reciprocal_mode("legacy", reciprocal_solver="pme", pme_mesh="auto",
                              compute_dtype="float32", pme_cutoff_dtype="float64")
```

The active selection, mesh and precision survive deepcopy, full-model save/load and
`extract_model` reconstruction. Switch back explicitly with
`model.set_les_reciprocal_mode(reciprocal_solver="direct")`.
In LES-PME, that returns to the direct reciprocal-sum family,
including its eligible blocked/recomputed path. Use `original` for the dense
reduction, or `blocked` for explicit block/recompute selection; omitted
`compute_dtype` in the setter preserves the current setting. The independently
installed upstream package remains the untouched numerical reference.

`compute_dtype` accepts `auto`, `float32`, `float64`. `auto` preserves the old
FP64 periodic PME / input-precision nonperiodic behavior. Explicit FP32/FP64
also sets nonperiodic direct-sum precision. `pme_cutoff_dtype` accepts `float32`
or `float64`; only PME accepts this option and `pme_mesh`. FP32 cutoff selection
can lose reciprocal modes at exact boundaries, so FP32 arithmetic should
normally retain FP64 cutoff tests. Neither option changes backbone precision
or adds FP64 total-energy accumulation to MACE.

Automatic boundaries (`is_periodic: null`) and the legacy kernel support q/u/Q,
potentials/fields and single-response iq/iu. Nonperiodic graphs use the exact
blocked direct sum, including multipoles and responses. Each mesh
dimension must be at least six and exceed twice the corresponding integer
reciprocal bound; geometry updates are checked again. Fixed N_max and smooth
kernels are not silently substituted.

## Multipoles and induced responses

Both LES solvers accept `u`, `quad`, `alpha` and `kappa` through Ewald, or
`latent_dipoles`, `latent_quads`, `latent_alphas`, `latent_kappas` through `Les`.
Missing features stay disabled; no new readout network is introduced in LES.
The common Ewald/PME batch entry accepts external fields as `[3]` (shared) or
`[B,3]` (one per cell). It selects each graph's field before evaluating induction,
including mixed periodic/nonperiodic batches. Single-structure kernel methods
continue to take `[3]`. MACE training and ASE metadata use the batched form.

PME assigns one charge, three dipole and six quadrupole components per channel.
Off-diagonal quadrupole sources are `Q_ij + Q_ji`; diagonal sources are `Q_ii`.
With the FFT's negative-phase convention the combined spectrum is
`Fq - i k_a Fu_a - (1/2) k_a k_b FQ_ab`. Potential and field are gathered
using the adjoint assignment; the field spectrum is `-i k * spectrum`.
Forces and cell derivatives are obtained by differentiating the mesh energy.
At finite mesh the spectral gathered field need not equal a positional
derivative of the assignment; both converge to the original LES operator.

Original q/u/Q self conventions are retained. The complete permanent-source
potential and field feed `iq = -kappa * phi` and `iu = alpha * (E + E_ext)`
once, followed by original `+1/2 phi*iq` and `-1/2 E*iu` energy terms. This
preserves original external-field and quadrupole-self conventions rather than
introducing a self-consistent polarization model. `use_epsilon_r_scaling`
returns original dielectric metadata; as in original Ewald it does not rescale
the energy. Mesh convergence must include the selected response parameters.

## MACE integration

The companion [maceles-mh](https://github.com/ListeningPost1379/maceles-mh) connects the shared `les_arguments` dictionary to scratch and foundation training. The readout options are `use_dipole`, `use_quad`, `use_induced_charge` and `use_induced_dipole`. LES implements the solver; the host model supplies learned features. See the companion repository's `docs/MH_LES_TRAINING.md` for training and ASE interfaces.

## Validation

See [Testing](testing.md) for environment setup and independent-reference commands.

The tests compare energies, forces, cell derivatives, potentials and force-loss
parameter gradients with original LES, including neutral/nonneutral systems,
self on/off, multiple charge channels and mesh refinement. Independent
gradcheck/gradgradcheck checks differentiate the PME discretization itself.
Zero-charge tests check charge Hessians and nonzero force-target loss gradients
on CPU/CUDA: Fourier power uses real^2 + imag^2, since complex abs^2 gives an
incorrect Hessian at zero coefficients in PyTorch.
MACELES checks include mixed molecular/periodic batches, FP32/FP64 and CUDA when
available, plus checkpoint/reconstruction retention. Numerical tolerances are
fixture-specific, not universal accuracy guarantees.

`test_full_solvers.py` loads the independently installed Ewald **and its own
make_kernels.py**. It compares both full solvers, individual optional features,
nonzero targets at zero multipoles, block tails, mesh refinement, mixed batches,
anisotropic alpha, external fields and response derivatives. Numerical
gradcheck/gradgradcheck exercises all q/u/Q/alpha/kappa/position/cell inputs.

## Limitations

- PME is an approximation; converge energy, force, cell derivatives and response errors for the intended systems. Automatic mesh selection is a sizing heuristic, not an accuracy guarantee.
- A finer mesh does not remove the legacy reciprocal hard-cutoff discontinuity when cells change.
- Explicit blocked and PME paths require legacy automatic boundaries. Blocked recomputation requires autocast disabled and periodic CUDA FP32 requires TF32 disabled.
- PME TorchScript/compile and multi-GPU training are not qualified. Use eager PyTorch; an eager checkpoint is not a compiled deployment artifact.
- Whole-model memory savings and training throughput depend on the host MLIP and batch geometry; no universal speedup is claimed.
- Native deployment/export support belongs to the external exporter and must be checked separately for the selected features and precision.
