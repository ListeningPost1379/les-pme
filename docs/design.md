# Exact blocked solver

See the [solver overview](../README.md) for original, blocked/recomputed and PME
selection. This note describes the exact blocked implementation. The packaged
PME training implementation is documented in [pme_training.md](pme_training.md).

## Preserving the original operator

The monopole optimization is in `src/les/module/reciprocal.py`, called by the eager legacy
`Ewald.compute_potential_triclinic` for monopoles without fields or induction.
It retains the original mode grid, cutoff, hemisphere factors, normalization,
and self term. It adds no model state or configuration.

The exact sum runs in blocks of 1024 modes. Its custom autograd forward saves
only positions, charges, wave vectors and weights. The analytic first derivative
is itself checkpointed when building a force graph; this avoids retaining the
atom-by-mode intermediates for force-loss backward. Derivatives through q(R),
k(cell), weights and volume remain connected. Simply checkpointing the original
whole energy did not reduce the measured force-training peak.

For the backward-compatible default `direct`, multipoles, induction, field/potential requests, mixed position/charge
dtypes, autocast, CUDA FP32 with TF32 enabled, and TorchScript retain their original code. The fixed-boundary
`Ewald_vectorized` path is also unchanged. MACE's optional `SmoothEwald` has its
own reciprocal implementation and does not use this optimization.

## Multipoles and responses

`reciprocal_solver: blocked` now extends to q/u/Q, potentials/fields and
single-response iq/iu, including multiple channels, anisotropic alpha and the
original external-field convention. `blocked.py` sums 1024 reciprocal modes
or 64-by-64 source/target atom pairs at a time. The original monopole-only fast
path stays in `reciprocal.py`. Source and target atom identities exclude self
pairs, including partially overlapping blocks.

Each block has a recomputing autograd function, and its backward VJP is itself
checkpointed when constructing a force graph. Independent clones of every
block argument take partial derivatives without double-counting a shared
upstream neural graph. Self terms and iq/iu are applied after summing all
blocks; applying response independently to each block would change the energy.
No all-atom-pair mask or rank-six kernel is retained across blocks.

The real-space kernel excludes the diagonal before taking square roots. The
independently installed original computes `norm(0)` on that diagonal and has
NaN position Hessians despite masking the kernel afterwards. Acceptance tests
therefore compare original nonperiodic energies/first derivatives directly,
then compare force-loss second derivatives to finite differences of original
first derivatives. Periodic second derivatives are compared directly.
The installed reference remains untouched, including its original kernel.

For small graphs, `real_multipoles` and `reciprocal_multipoles` take a dense
shortcut: when N*N real-space pairs or N*M atom/k-vector pairs fit within
`dense_pairs` (default 65536, i.e. N <= 256 for real space), the whole graph is
one ordinary autograd block without recomputation; larger graphs stream as
described above. `dense_pairs: 0` restores always-streaming behaviour. The
periodic monopole energy-only path in `reciprocal.py` is unchanged.

Original/direct defaults and original TorchScript remain available. Explicit
blocked is eager-only and requires autocast disabled; periodic CUDA FP32 also
requires TF32 disabled. Fixed-boundary/vectorized and smooth solvers are
separate. PME and its nonperiodic blocked path are described in
[pme_training.md](pme_training.md).
