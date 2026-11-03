# LES-PME

English | [简体中文](README.zh-CN.md)

Version: **0.2.0+lespme.1** · Extension author and maintainer: **Dehan Cao** (<caodehan@gmail.com>).

A PyTorch extension of [Latent Ewald Summation (LES)](https://github.com/ChengUCB/les), providing trainable dense, blocked/recomputed and PME solvers with permanent multipoles, single induced responses and force-loss double backward. PME approximates the original LES operator and requires mesh convergence checks.

## Installation

```sh
git clone https://github.com/ListeningPost1379/les-pme.git
cd les-pme
python -m pip install .
# Optional YAML configuration support:
python -m pip install '.[yaml]'
```

Requires PyTorch >= 2.5 and NumPy. Install a matching CUDA-enabled PyTorch for GPU use. This fork shares the distribution and import name `les` with upstream; installing it replaces upstream in the same environment, so use a separate virtual environment.

The companion [maceles-mh](https://github.com/ListeningPost1379/maceles-mh) checkout is required for its MACE-specific training features; a stock MACE installation is not sufficient.

## Periodic solvers

Select a solver through the `Les` constructor argument `reciprocal_solver`:

| Value | Description |
| --- | --- |
| `direct` (default) | Backward-compatible entry: uses the monopole `blocked` fast path when eligible, otherwise the dense sum. |
| `original` | Dense finite reciprocal sum, aligned with the upstream numerical reference. |
| `blocked` | Exact blocked/recomputed sum, including multipoles, induction, potentials and fields. Eager-only; disable TF32 for periodic FP32. |
| `pme` | PyTorch PME mesh approximation; converge mesh error separately. |

`Ewald_vectorized` (fixed boundaries) and MACE's optional `SmoothEwald` are separate variants and do not use these solver paths.

## Features and configuration

- **Permanent multipoles:** charges `q`, dipoles `u` and quadrupoles `Q`.
- **Single induced response:** induced charges (`kappa`) and induced dipoles (`alpha`, isotropic or anisotropic).
- **External fields:** `e_ext` accepts shared `[3]` or per-structure `[B, 3]` fields.
- **BEC:** Born effective charges.
- **Force-loss double backward:** energy → forces (`create_graph=True`) → parameter gradients.

| Option | Values | Notes |
| --- | --- | --- |
| `compute_dtype` | `auto`, `float32`, `float64` | Under `auto`, original/blocked follow input precision; periodic PME defaults to FP64 and nonperiodic PME follows the input. |
| `pme_mesh` | `"auto"` (default), `int` or `[nx, ny, nz]` | PME only. Grid resolution, not floating-point precision. Per cell axis of length `L`, `auto` picks `max(64, 2⌊L/dl⌋+2, ⌈3.5·L/dl⌉)`, rounded up to an FFT-friendly size. Any explicit integer or three integers, including `64`, specify a fixed mesh. |
| `dense_pairs` | Nonnegative integer; default `65536` (256²) | For blocked/PME, real-space N² pairs or reciprocal multipole/response N·M pairs (atoms × k-vectors) within this budget use one ordinary autograd block. Larger graphs stream with recomputation. `0` always streams. |
| `pme_cutoff_dtype` | `float64` (default), `float32` | PME only. Precision used to select the discrete reciprocal support; FP64 avoids mode-set changes at boundaries. |

Explicit precision and blocked/PME require the legacy kernel with automatic boundaries (`is_periodic=None`). See the [PME guide](docs/pme_training.md) for accuracy and deployment limitations.

## Quick start

```python
from les import Les
import torch

r = torch.rand(8, 3, dtype=torch.float64) * 6.0
q = torch.tensor([0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.05, -0.05], dtype=torch.float64)
cell = (torch.eye(3, dtype=torch.float64) * 8.0).unsqueeze(0)

les = Les({
    "use_atomwise": False,
    "reciprocal_solver": "blocked",   # or "original" / "pme"
    "compute_dtype": "float64",
    # "pme_mesh": "auto",             # PME only; default "auto", or a fixed [nx, ny, nz]
    # "pme_cutoff_dtype": "float64",
})

out = les(positions=r, cell=cell, latent_charges=q)
energy = out["E_lr"]                  # [n_graphs]
```

PME examples:

```python
les = Les({"use_atomwise": False, "reciprocal_solver": "pme"})                         # auto mesh
les = Les({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": [96, 96, 96]})  # fixed mesh
```

## Tensor shapes

| Quantity | Shape |
| --- | --- |
| `positions` / `r` | `[N, 3]` |
| `cell` | `[3, 3]` or `[B, 3, 3]`; a near-zero absolute determinant denotes a nonperiodic structure. |
| `batch` | `[N]`; optional, defaults to a single structure. |
| `latent_charges` / `q`, `latent_kappas` / `kappa` | `[N]` or `[N, C]` |
| `latent_dipoles` / `u` | `[N, 3]` or `[N, C, 3]` |
| `latent_quads` / `Q` | `[N, 3, 3]` or `[N, C, 3, 3]` |
| `latent_alphas` / `alpha` | `[N]` / `[N, C]` (isotropic) or `[N, 3, 3]` / `[N, C, 3, 3]` (anisotropic) |
| `e_ext` | `[3]` or `[B, 3]`, in the same graph order as `cell`. |

Multipole and response features are optional. Induction is a single response to the full permanent-source potential/field; cross and self terms match original LES.

## Tests

See [Testing](docs/testing.md) for standalone, independent-reference and optional MACE checks. A standalone CPU command is:

```sh
python -m pip install pytest
PYTHONPATH=src python -m pytest src/les/tests/test_audit_regressions.py src/les/tests/test_reciprocal_memory.py test/test_ewald_realspace -q
```

## Edge cases covered

- **Batch ordering:** non-monotonic or interleaved `batch` indices preserve input atom order; empty graphs yield zero energy.
- **Left-handed cells:** periodicity and volume use `|det(cell)|`.
- **Mixed dimensions:** one-/two-dimensional `q` and two-/three-dimensional `u` are tracked independently to avoid silent broadcasting.
- **Mixed periodic/nonperiodic batches with PME `auto`:** periodic PME retains FP64 output; `Ewald.forward` aligns dtypes when batched with nonperiodic FP32 graphs.

## Documentation and citation

The technical guides in `docs/` are maintained in English.

| Document | Content |
| --- | --- |
| [Library interface](docs/library.md) | Inputs, outputs and integration into a host model. |
| [Blocked solver](docs/design.md) | Exact blocking and recomputation, multipoles and solver boundaries. |
| [PME training](docs/pme_training.md) | Configuration, operator, validation and limitations. |
| [Testing](docs/testing.md) | Standalone checks and independent-reference setup. |
| [Theory](docs/theory.md) | LES theory overview. |
| [Citation](docs/citation.md) | Software citation and upstream BibTeX entries. |

Use [CITATION.cff](CITATION.cff) to cite the LES-PME software extensions by Dehan Cao, and also cite the original LES method paper:

> Cheng, B. *Latent Ewald summation for machine learning of long-range interactions.* npj Comput. Mater. (2025). https://www.nature.com/articles/s41524-025-01577-7

## License and provenance

This fork is derived from `ChengUCB/les` at `35b971a650cf30adb84fef7138761e81aa47b3a2`. Upstream authorship and citations are retained. The upstream license declaration is **CC BY-NC 4.0** (noncommercial); see [LICENSE.md](LICENSE.md). This repository does not distribute pretrained weights or datasets.
