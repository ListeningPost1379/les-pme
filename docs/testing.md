# Testing

Run from the checkout root. Fresh virtual environments have been checked with Python 3.12.15, PyTorch 2.10.0 and NumPy 2.2.6 on CPU. Candidate checks import an installed wheel; reference checks use a separate pinned upstream installation. CUDA checks were skipped on this host. The PyTorch dependency floor is 2.5. The full Python/PyTorch version matrix has not been qualified.

## Standalone CPU checks

These checks need PyTorch, NumPy and pytest, with no MACE installation or independent LES reference:

```sh
python -m pip install -e '.[yaml]' pytest
PYTHONPATH=src python -m pytest src/les/tests/test_audit_regressions.py src/les/tests/test_reciprocal_memory.py test/test_ewald_realspace -q
```

## Independent upstream comparison

Reference tests load the installed `les` distribution directly and import the candidate through `PYTHONPATH`. Do **not** install this fork into the reference environment: the packages share the same distribution name. Create a separate environment and install the pinned upstream there, then run from this checkout:

```sh
python -m venv .venv-reference
# Activate this environment before the following commands.
python -m pip install 'les @ git+https://github.com/ChengUCB/les.git@35b971a650cf30adb84fef7138761e81aa47b3a2' pytest
PYTHONPATH=src python -m pytest src/les/tests test/test_ewald_realspace -q
```

Check both locations before interpreting the results:

```sh
PYTHONPATH=src python -c 'import les, importlib.metadata; print(les.__file__); print(importlib.metadata.distribution("les").locate_file("les/module/ewald.py"))'
```

The first path must be this checkout's `src/les`; the second must be the separately installed upstream. Missing or substituted references invalidate an upstream-equivalence claim. CUDA cases skip when CUDA is unavailable.

## Optional MACE integration

A compatible maceles-mh **source checkout**, including its test fixtures and dependencies, is required. A pip-installed stock MACE is insufficient. In the reference environment, set `MACE_DIR` to that checkout:

```sh
PYTHONPATH="src:$MACE_DIR" python -m pytest tests/test_maceles_memory.py tests/test_maceles_pme.py tests/test_maceles_full_solvers.py -q
```

Training, dataset preparation and pretrained model use are documented in the companion repository. This package ships no datasets or pretrained weights.

## Legacy scripts and profiling

`python test/run_all.py` includes expensive compiler checks and smoke scripts; `RAN` means execution without numerical assertions. Some vectorized physics scripts fetch their reference using `git show origin/main:...`. A history-free export has no such ref and may skip that physics gate; use the explicit independent-reference tests above for numerical acceptance. A successful runner exit alone does not prove every reference gate ran.

For a CUDA profiling diagnostic, run `python tests/profile_reciprocal.py --solver pme --mesh 64` in the reference environment **without** a `PYTHONPATH` override. It is an isolated solver measurement, not a host-model throughput benchmark.
