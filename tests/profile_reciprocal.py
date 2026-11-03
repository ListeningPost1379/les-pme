"""Compare the installed pinned LES with this checkout, including force loss.

Run without overriding PYTHONPATH, with the original LES installed:
    python tests/profile_reciprocal.py
"""

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

import torch
import les.module
from les.module.ewald import Ewald


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--solver", choices=("recomputed", "pme"), default="recomputed")
    parser.add_argument("--mesh", type=int, default=64)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "src/les/module"
    if Path(importlib.util.find_spec("les.module.ewald").origin).resolve() == source / "ewald.py":
        raise RuntimeError("Run without PYTHONPATH=src: the installed LES is the reference")
    les.module.__path__.append(str(source))
    spec = importlib.util.spec_from_file_location("memory_ewald", source / "ewald.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = module.Ewald
    if args.solver == "pme":
        from les.module.pme import PME

        candidate = lambda **kwargs: PME(mesh=args.mesh, **kwargs)
    torch.set_default_dtype(torch.float64)
    # Leave room for other processes and keep this diagnostic below the WSL limit.
    torch.cuda.set_per_process_memory_fraction(.25)
    print(json.dumps({"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "float64"}))
    for n in args.sizes:
        results = []
        for label, cls in (("upstream", Ewald), (args.solver, candidate)):
            torch.manual_seed(42)
            r = (torch.rand(n, 3, device="cuda") * 20).requires_grad_()
            w = torch.randn(3, device="cuda", requires_grad=True)
            cell = torch.eye(3, device="cuda") * 20
            solver = cls(sigma=1, dl=2)

            def step():
                q = torch.sin(r @ w)
                energy = solver.compute_potential_triclinic(r, q, cell)["pot"].sum()
                force = -torch.autograd.grad(energy, r, create_graph=True)[0]
                gradient = torch.autograd.grad(force.square().mean() + .001 * energy.square(), w)[0]
                return torch.cat((energy.detach().view(1), gradient.detach())).cpu()

            step()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for _ in range(3):
                result = step()
            torch.cuda.synchronize()
            print(json.dumps({"n": n, "solver": label,
                              "increment_MiB": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
                              "ms": (time.perf_counter() - start) * 1000 / 3,
                              "energy_and_parameter_gradient": result.tolist()}), flush=True)
            results.append(result)
            del r, w, cell, solver
            gc.collect()
            torch.cuda.empty_cache()
        if args.solver == "recomputed":
            torch.testing.assert_close(*results, rtol=1e-10, atol=1e-9)
        else:
            assert all(torch.isfinite(result).all() for result in results)
            difference = (results[0] - results[1]).abs()
            print(json.dumps({"n": n, "mesh": args.mesh,
                              "energy_abs_error": difference[0].item(),
                              "parameter_gradient_max_abs_error": difference[1:].max().item(),
                              "parameter_gradient_scaled_error": (difference[1:] / results[0][1:].abs().clamp(min=1)).max().item()}))


if __name__ == "__main__":
    main()
