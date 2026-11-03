# LES-PME

Version: **0.2.0+lespme.1** · Extension author and maintainer: **Dehan Cao** (<caodehan@gmail.com>).

**中文** · 基于 [Latent Ewald Summation (LES)](https://github.com/ChengUCB/les) 的 PyTorch 扩展，提供可训练的稠密、分块重算及 PME 求解器，支持永久多极子、单次诱导响应和力损失二阶反传。保留原 LES 的物理约定，PME 是需要单独验证网格收敛的近似算法。

**English** · A PyTorch extension of [Latent Ewald Summation (LES)](https://github.com/ChengUCB/les), providing trainable dense, blocked/recomputed and PME solvers with permanent multipoles, single induced responses and force-loss double backward. PME approximates the original LES operator and requires mesh convergence checks.

## 安装 / Installation

```sh
git clone https://github.com/ListeningPost1379/les-pme.git
cd les-pme
python -m pip install .
# YAML configuration support / YAML 配置支持:
python -m pip install '.[yaml]'
```

需要 PyTorch >= 2.5 和 NumPy；GPU 使用需安装匹配的 PyTorch CUDA 版本。本包沿用上游发行名 `les` 和导入名 `les`，会替换同环境中的上游包。建议使用独立虚拟环境。

Requires PyTorch >= 2.5 and NumPy. Install a matching CUDA-enabled PyTorch for GPU use. This fork shares the distribution and import name `les` with upstream; use a separate environment. The companion [maceles-mh](https://github.com/ListeningPost1379/maceles-mh) checkout is required for its MACE-specific training features; a stock MACE installation is not sufficient.

---

## 周期求解器 / Periodic solvers

通过 `Les` 构造参数 `reciprocal_solver` 选择：

| 取值 / Value | 说明 / Description |
| --- | --- |
| `direct`（默认 / default） | 向后兼容入口：在可用时走单极子 `blocked` 快路径，其余情况回退到原稠密求和。 / Backward-compatible entry: uses the monopole `blocked` fast path when eligible, otherwise the dense sum. |
| `original` | 本检出内的稠密有限倒易空间求和（与上游数值参考对齐）。 / Dense finite reciprocal sum inside this checkout (aligned with the upstream numerical reference). |
| `blocked` | 分块重算的精确求和（多极子 / 诱导 / 势场均可）；eager；周期 FP32 需关闭 TF32。 / Exact blocked/recomputed sum (multipoles, induction, potentials/fields); eager-only; disable TF32 for periodic FP32. |
| `pme` | PyTorch PME 网格近似；需单独收敛网格误差。 / PyTorch PME mesh approximation; converge mesh error separately. |

`Ewald_vectorized`（固定边界）与 MACE 侧可选的 `SmoothEwald` 是独立变体，不走上表路径。

`Ewald_vectorized` (fixed boundaries) and MACE’s optional `SmoothEwald` are separate variants and do not use the table above.

---

## 功能 / Features

- **永久多极子 / Permanent multipoles**：电荷 `q`、偶极 `u`、四极 `Q`
- **单次诱导响应 / Single induced response**：诱导电荷（`kappa`）、诱导偶极（`alpha`，各向同性或各向异性）
- **外场 / External field**：`e_ext` 为共享 `[3]`，或按结构 `[B, 3]`
- **BEC**：Born 有效电荷
- **力损失二阶反向 / Force-loss double backward**：能量 → 力（`create_graph=True`）→ 参数梯度
- **精度 / Precision**：`compute_dtype` = `auto` / `float32` / `float64`；PME 另有 `pme_mesh`、`pme_cutoff_dtype`

| 选项 / Option | 取值 / Values | 说明 / Notes |
| --- | --- | --- |
| `compute_dtype` | `auto`, `float32`, `float64` | `auto`：`original`/`blocked` 跟随输入精度；周期 PME 默认 FP64，非周期 PME 跟随输入。 / `auto`: original/blocked follow input dtype; periodic PME defaults to FP64, nonperiodic PME follows input. |
| `pme_mesh` | `"auto"`（默认）、`int` 或 `[nx,ny,nz]` | 仅 `reciprocal_solver='pme'`。网格分辨率，不是浮点精度。`auto` 按晶胞各轴长度 `L` 与 `dl` 取 `max(64, 2⌊L/dl⌋+2, ⌈3.5·L/dl⌉)` 并上取到 FFT 友好尺寸；任何显式 `int` / `[nx,ny,nz]`（包括 `64`）都是固定网格。 / PME only. Grid resolution, not floating-point precision. `auto` picks `max(64, 2⌊L/dl⌋+2, ⌈3.5·L/dl⌉)` per cell axis, rounded up to an FFT-friendly size; any explicit `int` / `[nx,ny,nz]` (including `64`) is exactly that fixed mesh. |
| `dense_pairs` | 非负整数（默认 `65536` = 256²） | `blocked` / `pme`：实空间 N² 原子对或倒空间多极子/响应 N·M（原子 × 倒格矢）不超过该值时作为一个普通 autograd 块计算，更大的图才分块重算流式求和；`0` 表示始终分块。 / `blocked` / `pme`: real-space N² pairs or reciprocal multipole/response N·M (atoms × k-vectors) within this budget run as one ordinary autograd block; larger graphs stream with block recomputation. `0` always streams. |
| `pme_cutoff_dtype` | `float64`（默认）, `float32` | 仅 PME。离散倒易支撑选取精度；保留默认 FP64 可避免边界处模集变化。 / PME only. Discrete reciprocal-support selection; keep FP64 to avoid mode-set changes at boundaries. |

显式精度与 `blocked`/`pme` 需要遗留核 + 自动边界（`is_periodic=None`）。

Explicit precision and `blocked`/`pme` require the legacy kernel with automatic boundaries (`is_periodic=None`).

---

## 快速开始 / Quick start

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

PME 示例 / PME example：

```python
les = Les({"use_atomwise": False, "reciprocal_solver": "pme"})                         # auto mesh
les = Les({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": [96, 96, 96]})  # fixed mesh
```

---

## 张量形状 / Tensor shapes

| 量 / Quantity | 形状 / Shape |
| --- | --- |
| `positions` / `r` | `[N, 3]` |
| `cell` | `[3, 3]` 或 `[B, 3, 3]`（行列式绝对值过小视为非周期） |
| `batch` | `[N]`（可选；默认单结构） |
| `latent_charges` / `q`、`latent_kappas` / `kappa` | `[N]` 或 `[N, C]` |
| `latent_dipoles` / `u` | `[N, 3]` 或 `[N, C, 3]` |
| `latent_quads` / `Q` | `[N, 3, 3]` 或 `[N, C, 3, 3]` |
| `latent_alphas` / `alpha` | `[N]` / `[N, C]`（各向同性）或 `[N, 3, 3]` / `[N, C, 3, 3]`（各向异性） |
| `e_ext` | `[3]` 或 `[B, 3]`（与 `cell` 的图顺序一致） |

各特征可选。诱导是对完整永久源势/场的单次响应，交叉项与自相互作用与原 LES 一致。

Every feature is optional. Induction is a single response to the full permanent-source potential/field; cross and self terms match original LES.

---

## 测试 / Tests

See [Testing](docs/testing.md) for standalone, independent-reference and optional MACE checks. A standalone CPU command is:

```sh
python -m pip install pytest
PYTHONPATH=src python -m pytest src/les/tests/test_audit_regressions.py src/les/tests/test_reciprocal_memory.py test/test_ewald_realspace -q
```

---

## 已处理的边界情况 / Edge cases covered

- **Batch 顺序 / Batch ordering**：非单调或交错的 `batch` 仍按输入原子顺序回写；空图能量为零。 / Non-monotonic or interleaved `batch` indices preserve input atom order; empty graphs yield zero energy.
- **左手胞 / Left-handed cells**：周期判定与体积使用 `|det(cell)|`。 / Periodicity and volume use `|det(cell)|`.
- **混合维度 / Mixed dims**：`q` 的 1D/2D 与 `u` 的 2D/3D 独立跟踪，避免错误广播。 / 1D/2D `q` and 2D/3D `u` are tracked independently to avoid silent broadcasts.
- **混合周期批次 + PME `auto`**：周期部分保持 FP64 输出约定；与非周期 float32 同批时由 `Ewald.forward` 安全对齐 dtype。 / Periodic PME under `auto` keeps the FP64 output contract; mixed periodic/nonperiodic float32 batches are dtype-aligned in `Ewald.forward`.

---

## 文档与引用 / Docs and citation

| 文档 / Doc | 内容 / Content |
| --- | --- |
| [docs/design.md](docs/design.md) | `blocked` 设计、验证与 GPU 测量 / Blocked design, verification, GPU measurements |
| [docs/pme_training.md](docs/pme_training.md) | PME 配置、覆盖范围与限制 / PME configuration, coverage, limitations |
| [docs/theory.md](docs/theory.md) | LES 理论简述 / LES theory overview |
| [docs/citation.md](docs/citation.md) | 完整 BibTeX / Full BibTeX |

LES-PME 扩展的软件引用信息见 [CITATION.cff](CITATION.cff)，同时请引用原 LES 方法论文：

Use [CITATION.cff](CITATION.cff) to cite the LES-PME software extensions by Dehan Cao, and also cite the original LES method paper:

> Cheng, B. *Latent Ewald summation for machine learning of long-range interactions.* npj Comput. Mater. (2025). https://www.nature.com/articles/s41524-025-01577-7

---

## 许可与来源 / License and provenance

This fork is derived from `ChengUCB/les` at `35b971a650cf30adb84fef7138761e81aa47b3a2`. Upstream authorship and citations are retained. The upstream license declaration is **CC BY-NC 4.0** (noncommercial); see [LICENSE.md](LICENSE.md). This repository does not distribute pretrained weights or datasets.
