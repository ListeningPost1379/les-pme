# LES-PME

[English](README.md) | 简体中文

版本：**0.2.0+lespme.1** · 扩展作者与维护者：**Dehan Cao**（<caodehan@gmail.com>）。

基于 [Latent Ewald Summation (LES)](https://github.com/ChengUCB/les) 的 PyTorch 扩展，提供可训练的稠密、分块重算及 PME 求解器，支持永久多极子、单次诱导响应和力损失二阶反传。保留原 LES 的物理约定；PME 是近似算法，需要单独检查网格收敛。

## 安装

```sh
git clone https://github.com/ListeningPost1379/les-pme.git
cd les-pme
python -m pip install .
# 可选的 YAML 配置支持：
python -m pip install '.[yaml]'
```

需要 PyTorch >= 2.5 和 NumPy；GPU 使用需安装匹配的 PyTorch CUDA 版本。本包沿用上游发行名和导入名 `les`，安装时会替换同环境中的上游包，建议使用独立虚拟环境。

MACE 扩展训练功能需要配套的 [maceles-mh](https://github.com/ListeningPost1379/maceles-mh) 源码仓库，普通 MACE 安装不足以使用这些功能。

## 周期求解器

通过 `Les` 构造参数 `reciprocal_solver` 选择：

| 取值 | 说明 |
| --- | --- |
| `direct`（默认） | 向后兼容入口：在适用时走单极子 `blocked` 快路径，其余情况回退到稠密求和。 |
| `original` | 稠密有限倒易空间求和，与上游数值参考对齐。 |
| `blocked` | 分块重算的精确求和，支持多极子、诱导响应、势与场。仅支持 eager；周期 FP32 需关闭 TF32。 |
| `pme` | PyTorch PME 网格近似，需要单独检查网格误差收敛。 |

固定边界的 `Ewald_vectorized` 与 MACE 可选的 `SmoothEwald` 是独立变体，不走上述求解器路径。

## 功能与配置

- **永久多极子**：电荷 `q`、偶极 `u`、四极 `Q`。
- **单次诱导响应**：诱导电荷（`kappa`）和诱导偶极（`alpha`，各向同性或各向异性）。
- **外场**：`e_ext` 接受共享 `[3]` 或逐结构 `[B, 3]` 外场。
- **BEC**：Born 有效电荷。
- **力损失二阶反传**：能量 → 力（`create_graph=True`）→ 参数梯度。

| 选项 | 取值 | 说明 |
| --- | --- | --- |
| `compute_dtype` | `auto`、`float32`、`float64` | `auto` 下，original/blocked 跟随输入精度；周期 PME 默认 FP64，非周期 PME 跟随输入。 |
| `pme_mesh` | `"auto"`（默认）、`int` 或 `[nx, ny, nz]` | 仅 PME。表示网格分辨率，而非浮点精度。`auto` 按各晶胞轴长 `L` 取 `max(64, 2⌊L/dl⌋+2, ⌈3.5·L/dl⌉)` 并上取到 FFT 友好尺寸；任何显式整数或三个整数（包括 `64`）均表示固定网格。 |
| `dense_pairs` | 非负整数，默认 `65536`（256²） | blocked/PME 的实空间 N² 原子对或倒空间多极子/响应 N·M 对（原子 × 波矢）不超过预算时，用一个普通 autograd 块计算；更大的图才流式分块重算。`0` 表示始终分块。 |
| `pme_cutoff_dtype` | `float64`（默认）、`float32` | 仅 PME。控制离散倒易支撑的选取精度；FP64 可避免边界处模集变化。 |

显式精度与 blocked/PME 要求 legacy 核及自动边界（`is_periodic=None`）。精度和部署限制见英文 [PME 指南](docs/pme_training.md)。

## 快速开始

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

PME 示例：

```python
les = Les({"use_atomwise": False, "reciprocal_solver": "pme"})                         # auto mesh
les = Les({"use_atomwise": False, "reciprocal_solver": "pme", "pme_mesh": [96, 96, 96]})  # fixed mesh
```

## 张量形状

| 量 | 形状 |
| --- | --- |
| `positions` / `r` | `[N, 3]` |
| `cell` | `[3, 3]` 或 `[B, 3, 3]`；行列式绝对值接近零时表示非周期结构。 |
| `batch` | `[N]`，可选，默认单结构。 |
| `latent_charges` / `q`、`latent_kappas` / `kappa` | `[N]` 或 `[N, C]` |
| `latent_dipoles` / `u` | `[N, 3]` 或 `[N, C, 3]` |
| `latent_quads` / `Q` | `[N, 3, 3]` 或 `[N, C, 3, 3]` |
| `latent_alphas` / `alpha` | `[N]` / `[N, C]`（各向同性）或 `[N, 3, 3]` / `[N, C, 3, 3]`（各向异性） |
| `e_ext` | `[3]` 或 `[B, 3]`，与 `cell` 的图顺序一致。 |

多极子和响应特征均可选。诱导是对完整永久源势/场的单次响应，交叉项与自相互作用约定与原 LES 一致。

## 测试

独立测试、独立上游对照及可选 MACE 检查的环境要求见英文 [测试指南](docs/testing.md)。独立 CPU 检查命令：

```sh
python -m pip install pytest
PYTHONPATH=src python -m pytest src/les/tests/test_audit_regressions.py src/les/tests/test_reciprocal_memory.py test/test_ewald_realspace -q
```

## 已覆盖的边界情况

- **Batch 顺序**：非单调或交错的 `batch` 仍按输入原子顺序回写；空图能量为零。
- **左手胞**：周期判定与体积使用 `|det(cell)|`。
- **混合维度**：分别跟踪一维/二维 `q` 和二维/三维 `u`，避免静默广播。
- **混合周期批次与 PME `auto`**：周期 PME 保持 FP64 输出；与非周期 FP32 图同批时由 `Ewald.forward` 对齐 dtype。

## 文档与引用

`docs/` 下的技术文档统一使用英文。

| 文档 | 内容 |
| --- | --- |
| [库接口](docs/library.md) | 输入、输出及宿主模型接入。 |
| [分块求解器](docs/design.md) | 精确分块与重算、多极子及求解器边界。 |
| [PME 训练](docs/pme_training.md) | 配置、算子、验证及限制。 |
| [测试](docs/testing.md) | 独立检查与上游对照环境。 |
| [理论](docs/theory.md) | LES 理论概述。 |
| [引用](docs/citation.md) | 软件引用与上游论文 BibTeX。 |

LES-PME 扩展作者 Dehan Cao 的软件引用信息见 [CITATION.cff](CITATION.cff)，同时请引用原 LES 方法论文：

> Cheng, B. *Latent Ewald summation for machine learning of long-range interactions.* npj Comput. Mater. (2025). https://www.nature.com/articles/s41524-025-01577-7

## 许可与来源

本扩展基于 `ChengUCB/les@35b971a650cf30adb84fef7138761e81aa47b3a2`，保留上游署名与引用。上游许可声明为 **CC BY-NC 4.0**（非商业使用），详见 [LICENSE.md](LICENSE.md)。本仓库不分发预训练权重或数据集。
