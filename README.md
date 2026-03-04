# mlx-vis

Dimensionality reduction on Apple Silicon. UMAP, t-SNE, PaCMAP, and NNDescent - all running on Metal via MLX.

## Install

```bash
uv pip install mlx-vis
```

From source:

```bash
git clone --recurse-submodules https://github.com/hanxiao/mlx-vis.git
cd mlx-vis
uv pip install .
```

## Usage

```python
import numpy as np
from mlx_vis import UMAP, TSNE, PaCMAP, NNDescent

X = np.random.randn(10000, 128).astype(np.float32)

# UMAP
Y = UMAP(n_components=2, n_neighbors=15).fit_transform(X)

# t-SNE
Y = TSNE(n_components=2, perplexity=30).fit_transform(X)

# PaCMAP
Y = PaCMAP(n_components=2, n_neighbors=10).fit_transform(X)

# NNDescent (approximate k-NN graph)
indices, distances = NNDescent(k=15).build(X)
```

Submodule imports also work:

```python
from mlx_vis.umap import UMAP
from mlx_vis.tsne import TSNE
from mlx_vis.pacmap import PaCMAP
from mlx_vis.nndescent import NNDescent
```

## Methods

| Method | Class | Main API | Output |
|--------|-------|----------|--------|
| UMAP | `UMAP(n_components, n_neighbors, min_dist, ...)` | `fit_transform(X)` | `np.ndarray (n, d)` |
| t-SNE | `TSNE(n_components, perplexity, ...)` | `fit_transform(X)` | `np.ndarray (n, d)` |
| PaCMAP | `PaCMAP(n_components, n_neighbors, ...)` | `fit_transform(X)` | `np.ndarray (n, d)` |
| NNDescent | `NNDescent(k, n_iters, ...)` | `build(X)` | `(indices, distances)` |

## Dependencies

- `mlx >= 0.20.0`
- `numpy >= 1.24.0`

## License

Apache-2.0
