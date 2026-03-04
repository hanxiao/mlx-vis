"""mlx-vis: unified dimensionality reduction on Apple Silicon."""

__version__ = "0.1.0"

from mlx_vis._umap.umap import UMAP
from mlx_vis._tsne.tsne import TSNE
from mlx_vis._pacmap.pacmap import PaCMAP
from mlx_vis._nndescent.nndescent import NNDescent

from mlx_vis.plot import scatter, animate

__all__ = ["UMAP", "TSNE", "PaCMAP", "NNDescent", "scatter", "animate"]
