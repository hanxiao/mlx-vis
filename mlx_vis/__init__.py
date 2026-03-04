"""mlx-vis: unified dimensionality reduction on Apple Silicon."""

__version__ = "0.1.0"

from mlx_vis._umap.umap import UMAP
from mlx_vis._tsne.tsne import TSNE
from mlx_vis._pacmap.pacmap import PaCMAP
from mlx_vis._trimap.trimap import TriMap
from mlx_vis._dreams.dreams import DREAMS
from mlx_vis._nndescent.nndescent import NNDescent

from mlx_vis.plot import scatter, scatter_gpu, animate, animate_gpu

__all__ = ["UMAP", "TSNE", "PaCMAP", "TriMap", "DREAMS", "NNDescent",
           "scatter", "scatter_gpu", "animate", "animate_gpu"]
