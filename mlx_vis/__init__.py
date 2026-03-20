"""mlx-vis: unified dimensionality reduction on Apple Silicon."""

__version__ = "0.7.0"

from mlx_vis._umap.umap import UMAP
from mlx_vis._tsne.tsne import TSNE
from mlx_vis._pacmap.pacmap import LocalMAP, PaCMAP
from mlx_vis._trimap.trimap import TriMap
from mlx_vis._dreams.dreams import DREAMS
from mlx_vis._cne.cne import CNE
from mlx_vis._mmae.mmae import MMAE
from mlx_vis._nndescent.nndescent import NNDescent

from mlx_vis.plot import scatter, scatter_gpu, animate, animate_gpu, morph_gpu, morph_all_effect

__all__ = ["UMAP", "TSNE", "PaCMAP", "LocalMAP", "TriMap", "DREAMS", "CNE", "MMAE", "NNDescent",
           "scatter", "scatter_gpu", "animate", "animate_gpu", "morph_gpu", "morph_all_effect"]
