"""Custom build that copies submodule sources into mlx_vis at build time."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_COPIES = {
    "libs/umap-mlx/umap_mlx": "mlx_vis/_umap",
    "libs/tsne-mlx/tsne_mlx": "mlx_vis/_tsne",
    "libs/pacmap-mlx/pacmap_mlx": "mlx_vis/_pacmap",
    "libs/nndescent-mlx/nndescent_mlx": "mlx_vis/_nndescent",
}


class BuildPyWithSubmodules(build_py):
    """Copy submodule Python sources into the package before building."""

    def run(self):
        root = Path(__file__).parent
        for src_rel, dst_rel in _COPIES.items():
            src = root / src_rel
            dst = root / dst_rel
            if not src.exists():
                raise FileNotFoundError(
                    f"Submodule source not found: {src}\n"
                    "Run: git submodule update --init --recursive"
                )
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        super().run()


setup(cmdclass={"build_py": BuildPyWithSubmodules})
