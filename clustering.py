import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#environment settings
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80, facecolor="white", frameon=False)

adata = sc.read_h5ad("Dataset_after_dim_reduction.h5ad")

sc.tl.leiden(adata, flavor="igraph", n_iterations = 2)

#different resolutions

sc.tl.leiden(
    adata, key_added="leiden_res0_25", resolution=0.25, flavor="igraph", n_iterations=2
)
sc.tl.leiden(
    adata, key_added="leiden_res0_5", resolution=0.5, flavor="igraph", n_iterations=2
)
sc.tl.leiden(
    adata, key_added="leiden_res1", resolution=1, flavor="igraph", n_iterations=2
)

sc.pl.umap(
    adata,
    color=["leiden_res0_25", "leiden_res0_5", "leiden_res1"],
    legend_loc="on data",
    save="_leiden_comparativo.png"
)

adata.write_h5ad("clustering.h5ad")