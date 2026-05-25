import logging

import lamindb as ln
import numpy as np
import scanpy as sc
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.sparse import issparse

#Surpressing verbose logging from scanpy
sc.settings.verbosity = 0

#Setting figure parameters for clean, minimal plots
sc.settings.set_figure_params(dpi=80, facecolor="white", frameon=False)

rcb.logger.setLevel(logging.ERROR)

adata = sc.read_h5ad("Dataset after QC.h5ad")
#print(adata)

p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False)
#plt.show()

scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
#log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
p1 = sns.histplot(adata.obs["total_counts"], bins=100, kde=False, ax=axes[0])
axes[0].set_title("total_counts")
p2 = sns.histplot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[1])
axes[1].set_title("Shifted logarithm")
fig.savefig("figures/Comparação antes e após normalização dos dados.png", dpi=300, bbox_inches="tight")
plt.show()

adata.write_h5ad("Dataset after normalization.h5ad")