import shutil
import sys
from pathlib import Path

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas.core.indexes.base as pandas_indexes_base
import scarches as sca
from celltypist import models
from scipy.sparse import csr_matrix

adata = sc.read_h5ad("Dataset_after_clustering.h5ad")

sc.tl.rank_genes_groups(adata, groupby='leiden_res0_5', method='wilcoxon')

sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False, save="_top5_marcadores.png")