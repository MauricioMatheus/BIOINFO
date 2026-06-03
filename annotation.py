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

adata = sc.read_h5ad("BIOINFO/clustering.h5ad")

sc.tl.rank_genes_groups(adata, groupby='leiden_res0_5', method='wilcoxon')

sc.pl.rank_genes_groups(adata, n_genes=5, sharey=False, save="BIOINFO/_top5_marcadores.png", show=False)

cl_annotation = {
    "0": "Células T",          # Marcador IL7R
    "1": "Eritrócitos",        # Marcador HBB
    "2": "Células Desconhecidas", # Marcadores muito misturados/fracos
    "3": "Eritrócitos",        # Marcador HBB
    "4": "Células Tronco/Progenitoras", # Marcador CDK6
    "5": "Células Desconhecidas",
    "6": "Células B",          # Marcadores BANK1, CD74
    "7": "Monócitos",          # Marcador VCAN
    "8": "Células Desconhecidas"
}

# Aplicando a tradução (O carimbo)

adata.obs["Cell_Type"] = adata.obs['leiden_res0_5'].map(cl_annotation)

#UMAP com nomes biológicos
sc.pl.umap(adata, color="Cell_Type", legend_loc="on data", title="Anotação Celular")
