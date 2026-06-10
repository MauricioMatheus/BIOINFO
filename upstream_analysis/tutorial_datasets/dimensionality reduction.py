import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False )

adata = sc.read_h5ad("Dataset after feature selection.h5ad")

'''O algoritmo começa olhando para a tabela .obs (que guarda os metadados das células) e varre a coluna 'scDblFinder_class' linha por linha. Para cada célula, ele faz a pergunta: "O valor aqui dentro é exatamente igual a 'singlet'?" O resultado disso é uma lista (um vetor) do mesmo tamanho do seu número de células, preenchida apenas com valores lógicos: True (Verdadeiro) para as células que são seguras e False (Falso) para as células que o algoritmo marcou como doublets.

'''
adata = adata[adata.obs['scDblFinder_class'] == 1].copy()

sc.pp.pca(adata, svd_solver="arpack", mask_var="highly_variable")

sc.pl.pca_scatter(adata, color="total_counts", save="_scatter.png", show=False) #pca já vai salvar como nome inicial por causa do sc.pl.pca_scater

sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color="total_counts", save="_total_counts.png")

sc.pl.umap(
    adata,
    color=["total_counts", "pct_counts_mt", "scDblFinder_score", "scDblFinder_class"], save="_total_counts_limpo.png"
)

adata.write_h5ad("Dataset_after_dim_reduction.h5ad", compression="gzip")