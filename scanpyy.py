import scanpy as sc
sc.settings.set_figure_params(dpi=80, facecolor="white")
sc.logging.print_header()

adata = sc.datasets.pbmc3k()

#Gráfico de Diagnóstico de Controle de Qualidade (Q)                                                                              )
#sc.pl.highest_expr_genes(adata, n_top=20)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
#arpack ignora todos os 0 e otimiza o processamento e economia de RAM
sc.tl.pca(adata, svd_solver="arpack")
sc.pl.pca_variance_ratio(adata)
sc.pl.pca(adata, color="CST3")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["CST3", "NKG7", "PPBP"])