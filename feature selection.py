import logging
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#environment settings
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=100, facecolor="white", frameon=False)

#Data loading

adata = sc.read_h5ad("Dataset after normalization.h5ad")
print(f"Dimensão original: {adata.shape[0]} células x {adata.shape[1]} genes")

#Feature Selection with seurat v3 flavor

#Scanpy looks for the 4000 more informative genes in the matrix

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=4000,
    layer="counts", # CRUCIAL: Aponta para a gaveta de dados BRUTOS (exigência do seurat_v3)
    flavor="seurat_v3", #Livre do logarítmo e o (+1). Usa os dados brutos
    inplace=True # Salva os resultados automaticamente em adata.var
)

#Data Visualization

sc.pl.highly_variable_genes(adata, show=False)

#Saving the plot

plt.savefig("figures/Filter - Post feature selection.png", bbox_inches="tight")
plt.show()

#Filter: Keeping the columns where de boolean column "highly_variable" is true

#using .copy() to ensure that RAM releases the rest of the old data
adata_filtered = adata[:, adata.var["highly_variable"]].copy()

'''A vírgula antes do : (Linhas/Células): O símbolo de dois-pontos : sozinho significa "pegue tudo". Ou seja, você está ordenando: "Mantenha todas as minhas 14.814 linhas (células) intocadas".'''

print(f"Dimensão enxuta: {adata_filtered.shape[0]} células x {adata_filtered.shape[1]} genes altamente variáveis")

adata_filtered.write_h5ad("Dataset after feature selection.h5ad")

'''Se você escrevesse adata = adata[:, ...], você destruiria a matriz gigante original da sua memória RAM e a substituiria pela versão enxuta. Se 5 minutos depois você percebesse que cometeu um erro e quisesse olhar um gene que foi deletado, você teria que rodar o script inteiro desde o começo para carregar o arquivo pesado de novo.'''