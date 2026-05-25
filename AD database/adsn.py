'''import os
import scanpy as sc
import pandas as pd

pasta=os.path.dirname(__file__)

adata=sc.read_10x_mtx(pasta)
adata.var_names_make_unique()

#lendo o metadado que também tá na mesma pasta

metadata = pd.read_csv(os.path.join(pasta, "adsn_metadata.txt"), sep="\t", index_col=0)
adata.obs = metadata.reindex(adata.obs_names)
print(adata)
print(adata.obs.head())'''


import os
import scanpy as sc
import pandas as pd
import scipy.io

pasta = os.path.dirname(__file__)
os.chdir(pasta)

print("1. Lendo a matriz bruta (matrix.mtx)...")
with open("matrix.mtx", "rb") as f:
    # Lê a matriz, inverte para o padrão Python (Células x Genes) e otimiza a RAM
    X = scipy.io.mmread(f).T.tocsr()

print("2. Lendo os códigos de barras (barcodes.tsv)...")
obs = pd.read_csv("barcodes.tsv", header=None, sep="\t", names=["barcode"])
obs.set_index("barcode", inplace=True)

print("3. Lendo os genes (features.tsv)...")
# Como os autores deixaram apenas 1 coluna, lemos apenas ela e a transformamos no índice
var = pd.read_csv("features.tsv", header=None, sep="\t", names=["gene_symbol"])
var.set_index("gene_symbol", inplace=True)

print("4. Montando o objeto AnnData...")
adata = sc.AnnData(X=X, obs=obs, var=var)
adata.var_names_make_unique()

print("5. Acoplando os metadados clínicos...")
metadata = pd.read_csv("adsn_metadata.txt", sep="\t", index_col=0)
adata.obs = metadata.reindex(adata.obs_names)

print("\n=== SUCESSO ABSOLUTO! ESTRUTURA DO DATASET ===")
print(adata)
print("\n=== PRIMEIRAS LINHAS DOS METADADOS ===")
print(adata.obs.head())