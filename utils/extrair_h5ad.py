import scanpy as sc
import scipy.io
import scipy.sparse
import pandas as pd
import sys
import os

arquivo_entrada = sys.argv[1]

print(f"-> Carregando AnnData: {arquivo_entrada}")
adata = sc.read_h5ad(arquivo_entrada)

# TRAVA DE SEGURANÇA 1: Unicidade Absoluta
# Força que não exista NENHUMA célula ou gene com nome repetido (O Seurat odeia duplicatas)
adata.obs_names_make_unique()
adata.var_names_make_unique()

# TRAVA DE SEGURANÇA 2: A Prensa de Compressão
# Se a matriz estiver densa (numpy array), forçamos a conversão para CSR (Compressed Sparse Row)
# Isso impede que o computador trave tentando gravar bilhões de zeros no disco
if not scipy.sparse.issparse(adata.X):
    print("-> Matriz densa detectada! Comprimindo para formato esparso (CSR)...")
    adata.X = scipy.sparse.csr_matrix(adata.X)

matriz = adata.X

print("-> Exportando matriz esparsa (.mtx)...")
scipy.io.mmwrite("temp_matriz.mtx", matriz)

print("-> Exportando identificadores (Células e Genes)...")
pd.Series(adata.obs_names).to_csv("temp_celulas.csv", index=False, header=False)
pd.Series(adata.var_names).to_csv("temp_genes.csv", index=False, header=False)

# TRAVA DE SEGURANÇA 3: Sanitização de Valores Nulos
# O R pode se confundir com NaNs do Pandas. Convertendo tudo para texto seguro.
print("-> Exportando metadados clínicos...")
metadados_limpos = adata.obs.fillna("NA")
metadados_limpos.to_csv("temp_metadados.csv")

print("-> Extração finalizada com sucesso!")