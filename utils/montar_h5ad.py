import scanpy as sc
import pandas as pd
import sys
import os

# Pega o nome do arquivo final passado no terminal
arquivo_saida = sys.argv[1] # Ex: "Microglial_Final.h5ad"

print("-> Lendo arquivos brutos e transpondo a matriz...")
adata = sc.read_mtx("temp_matriz.mtx").T
adata.obs_names = pd.read_csv("temp_celulas.csv", header=None)[0]
adata.var_names = pd.read_csv("temp_genes.csv", header=None)[0]
adata.obs = pd.read_csv("temp_metadados.csv", index_col=0)

print(f"-> Salvando AnnData em: {arquivo_saida}")
adata.write_h5ad(arquivo_saida)

print("-> Limpando arquivos temporarios...")
os.remove("temp_matriz.mtx")
os.remove("temp_celulas.csv")
os.remove("temp_genes.csv")
os.remove("temp_metadados.csv")

print("-> Processo concluído com sucesso!")