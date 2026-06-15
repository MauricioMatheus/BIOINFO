import scanpy as sc
import pandas as pd
import sys
import os

arquivo_saida = sys.argv[1]

print("-> Lendo arquivos brutos e transpondo a matriz...")
adata = sc.read_mtx("temp_matriz.mtx").T

# TRAVA DE SEGURANÇA 3: O "Escudo Anti-Inteiros"
# Força .astype(str) para o pandas não transformar códigos de barras puramente numéricos em inteiros
adata.obs_names = pd.read_csv("temp_celulas.csv", header=None)[0].astype(str).values
adata.var_names = pd.read_csv("temp_genes.csv", header=None)[0].astype(str).values

print("-> Alinhando metadados...")
metadados = pd.read_csv("temp_metadados.csv", index_col=0)
metadados.index = metadados.index.astype(str)

# TRAVA DE SEGURANÇA 4: O "Encaixe Milimétrico"
# O .loc garante alinhamento exato entre a ordem das células na matriz e a ordem na tabela
adata.obs = metadados.loc[adata.obs_names]

print(f"-> Salvando AnnData em: {arquivo_saida}")
adata.write_h5ad(arquivo_saida)

print("-> Limpando arquivos temporarios...")
os.remove("temp_matriz.mtx")
os.remove("temp_celulas.csv")
os.remove("temp_genes.csv")
os.remove("temp_metadados.csv")

print("-> Processo concluído com sucesso!")