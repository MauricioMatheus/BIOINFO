import scanpy as sc

# 1. Carrega o seu novo arquivo
adata = sc.read_h5ad("Microglial.h5ad")

# 2. Imprime a Impressão Digital Matemática
print("=== AUDITORIA DO PYTHON ===")
print(f"Total de Células (Linhas): {adata.n_obs}")
print(f"Total de Genes (Colunas): {adata.n_vars}")
print(f"Soma Absoluta de todo o RNA: {adata.X.sum()}")
print(f"Maior pico de RNA em uma única célula: {adata.X.max()}")