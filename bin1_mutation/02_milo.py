import os
import scanpy as sc
import pertpy as pt
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# ETAPA 1: Carregamento e Preparação Global
# =========================================================
print("Carregando o objeto anotado...")
adata = sc.read_h5ad("bin1_mutation/adata_limpo_anotado.h5ad")

pasta_saida = "bin1_mutation/figures/milo_analysis"
os.makedirs(pasta_saida, exist_ok=True)

# Renomeando colunas para o motor matemático não confundir os pontos
adata.obs.rename(columns={
    'Age.at.death': 'Age_at_death',
    'Braak.stage': 'Braak_stage',
    'Thal.phase': 'Thal_phase'
}, inplace=True)

# Criando a variável do Modelo Recessivo
mapeamento_recessivo = {
    '0/0': '0/0_0/1', 
    '0/1': '0/0_0/1', 
    '1/1': '1/1_Recessivo'
}
adata.obs['Modelo_Recessivo'] = adata.obs['Mutacao_BIN1'].map(mapeamento_recessivo)

# Limpeza Global de NAs
# Limpando todas as variáveis de uma vez para garantir que 
# ambos os modelos testem exatamente o mesmo número de células e pacientes.
adata = adata[~adata.obs['Modelo_Recessivo'].isna() & 
              ~adata.obs['ADNC'].isna() & 
              ~adata.obs['Braak_stage'].isna() & 
              ~adata.obs['Thal_phase'].isna() & 
              ~adata.obs['Age_at_death'].isna()].copy()

# Garantindo que todas as covariáveis sejam fatores estatísticos (categorias/strings)
for col in ['Age_at_death', 'ADNC', 'Braak_stage', 'Thal_phase']:
    adata.obs[col] = adata.obs[col].astype(str)

# =========================================================
# ETAPA 2: Construção Base do Milo 
# =========================================================
print("Construindo as vizinhanças (Neighborhoods)...")
milo = pt.tl.Milo()

# Calcula as vizinhanças e conta os doadores na base unificada
milo.make_nhoods(adata, prop=0.1, n_neighbors=30)
milo.count_nhoods(adata, sample_col="donor_id")

# =========================================================
# ETAPA 3: ROTA 1 - Modelo de Consenso Global (ADNC)
# =========================================================
print("\n--- Iniciando Rota 1: Modelo ADNC ---")

# Clonando o objeto para não sujar o original
adata_adnc = adata.copy()

milo.da_nhoods(
    adata_adnc, 
    design="~ Age_at_death + ADNC + Modelo_Recessivo", 
    model_contrasts="Modelo_Recessivo1/1_Recessivo - Modelo_Recessivo0/0_0/1"
)

# Gráficos Rota 1
plt.figure(figsize=(8, 5))
plt.hist(adata_adnc.uns["nhood_adata"].obs["SpatialFDR"], bins=50)
plt.xlabel("Spatial FDR (P-valor ajustado)")
plt.ylabel("Frequência de Vizinhanças")
plt.title("Significância - Consenso Global (Idade + ADNC)")
plt.axvline(x=0.1, color='red', linestyle='--')
plt.savefig(f"{pasta_saida}/FDR_01_ADNC.png", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
pt.pl.milo.nhood_graph(
    adata_adnc, alpha=0.1, min_size=2, plot_edges=True,
    title="Impacto do BIN1 (Modelo Recessivo)\nControlado por Idade e Gravidade ADNC", ax=ax
)
plt.savefig(f"{pasta_saida}/Grafo_01_ADNC.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# ETAPA 4: ROTA 2 - Modelo Biológico (Braak + Thal)
# =========================================================
print("\n--- Iniciando Rota 2: Modelo Braak e Thal ---")

# Clonando o objeto original novamente
adata_bio = adata.copy()

milo.da_nhoods(
    adata_bio, 
    design="~ Age_at_death + Braak_stage + Thal_phase + Modelo_Recessivo", 
    model_contrasts="Modelo_Recessivo1/1_Recessivo - Modelo_Recessivo0/0_0/1"
)

# Gráficos Rota 2
plt.figure(figsize=(8, 5))
plt.hist(adata_bio.uns["nhood_adata"].obs["SpatialFDR"], bins=50)
plt.xlabel("Spatial FDR (P-valor ajustado)")
plt.ylabel("Frequência de Vizinhanças")
plt.title("Significância - Mecanismo Biológico (Idade + Tau + Amiloide)")
plt.axvline(x=0.1, color='red', linestyle='--')
plt.savefig(f"{pasta_saida}/FDR_02_Braak_Thal.png", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
pt.pl.milo.nhood_graph(
    adata_bio, alpha=0.1, min_size=2, plot_edges=True,
    title="Impacto do BIN1 (Modelo Recessivo)\nControlado por Idade, Braak (Tau) e Thal (Amiloide)", ax=ax
)
plt.savefig(f"{pasta_saida}/Grafo_02_Braak_Thal.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nAnálises concluídas! As duas rotas foram salvas na pasta.")