import pandas as pd
import scanpy as sc
import celltypist
import seaborn as sns
import matplotlib.pyplot as plt
from celltypist import models

sc.settings.figdir = 'figures/annotation/automated/'

adata = sc.read_h5ad("h5ad/clustering.h5ad")

# O CellTypist exige estritamente dados crus (counts) normalizados para 10k e em log.

adata_celltypist = adata.copy()
adata_celltypist.X = adata.layers["counts"]
sc.pp.normalize_total(adata_celltypist, target_sum=10**4)
sc.pp.log1p(adata_celltypist)
# Descomprimindo a matriz para a IA ler
adata_celltypist.X = adata_celltypist.X.toarray() 

# Baixando os "Cérebros" da IA
models.download_models(force_update=False, model=["Immune_All_Low.pkl", "Immune_All_High.pkl"])

# Carregando os modelos para a memória

model_high = models.Model.load(model="Immune_All_High.pkl")
model_low = models.Model.load(model="Immune_All_Low.pkl")

# Rodando a IA (Genérica/Coarse)
print("Rodando predição genérica...")

predictions_high = celltypist.annotate(adata_celltypist, model=model_high, majority_voting=True)
predictions_high_adata = predictions_high.to_adata()

# Transferindo as respostas da IA para o objeto adata original

adata.obs["IA_Anotacao_Generica"] = predictions_high_adata.obs.loc[adata.obs.index, "majority_voting"]
# conf_score puxa a nota (de 0 a 1) da confiança da IA e salva no objeto principal.
adata.obs["IA_Confianca_Generica"] = predictions_high_adata.obs.loc[adata.obs.index, "conf_score"] #
 
# Rodando a IA (Fina/Low)
print("Rodando predição fina.....")

predictions_low = celltypist.annotate(adata_celltypist, model=model_low, majority_voting=True)
predictions_low_adata = predictions_low.to_adata()

# Transferindo as respostas

adata.obs["IA_Anotacao_Fina"] = predictions_low_adata.obs.loc[adata.obs.index, "majority_voting"]
# conf_score puxa a nota (de 0 a 1) da confiança da IA e salva no objeto principal.
adata.obs["IA_Confianca_Fina"] = predictions_low_adata.obs.loc[adata.obs.index, "conf_score"] 

# Criando ponto de corte de confiança (<80% se torna unknown)

#Generica
# 1. Converte para texto livre
adata.obs["IA_Anotacao_Generica_Segura"] = adata.obs["IA_Anotacao_Generica"].astype(str)
# 2. Aplica o filtro
adata.obs.loc[adata.obs["IA_Confianca_Generica"] < 0.8, "IA_Anotacao_Generica_Segura"] = "Unknown"
# 3. Devolve para o formato Categórico
adata.obs["IA_Anotacao_Generica_Segura"] = adata.obs["IA_Anotacao_Generica_Segura"].astype("category")

#Fina
# 1. Converte para texto livre
adata.obs["IA_Anotacao_Fina_Segura"] = adata.obs["IA_Anotacao_Fina"].astype(str)
# 2. Aplica o filtro
adata.obs.loc[adata.obs["IA_Confianca_Fina"] < 0.8, "IA_Anotacao_Fina_Segura"] = "Unknown"
# 3. Devolve para o formato Categórico
adata.obs["IA_Anotacao_Fina_Segura"] = adata.obs["IA_Anotacao_Fina_Segura"].astype("category")




###Plots

# UMAP anotação genérica
sc.pl.umap(
    adata,
    color=["IA_Anotacao_Generica", "IA_Confianca_Generica"],
    frameon=False,#Tira a borda do gráfico
    sort_order=False,
    wspace=1, #Espaço entre os gráficos
    legend_loc="on data",
    title=["Anotação Genérica", "Score de Confiança"],
    save="_anotacao_generica_completa.png",
    show=False
)

# UMAP anotação genérica segura
sc.pl.umap(
    adata,
    color="IA_Anotacao_Generica_Segura",
    frameon=False,
    legend_loc="right margin",
    title="Anotação Genérica (Filtrada > 80%)",
    save="_anotacao_generica_filtrada_unknown.png",
    show=False,
)

# UMAP anotação fina
sc.pl.umap(
    adata,
    color=["IA_Anotacao_Fina", "IA_Confianca_Fina"],
    frameon=False,
    sort_order=False,
    wspace=1,
    legend_loc="right margin",
    title=["Anotação Fina", "Score de Confiança"],
    save="_anotacao_fina_completa.png",
    show=False
)

# UMAP anotação fina segura

sc.pl.umap(
    adata,
    color="IA_Anotacao_Fina_Segura",
    frameon=False,
    legend_loc="right margin",
    title="Anotação Fina (Filtrada > 80%)",
    save="_anotacao_fina_filtrada_unknown.png",
    show=False,
)

#Dendrograma baseada na visão da IA

sc.tl.dendrogram(adata, groupby="IA_Anotacao_Fina")
sc.pl.dendrogram(
    adata,
    groupby="IA_Anotacao_Fina",
    save="_dendrograma_IA.png",
    show=False,
)

# Boxplot de auditoria: Confiança por tipo celular
fig, ax = plt.subplots(figsize=(12, 5))
# Organizando do cluster mais confiável para o menos confiável
ordem_confianca = (
    adata.obs.groupby("IA_Anotacao_Fina")
    .agg({"IA_Confianca_Fina": "median"})
    .sort_values(by="IA_Confianca_Fina", ascending=False)
)
sns.boxplot(
    data=adata.obs,
    x="IA_Anotacao_Fina",
    y="IA_Confianca_Fina",
    color="lightgrey",
    ax=ax,
    order=ordem_confianca.index
)
ax.tick_params(rotation=90, axis="x")
plt.title("Confiança da IA por Tipo Celular")
plt.tight_layout()
plt.savefig("figures/annotation/automated/boxplot_celltypist.png")

adata.write_h5ad("h5ad/automated_annotation.h5ad", compression="gzip")
