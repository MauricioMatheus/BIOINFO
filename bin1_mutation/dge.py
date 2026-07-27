# Differential Gene Expression

import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# 1. Carregando os dados que já passaram pelo Harmony e anotação
print("Carregando o AnnData anotado...")
adata_limpo = sc.read_h5ad("bin1_mutation/adata_limpo_anotado.h5ad")

# 2. Recriando o metadado de pacientes únicos (necessário para o DESeq2)
pacientes_unicos = adata_limpo.obs.drop_duplicates(subset=['donor_id'])

# ============================================================
# ETAPA 1: Agregação Pseudobulk CIRÚRGICA (Apenas DAM)
# ============================================================

print("\n--- Isolando o Cluster DAM para Pseudobulk ---")

# Filtrando o AnnData para manter apenas as células do verdadeiro Cluster DAM
adata_dam = adata_limpo[adata_limpo.obs['Estado_Microglial'] == 'Micróglia Associada à Doença - DAM (SPP1+, FTH1+)'].copy()

# O DESeq2 exige a matriz bruta (números inteiros não normalizados) salvos no .raw
if adata_dam.raw is not None:
    matriz_counts = adata_dam.raw.X
else:
    matriz_counts = adata_dam.X
    
if hasattr(matriz_counts, 'toarray'):
    matriz_densa = matriz_counts.toarray()
else:
    matriz_densa = matriz_counts

df_counts = pd.DataFrame(
    matriz_densa, 
    index=adata_dam.obs['donor_id'], 
    columns=adata_dam.var_names      
)

pseudobulk_df = df_counts.groupby(df_counts.index).sum()

print("Dimensão da matriz de pacientes (pseudobulk DAM): ", pseudobulk_df.shape)

# ============================================================
# Etapa 2: Expressão Diferencial com Correção de Covariáveis
# ============================================================

print("\n--- Iniciando Inferência Estatística Pseudobulk (DESeq2) ---")

# Preparando metadados
df_metadata = pacientes_unicos.set_index('donor_id')[["Mutacao_BIN1", "ethnicity", "Braak.stage"]].copy()
df_metadata = df_metadata.loc[pseudobulk_df.index]

df_metadata['Mutacao_BIN1'] = df_metadata['Mutacao_BIN1'].replace('1/0', '0/1')
pacientes_filtro = df_metadata['Mutacao_BIN1'].isin(['0/0', '1/1'])

counts_finais = pseudobulk_df.loc[pacientes_filtro].copy()
metadata_final = df_metadata.loc[pacientes_filtro].copy()

# Tratamento e Limpeza de Metadados
metadata_final['Mutacao_BIN1'] = metadata_final['Mutacao_BIN1'].astype(str)
metadata_final['ethnicity'] = metadata_final['ethnicity'].fillna('Unknown').astype(str)
metadata_final = metadata_final.dropna(subset=['Mutacao_BIN1'])

counts_finais = counts_finais.loc[metadata_final.index]

genes_validos = counts_finais.sum(axis=0) >= 10
counts_finais = counts_finais.loc[:, genes_validos]

# Design controlando para Etnia (Batch Effect / Ancestralidade)
dds = DeseqDataSet(
    counts=counts_finais.astype(int), 
    metadata=metadata_final,
    design="~ethnicity + Mutacao_BIN1", 
    n_cpus=1
)

dds.deseq2()

stat_res = DeseqStats(
    dds,
    contrast=["Mutacao_BIN1", "1/1", "0/0"],
    n_cpus=1 
)
stat_res.summary()

df_resultados = stat_res.results_df
df_resultados = df_resultados.dropna(subset=['padj', 'log2FoldChange'])

if 'SPP1' in df_resultados.index:
    spp1_res = df_resultados.loc['SPP1']
    print("\n=== VALIDAÇÃO ESTATÍSTICA FINAL DO GENE SPP1 ===")
    print(f"Log2 Fold Change: {spp1_res['log2FoldChange']:.4f}")
    print(f"P-valor Ajustado (FDR): {spp1_res['padj']:.4e}")
    
df_resultados.to_csv("bin1_mutation/resultados_expressao_diferencial_DESeq2_Controlado.csv")

# ============================================================
# ETAPA 3: Visualização Final em Volcano Plot
# ============================================================

print("\n--- Gerando Volcano Plot ---")

plt.figure(figsize=(10, 7))

# Definindo os limiares estatísticos (Thresholds)
# padj < 0.05 (Significância estatística) e |log2FoldChange| > 0.5 (Relevância biológica)
sig = (df_resultados['padj'] < 0.05) & (np.abs(df_resultados['log2FoldChange']) > 0.5)
up_regulated = sig & (df_resultados['log2FoldChange'] > 0)
down_regulated = sig & (df_resultados['log2FoldChange'] < 0)

# Plotando os genes neutros (fundo do gráfico)
plt.scatter(
    df_resultados.loc[~sig, 'log2FoldChange'],
    -np.log10(df_resultados.loc[~sig, 'padj']),
    color='lightgrey', alpha=0.5, s=15, label='Não Significativo'
)

# Plotando genes Down-regulados (Esquerda - Azul)
plt.scatter(
    df_resultados.loc[down_regulated, 'log2FoldChange'],
    -np.log10(df_resultados.loc[down_regulated, 'padj']),
    color='blue', alpha=0.8, s=30, label='Down-regulado (padj < 0.05)'
)

# Plotando genes Up-regulados (Direita - Vermelho)
plt.scatter(
    df_resultados.loc[up_regulated, 'log2FoldChange'],
    -np.log10(df_resultados.loc[up_regulated, 'padj']),
    color='red', alpha=0.8, s=30, label='Up-regulado (padj < 0.05)'
)

# Destacando o SPP1 (Marcador principal da DAM)
if 'SPP1' in df_resultados.index:
    spp1_fc = df_resultados.loc['SPP1', 'log2FoldChange']
    spp1_p = -np.log10(df_resultados.loc['SPP1', 'padj'])
    plt.scatter(spp1_fc, spp1_p, color='black', marker='*', s=150, zorder=5)
    plt.annotate(
        'SPP1',
        (spp1_fc, spp1_p),
        textcoords="offset points",
        xytext=(10, 10),
        ha='left',
        fontweight='bold',
        color='black',
        fontsize=12
    )

# Linhas de corte (Threshold lines)
plt.axhline(-np.log10(0.05), linestyle='--', color='black', linewidth=1)
plt.axvline(0.5, linestyle='--', color='black', linewidth=1)
plt.axvline(-0.5, linestyle='--', color='black', linewidth=1)


plt.title("Expressão Diferencial em Micróglia DAM (BIN1 1/1 vs 0/0)", fontsize=14, fontweight='bold')
plt.xlabel("Log2 Fold Change (Efeito da Mutação)", fontsize=12)
plt.ylabel("-Log10 P-valor Ajustado (Significância)", fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.savefig("bin1_mutation/figures/Volcano_Plot_DAM_BIN1.png", dpi=300, bbox_inches="tight")
plt.show()