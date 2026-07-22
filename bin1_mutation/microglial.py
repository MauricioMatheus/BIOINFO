import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyclustree import clustree
from sklearn.metrics import adjusted_rand_score
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

adata = sc.read_h5ad("bin1_mutation/BIN1_Mutation.h5ad")


# =========================================================
# ETAPA 1: Mapeamento clínico e Demográfico
# =========================================================

#Isolando os metadados mantendo 1 linha por paciente
# Apenas uma célula por paciente, impedindo viés celular pois há 36.483 linhas no dataset
pacientes_unicos = adata.obs.drop_duplicates(subset=['donor_id']) 

# Lista com os marcadores clínicos clássicos da doença

marcadores_clinicos = ['Braak.stage', 'CERAD.score', 'Thal.phase', 'ADNC']

for marcador in marcadores_clinicos:
    if marcador in pacientes_unicos.columns:
        #Cruzando genótipo BIN1 com Braak [normalizado por %]
        tabela_cruzada = pd.crosstab(
        pacientes_unicos['Mutacao_BIN1'],
        pacientes_unicos[marcador],
        normalize='index' #Normalizando
        )* 100

    #plotando o heatmap

        plt.figure(figsize=(8, 4))
        sns.heatmap(tabela_cruzada, annot=True, cmap="Reds", fmt=".1f")
        plt.title(f"Proporção de {marcador} (%) por Genótipo BIN1 (Pacientes Únicos)")
        plt.ylabel("Genótipo BIN1 (rs6733839)")
        plt.xlabel(marcador)
        plt.tight_layout()
        nome_arquivo = marcador.replace('.', '_').replace(' ', '_')
        plt.savefig(f"bin1_mutation/figures/heatmap_{nome_arquivo}_bin1.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print(f"A coluna '{marcador}' não foi encontrada em adata.obs")

# ===========================================================
#ETAPA 2: Clustering (Leiden) e Fine Tuning
# ===========================================================

# Certifica que o PCA e os vizinhos estão calculados (caso não venham prontos do R)
sc.pp.pca(adata)
sc.pp.neighbors(adata)

resolucoes = [0.05, 0.1, 0.15, 0.175, 0.2, 0.25]
for res in resolucoes:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}', flavor="igraph")

# Especifica o prefixo das colunas criadas para a biblioteca scclustree

# lista com os nomes das colunas
chaves_cluster = [f'leiden_{res}' for res in resolucoes]


plt.figure(figsize=(10, 8))
clustree(
    adata,
    cluster_keys=chaves_cluster,
    title="Estabilidade dos Clusters (Fine Tuning) - Micróglias"
)
plt.tight_layout()
plt.savefig("bin1_mutation/figures/Fine Tuning da Estabilidade dos Clusters - Micróglias.png", dpi=300, bbox_inches="tight" )
plt.show()

#=============================================================
# ETAPA 3: Prova de Fidelidade Biológica (ARI)
#=============================================================

# O clustree é a sua prova de estabilidade estrutural e o ARI é a sua prova de fidelidade biológica ao SEA-AD.

# Se provou ineficiente para essa dataset!!

gabarito = adata.obs['Supertype'].fillna("Desconhecido").astype(str)
resultados_ari = {}

# Calculando ARI para cada resolução testada

for res in resolucoes:
    predicao = adata.obs[f'leiden_{res}'].astype(str)
    score = adjusted_rand_score(gabarito, predicao)
    resultados_ari[res] = score
    print(f"Resolução {res}: ARI = {score:.4f}")

# Encontrando a vencedora matematicamente

res_vencedora = max(resultados_ari, key=resultados_ari.get)
print(f"\nA resolução matematicamente mais fidedigna ao SEA-AD é: {res_vencedora}")

# Heatmap de Contingência da resolução vencedora (Prova final biológica)

crosstab_clusters = pd.crosstab(
    adata.obs['Supertype'],
    adata.obs[f'leiden_{res_vencedora}'],
    normalize='columns' #Normalizando colunas para visualizar os clusters
) * 100

plt.figure(figsize=(10, 6))
sns.heatmap(crosstab_clusters, cmap="Blues", annot=True, fmt=".1f")
plt.title(f"Correspondência: Leiden {res_vencedora} vs SEA_AD Supertype (%)")
plt.xlabel(f"Clusters Leiden (Resolução {res_vencedora})")
plt.ylabel("Anotação Original SEA-AD (Supertype)")
plt.tight_layout()
plt.savefig("bin1_mutation/figures/Heatmap_Contigencia_Best_Resolution.png", dpi=300, bbox_inches="tight")
plt.show()

#A resolução Ideal é 0.15

#============================================================
# ETAPA 4: Visualização UMAP
#============================================================

sc.tl.umap(adata)

clusters = {
    '0': 'Homeostática 1 (Repouso)',
    '1': 'Homeostática 2',
    '2': 'Subpopulação (DOCK4+)',
    '3': 'Estado DAM (SPP1+)',
}

# Mapeando os nomes matemáticos para uma nova coluna biológica
adata.obs['Estado_Microglial'] = adata.obs['leiden_0.15'].astype(str).map(clusters)

sc.pl.umap(
    adata,
    color='Estado_Microglial',
    title="Projeção UMAP dos Estados Microgliais (Resolução 0.15)",
    palette='Set1',
    show=False
)
plt.savefig("bin1_mutation/figures/UMAP_Anotado.png", dpi=300, bbox_inches="tight")
plt.show()

#=============================================================
# ETAPA 5: Anotação Celular e Limpeza de Ruído 
#=============================================================

# Lista de genes que são puro ruído técnico no cérebro
# (MALAT1 é o principal, mas filtra-se mitocondriais e ribossomais por segurança)

genes_ruido = [gene for gene in adata.var_names if gene == "MALAT1" or gene.startswith(('MT-', 'RPS', "RPL"))]

#Criando uma matriz nova livre desse ruído

adata_limpo = adata[:, ~adata.var_names.isin(genes_ruido)].copy()

# Backup bruto: dados brutos para o pseudobulk
adata_limpo.raw = adata_limpo

# Normalizando e logaritmizando os dados para wilcoxon e dotplot

sc.pp.normalize_total(adata_limpo, target_sum=1e4)
sc.pp.log1p(adata_limpo)

sc.tl.rank_genes_groups(adata_limpo, groupby='leiden_0.15', method='wilcoxon', key_added='marcadores_wilcoxon')

# Dotplot para top 5 genes de cada cluster

# Organizando o dotplot
sc.tl.dendrogram(adata_limpo, groupby='leiden_0.15')

sc.pl.rank_genes_groups_dotplot(
    adata_limpo,
    n_genes=5,
    key='marcadores_wilcoxon',
    groupby='leiden_0.15',
    title="Top 5 Genes Marcadores por Cluster (Leiden 0.15)",
    show=False 
)

plt.savefig("bin1_mutation/figures/top5_marcadores.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Etapa 6: Impacto da Mutação nas Proporções Celulares
# ============================================================

# Atualizando a variável subclass por leiden_0.15: Impacto da mutação nas proporções

contagem_celulas = pd.crosstab( # Cruzando o genótipo com os clusters validados
    adata.obs['Mutacao_BIN1'],
    adata.obs['Estado_Microglial'],
    normalize='index'
) * 100

# Forçando a ordem lógica (biológica) das colunas
ordem_desejada = ['Homeostática 1 (Repouso)', 'Homeostática 2', 'Subpopulação (DOCK4+)', 'Estado DAM (SPP1+)']
contagem_celulas = contagem_celulas[ordem_desejada]

#Gráfico de barras empilhadas

contagem_celulas.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title("Distribuição dos Estados Microgliais (Leiden 0.15) por Genótipo BIN1")
plt.ylabel("Proporção das células (%)")
plt.xlabel("Genótipo BIN1 (rs6733839)")
plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
plt.tight_layout()
plt.savefig("bin1_mutation/figures/Proporção_clusters_BIN1.png", dpi=300, bbox_inches="tight")
plt.show()

#=============================================================
# ETAPA 7: Expressão Gênica por Genótipo e Ancestralidade
# ============================================================

# Lista de genes: BIN1 (mecanismo/mutação) e SPP1 (função/estado DAM)

genes_alvo = ['BIN1', 'SPP1']

for gene in genes_alvo:
    # Extraindo os dados de expressão
    if hasattr(adata_limpo.X, 'toarray'):
        expressao_gene = adata_limpo[:, gene].X.toarray().flatten()
    else:
        expressao_gene = adata_limpo[:, gene].X.flatten()
    
    # Criando um dataframe pro gene atual
    df_ancestralidade = pd.DataFrame({
        'Expressão': expressao_gene,
        'Genótipo': adata_limpo.obs['Mutacao_BIN1'],
        'Ancestralidade': adata_limpo.obs['ethnicity']
    })
    
    # Violin Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_ancestralidade,
        x='Genótipo',
        y='Expressão',
        hue='Ancestralidade',
        palette='Set2',
        inner="quartile",
        linewidth=1.2,
        density_norm="width"
    )
    
    plt.title(f"Distribuição da Expressão de {gene} por Genótipo e Ancestralidade)")
    plt.ylabel(f"Expressão Normalizada ({gene})")
    plt.xlabel("Genótipo BIN1 (rs6733839)")
    plt.legend(title="Ancestralidade", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"bin1_mutation/figures/Expressão_{gene}_Ancestralidade.png", dpi=300, bbox_inches="tight")
    plt.show()
    
# ============================================================
# ETAPA 8: Agregação Pseudobulk
# ============================================================

# Garantindo uso da matriz bruta, pois o deseq2 exige n. inteiros

if adata_limpo.raw is not None:
    matriz_counts = adata_limpo.raw.X
else:
    matriz_counts = adata_limpo.X
    
# Descomprimindo a matriz caso ela seja esparsa para o pandas conseguir ler

if hasattr(matriz_counts, 'toarray'):
    matriz_densa = matriz_counts.toarray()
else:
    matriz_densa = matriz_counts

# Cirando a tabela associando as contagens aos donor_ids

df_counts = pd.DataFrame(
    matriz_densa, #numeros puros
    index=adata_limpo.obs['donor_id'], # Nome das linhas
    columns=adata_limpo.var_names #Cabeçalho das colunas [genes]
)

# Somando todas as células do mesmo paciente (Agregação)

pseudobulk_df = df_counts.groupby(df_counts.index).sum()

print("Dimensão da matriz celular original: ", adata_limpo.shape)
print("Dimensão da matriz de pacientes (pseudobulk): ", pseudobulk_df.shape)

# ============================================================
# Etapa 9: Exportação para análise de Expressão Diferencial
# ============================================================

print("\n--- Iniciando Inferência Estatística Pseudobulk ---")

# Preparando os metadados no nivel do paciente
# Filtrando os pacientes unicos e alinhando os índices

df_metadata = pacientes_unicos.set_index('donor_id')[["Mutacao_BIN1", "ethnicity", "Braak.stage"]].copy()

# Garantindo que a ordem dos pacientes seja igual à do pseudobulk_df

df_metadata = df_metadata.loc[pseudobulk_df.index]

# Filtrando apenas os genótipos extremos para contraste estatístico (0/0 vs 1/1)
# Isolando grupo selvagem e mutado homozigoto

pacientes_filtro = df_metadata['Mutacao_BIN1'].isin(['0/0', '1/1'])
counts_finais = pseudobulk_df.loc[pacientes_filtro].copy()
metadata_final = df_metadata.loc[pacientes_filtro].copy()

# Pré-filtragem de genes com contagem global muito baixo ( < 10 leituras totais) [Melhora o poder estatítisco removendo ruído de baixa expressão]

# Alterações feitas com IA:

# 1. Força a coluna do genótipo a ser do tipo 'string' pura. 
# Por que é seguro? Evita que o Pandas confunda o '0/0' com uma data ou divisão matemática, o que quebra a matriz.
metadata_final['Mutacao_BIN1'] = metadata_final['Mutacao_BIN1'].astype(str)

# 2. Remove qualquer paciente que, por algum erro no arquivo original, tenha a anotação da mutação em branco (NaN).
# Por que é seguro? O DESeq2 não consegue calcular diferença de expressão para um paciente "sem grupo". Isso previne a "Singular Matrix".
metadata_final = metadata_final.dropna(subset=['Mutacao_BIN1'])

# 3. Realinha a matriz de expressão para garantir que ela tenha exatamente os mesmos pacientes do metadado limpo.
# Por que é seguro? Garante simetria perfeita entre o "X" e o "Y" da equação de regressão.
counts_finais = counts_finais.loc[metadata_final.index]

genes_validos = counts_finais.sum(axis=0) >= 10
counts_finais = counts_finais.loc[:, genes_validos]

print(f"Pacientes analisados (0/0) vs 1/1: {counts_finais.shape[0]}")
print(f"Genes analisados após filtragem: {counts_finais.shape[1]}")

# Inicialização e ajuste do modelo DESeq2
# n_cpus= -1 utiliza todos os núcleos do processador para paralelismo
dds = DeseqDataSet(
    counts=counts_finais.astype(int), # O DESeq2 exige que a matriz de contagem tenha estritamente números inteiros
    metadata=metadata_final,
    design="~Mutacao_BIN1",
    n_cpus=1
)

# Ajuste dos fatores de tamanho e dispersão bayesiana
dds.deseq2()

# Teste estatístico (Wald Test)
# Testando o impacto do genótipo '1/1' em relação ao '0/0' (controle)

stat_res = DeseqStats(
    dds,
    contrast=["Mutacao_BIN1", "1/1", "0/0"],
    n_cpus=1 
)
stat_res.summary()

# Extraindo tabela final de resultados contendo Log2FoldChange e p-adj

df_resultados = stat_res.results_df

# Removendo genes que receberam 'NaN' do algoritmo do DESeq2
df_resultados = df_resultados.dropna(subset=['padj', 'log2FoldChange'])

# Exibindo o resultado para o gene SPP1

if 'SPP1' in df_resultados.index:
    spp1_res = df_resultados.loc['SPP1']
    print("\n=== VALIDAÇÃO ESTATÍSTICA FINAL DO GENE SPP1 ===")
    print(f"Log2 Fold Change: {spp1_res['log2FoldChange']:.4f}")
    print(f"P-valor Ajustado (FDR): {spp1_res['padj']:.4e}")
    
# Salvando a tabela com todos os genes e suas significâncias estatísticas

df_resultados.to_csv("bin1_mutation/resultados_expressao_diferencial_DESeq2.csv")

# Visualização final em Volcano Plot

plt.figure(figsize=(9, 6))

# Definindo pontos significativos (p-adj < 0.05 e |log2FC| > 0.5)
sig = (df_resultados['padj'] < 0.05) & (np.abs(df_resultados['log2FoldChange']) > 0.5)

# Plot dos genes neutros

plt.scatter(
    df_resultados.loc[~sig, 'log2FoldChange'],
    -np.log10(df_resultados.loc[~sig, 'padj']),
    color='grey', alpha=0.4, s=15, label='Não Significativo'
)

# Plot dos genes significativos

plt.scatter(
    df_resultados.loc[sig, 'log2FoldChange'],
    -np.log10(df_resultados.loc[sig, 'padj']),
    color='red', alpha=0.8, s=25, label='Significativo (padj < 0.05)'
) 

# Destacando o SPP1 no gráfico

if 'SPP1' in df_resultados.index:
    spp1_fc = df_resultados.loc['SPP1', 'log2FoldChange']
    spp1_p = -np.log10(df_resultados.loc['SPP1', 'padj'])
    plt.scatter(spp1_fc, spp1_p, color='blue', s=80, zorder=5)
    plt.annotate(
        'SPP1 (DAM)',
        (spp1_fc, spp1_p),
        textcoords="offset points",
        xytext=(10, 10),
        ha='center',
        fontweight='bold',
        color='blue'
    )

plt.axhline(-np.log10(0.05), linestyle='--', color='black', linewidth=0.8)
plt.axvline(0, linestyle='--', color='black', linewidth=0.8)
plt.title("Volcano Plot: Expressão Diferencial (Somente Cluster DAM SPP1+) - BIN1 1/1 vs 0/0")
plt.xlabel("Log2 Fold Change")
plt.ylabel("-Log10 P-valor ajustado")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("bin1_mutation/figures/Volcano_Plot_DESeq2_Cluster_DAM2.png", dpi=300, bbox_inches="tight")
plt.show()