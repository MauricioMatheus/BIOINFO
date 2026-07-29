import numpy as np
import scanpy as sc
import scanpy.external as sce
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyclustree import clustree
from sklearn.metrics import adjusted_rand_score
from scipy.stats import kruskal # cálculo do p-valor baseado nas proporções.
from scipy.stats import mannwhitneyu

adata = sc.read_h5ad("bin1_mutation/BIN1_Mutation.h5ad")


# =========================================================
# ETAPA 0: Filtro de Coorte (Apenas Pacientes com Neuropatologia AD)
# =========================================================

print(f"Número de células antes do filtro clínico: {adata.shape[0]}")

# Mantendo apenas os pacientes que possuem alguma alteração neuropatológica de Alzheimer

# Ignorando a categoria 'Not AD'
adata = adata[adata.obs['ADNC'].isin(['High', 'Intermediate', 'Low'])].copy()

# Limpando a categoria 'Not AD' da memória do objeto para evitar erros em gráficos futuros
adata.obs['ADNC'] = adata.obs['ADNC'].cat.remove_unused_categories() 

print(f"Número de células após isolar o grupo AD: {adata.shape[0]}")

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
        plt.savefig(f"bin1_mutation/figures/todos_genotipos/heatmap_{nome_arquivo}_bin1.png", dpi=300, bbox_inches="tight")
        #plt.show()
    else:
        print(f"A coluna '{marcador}' não foi encontrada em adata.obs")

# ===========================================================
#ETAPA 2: Clustering (Leiden) e Fine Tuning
# ===========================================================

# Certifica que o PCA e os vizinhos estão calculados (caso não venham prontos do R)
sc.pp.pca(adata)

print("Iniciando integração com Harmony... ")

# Usando donor_id como chave de lote

sce.pp.harmony_integrate(adata, key='donor_id')

sc.pp.neighbors(adata, use_rep='X_pca_harmony') # Use reprentative

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
plt.savefig("bin1_mutation/figures/todos_genotipos/Fine Tuning da Estabilidade dos Clusters - Micróglias.png", dpi=300, bbox_inches="tight" )
#plt.show()

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
plt.savefig("bin1_mutation/figures/todos_genotipos/Heatmap_Contigencia_Best_Resolution.png", dpi=300, bbox_inches="tight")
#plt.show()

#A resolução Ideal é 0.15

#============================================================
# ETAPA 4: Visualização UMAP
#============================================================

sc.tl.umap(adata)

clusters = {
    '0': 'Micróglia Homeostática Basal (ADAM7-AS1+, DLEU7+)',
    '1': 'Micróglia Homeostática Fagocítica (FRMD4A+, PLXDC2+)',
    '2': 'Micróglia Associada à Doença - DAM (SPP1+ alto, FTH1+)',
    '3': 'Ruído Técnico / Doublets Neuronais (CADM2+, NRG3+)',
    '4': 'Micróglia Ativada por Estresse Metabólico (HIF1A+, SPP1+ mod)'
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
plt.savefig("bin1_mutation/figures/todos_genotipos/UMAP_Anotado.png", dpi=300, bbox_inches="tight")
#plt.show()

#=============================================================
# ETAPA 5: Anotação Celular e Limpeza de Ruído 
#=============================================================

# Lista de genes que são puro ruído técnico no cérebro
# (MALAT1 é o principal, mas filtra-se mitocondriais e ribossomais por segurança)

genes_ruido = [gene for gene in adata.var_names if gene == "MALAT1" or gene.startswith(('MT-', 'RPS', "RPL"))]

#Criando uma matriz nova livre desse ruído

adata_limpo = adata[:, ~adata.var_names.isin(genes_ruido)].copy()

adata_limpo = adata_limpo[adata_limpo.obs['Estado_Microglial'] != 'Ruído Técnico / Doublets Neuronais (CADM2+, NRG3+)'].copy()

adata_limpo.obs['Estado_Microglial'] = adata_limpo.obs['Estado_Microglial'].cat.remove_unused_categories()

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

plt.savefig("bin1_mutation/figures/todos_genotipos/top5_marcadores_limpo.png", dpi=300, bbox_inches="tight")
#plt.show()

# ============================================================
# Etapa 6: Impacto da Mutação nas Proporções Celulares
# ============================================================

# Atualizando a variável subclass por leiden_0.15: Impacto da mutação nas proporções

contagem_celulas = pd.crosstab( # Cruzando o genótipo com os clusters validados
    adata_limpo.obs['Mutacao_BIN1'],
    adata_limpo.obs['Estado_Microglial'],
    normalize='index'
) * 100

# Forçando a ordem lógica (biológica) das colunas
ordem_desejada = [
    'Micróglia Homeostática Basal (ADAM7-AS1+, DLEU7+)', 
    'Micróglia Homeostática Fagocítica (FRMD4A+, PLXDC2+)', 
    'Micróglia Ativada por Estresse Metabólico (HIF1A+, SPP1+ mod)', 
    'Micróglia Associada à Doença - DAM (SPP1+ alto, FTH1+)' 
]
contagem_celulas = contagem_celulas[ordem_desejada]

#Gráfico de barras empilhadas

contagem_celulas.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title("Distribuição dos Estados Microgliais (Leiden 0.15) por Genótipo BIN1")
plt.ylabel("Proporção das células (%)")
plt.xlabel("Genótipo BIN1 (rs6733839)")
plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
plt.tight_layout()
plt.savefig("bin1_mutation/figures/todos_genotipos/Proporção_clusters_BIN1_limpo.png", dpi=300, bbox_inches="tight")
#plt.show()

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
    
    plt.title(f"Distribuição da Expressão de {gene} por Genótipo e Ancestralidade")
    plt.ylabel(f"Expressão Normalizada ({gene})")
    plt.xlabel("Genótipo BIN1 (rs6733839)")
    plt.legend(title="Ancestralidade", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"bin1_mutation/figures/todos_genotipos/Expressão_{gene}_Ancestralidade_limpo.png", dpi=300, bbox_inches="tight")
    #plt.show()
    
# ===========================================================
# ETAPA 7.5: Expressão de BIN1 por Cluster (Estado Microglial)
# ===========================================================

print("\n=== Gerando gráficos de expressão de BIN1 por cluster ===")

# Gráfico Básico Nativo do Scanpy (Apenas Clusters)

fig, ax = plt.subplots(figsize=(14, 7))

sc.pl.violin(
    adata_limpo, 
    keys=['BIN1'], 
    groupby='Estado_Microglial', 
    rotation=45, 
    palette='Set2',
    ax=ax,              # Ancora o desenho do scanpy na "tela" customizada
    show=False
)
plt.title("Expressão de BIN1 por Estado Microglial")
plt.xticks(fontsize=12, ha='right') # Aumentando a fonte e centralizando a ponta do texto sob o violino
plt.yticks(fontsize=12)
plt.ylabel("Expressão Normalizada (BIN1)", fontsize=12)
plt.xlabel("") # Removendo o título do eixo X para limpar espaço
plt.tight_layout()
plt.savefig("bin1_mutation/figures/todos_genotipos/Violin_BIN1_Apenas_Clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# Gráfico Avançado Seaborn (Clusters divididos pelo Modelo Recessivo)

# Extraindo a expressão exata do gene BIN1
if hasattr(adata_limpo.X, 'toarray'):
    exp_bin1 = adata_limpo[:, 'BIN1'].X.toarray().flatten()
else:
    exp_bin1 = adata_limpo[:, 'BIN1'].X.flatten()

# Criando um mapeamento rápido do modelo recessivo para os metadados
mapeamento_recessivo = {
    '0/0': '0/0 + 0/1', 
    '0/1': '0/0 + 0/1', 
    '1/1': '1/1 (Homozigoto Recessivo)'
}

# Criando um DataFrame focado no BIN1
df_bin1_cluster = pd.DataFrame({
    'Expressão_BIN1': exp_bin1,
    'Estado_Microglial': adata_limpo.obs['Estado_Microglial'],
    'Genótipo_Recessivo': adata_limpo.obs['Mutacao_BIN1'].map(mapeamento_recessivo)
})

# Removendo NAs (caso algum paciente não tenha o genótipo catalogado)
df_bin1_cluster = df_bin1_cluster.dropna(subset=['Genótipo_Recessivo'])

# Plotando o violino dividido (split)
plt.figure(figsize=(14, 7))
sns.violinplot(
    data=df_bin1_cluster,
    x='Estado_Microglial',
    y='Expressão_BIN1',
    hue='Genótipo_Recessivo',
    split=True,         # Juntando as duas metades no mesmo violin
    inner="quart",      # Mostrando os quartis matemáticos por dentro
    linewidth=1.2,
    cut=0,
    palette={'0/0 + 0/1': '#bcbddc', '1/1 (Homozigoto Recessivo)': '#a1d99b'}
)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title("Expressão de BIN1 por Estado Microglial e Genótipo (Modelo Recessivo)")
plt.ylabel("Expressão Normalizada (BIN1)", fontsize=12)
plt.xlabel("")
plt.legend(title="Modelo Genético", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
plt.tight_layout()

plt.savefig("bin1_mutation/figures/todos_genotipos/Violin_BIN1_Clusters_e_Recessivo.png", dpi=300, bbox_inches="tight")
plt.close()
    
# ===========================================================
# ETAPAS 8 e 9: Abundância Celular e Estatística para todos os Tipos Celulares
# ===========================================================

print("\n=== Análise de Abundância Celular por Doador (Todos os Clusters) ===")

# Cálculo das proporções base
contagem_doador = pd.crosstab(adata_limpo.obs['donor_id'], adata_limpo.obs['Estado_Microglial'])
total_celulas_doador = contagem_doador.sum(axis=1)    
proporcao_doador = contagem_doador.div(total_celulas_doador, axis=0) * 100

# Configurações Globais de Plotagem
ordem_genotipos = ['0/0', '0/1', '1/1']
cores = {'0/0': '#440154', '0/1': '#35b779', '1/1': '#21918c'}

# Pegando todos os nomes dos clusters validados (as colunas da proporção)
estados_celulares = proporcao_doador.columns.tolist()

# Loop de Análise (Roda uma vez para cada tipo celular)
for estado in estados_celulares:
    print(f"\nProcessando estado: {estado}")
    
    # Montando o DataFrame específico para o cluster da rodada
    df_estado = pd.DataFrame({
        'Porcentagem': proporcao_doador[estado],
        'Total_Celulas': total_celulas_doador
    })
    
    # Mesclando com os metadados do paciente
    df_estado = df_estado.merge(
        pacientes_unicos[['donor_id', 'Mutacao_BIN1']].set_index('donor_id'), 
        left_index=True, right_index=True
    )
    
    # Limpando e filtrando os 3 genótipos alvo
    df_estado = df_estado.dropna(subset=['Mutacao_BIN1'])
    df_estado['Mutacao_BIN1'] = df_estado['Mutacao_BIN1'].astype(str)
    df_teste = df_estado[df_estado['Mutacao_BIN1'].isin(['0/0', '0/1', '1/1'])].copy()
    
    # Agrupamento em Modelo Recessivo (1/1 vs Outros)
    
    mapeamento_recessivo = {
        '0/0': '0/0 + 0/1', 
        '0/1': '0/0 + 0/1', 
        '1/1': '1/1 (Homozigoto Recessivo)'
    }
    df_teste['Agrupamento_BIN1'] = df_teste['Mutacao_BIN1'].map(mapeamento_recessivo)
    
    ordem_agrupada = ['0/0 + 0/1', '1/1 (Homozigoto Recessivo)']
    cores_agrupadas = {'0/0 + 0/1': '#440154', '1/1 (Homozigoto Recessivo)': '#21918c'}
    
    # Separando as distribuições matemáticas
    grupo_outros = df_teste[df_teste['Agrupamento_BIN1'] == '0/0 + 0/1']['Porcentagem']
    grupo_11 = df_teste[df_teste['Agrupamento_BIN1'] == '1/1 (Homozigoto Recessivo)']['Porcentagem']
    
    # Teste de Mann-Whitney U (Ideal para 2 grupos)
    stat, p_valor = mannwhitneyu(grupo_outros, grupo_11, alternative='two-sided')
    print(f"P-valor (Mann-Whitney U): {p_valor:.4f}")
    
    # PLOT
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(
        data=df_teste,
        x='Agrupamento_BIN1',
        y='Porcentagem',
        order=ordem_agrupada,
        color='lightgray',
        showfliers=False
    )
    
    sns.stripplot(
        data=df_teste,
        x='Agrupamento_BIN1',
        y='Porcentagem',
        order=ordem_agrupada,
        hue='Agrupamento_BIN1',
        palette=cores_agrupadas,
        size=8,
        jitter=True,
        alpha=0.7,
        legend=False
    )
    
    # Ajustando o título para não ficar gigante
    nome_curto = estado.split('(')[0].strip()
    
    plt.title(f"Abundância: {nome_curto}\nHomozigoto Recessivo vs Outros")
    plt.xlabel("Modelo Recessivo BIN1")
    plt.ylabel("Proporção no Doador (%)")
    
    # Prevenção matemática para o texto do p-valor
    ymax = df_teste['Porcentagem'].max()
    pos_y = ymax * 0.95 if ymax > 0 else 0.5
    
    plt.text(
        x=0.5, y=pos_y,  # Centralizado entre os dois boxplots
        s=f"Mann-Whitney p-valor = {p_valor:.4f}",
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout()
    
    # Salvando a figura
    nome_arquivo = nome_curto.replace(' ', '_').replace('-', '').replace('/', '')
    plt.savefig(f"bin1_mutation/figures/todos_genotipos/Abundancia_Recessivo_{nome_arquivo}.png", dpi=300, bbox_inches="tight")
    
    plt.close()

# adata_limpo.write_h5ad("bin1_mutation/adata_limpo_anotado.h5ad", compression='gzip')