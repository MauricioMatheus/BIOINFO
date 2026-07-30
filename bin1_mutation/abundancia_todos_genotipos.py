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

# Impacto do Estágio de Braak nas Proporções Celulares

print("Gerando gráfico de proporções por Estágio de Braak...")

# Verificando se a coluna existe
if 'Braak.stage' in adata_limpo.obs.columns:
    
    # Removendo possíveis NAs na coluna Braak para não sujar o gráfico
    df_braak = adata_limpo.obs.dropna(subset=['Braak.stage']).copy()

    # Cruzando o Estágio de Braak com os clusters validados
    contagem_braak = pd.crosstab( 
        df_braak['Braak.stage'],
        df_braak['Estado_Microglial'],
        normalize='index'
    ) * 100

    # Forçando a mesma ordem lógica (biológica) das colunas usada no gráfico anterior
    contagem_braak = contagem_braak[ordem_desejada]

    # Gráfico de barras empilhadas
    contagem_braak.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title("Distribuição dos Estados Microgliais (Leiden 0.15) por Estágio de Braak")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("Estágio de Braak (Progressão da Patologia)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig("bin1_mutation/figures/todos_genotipos/Proporcao_clusters_Braak_limpo.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Gráfico de Braak salvo com sucesso!")
else:
    print("Atenção: Coluna 'Braak.stage' não encontrada nos metadados.")

# Impacto da Fase de Thal nas Proporções Celulares

print("\n=== Gerando gráfico de proporções por Fase de Thal ===")

if 'Thal.phase' in adata_limpo.obs.columns:
    
    # GRÁFICO DE BARRAS EMPILHADAS
    
    # Removendo possíveis NAs na coluna Thal para não sujar o gráfico
    df_thal = adata_limpo.obs.dropna(subset=['Thal.phase']).copy()

    # Cruzando a Fase de Thal com os clusters validados
    contagem_thal = pd.crosstab( 
        df_thal['Thal.phase'],
        df_thal['Estado_Microglial'],
        normalize='index'
    ) * 100

    # Forçando a mesma ordem lógica (biológica) das colunas
    contagem_thal = contagem_thal[ordem_desejada]

    # Plot
    contagem_thal.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title("Distribuição dos Estados Microgliais (Leiden 0.15) por Fase de Thal")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("Fase de Thal (Acúmulo de Beta-amiloide)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig("bin1_mutation/figures/todos_genotipos/Proporcao_clusters_Thal_limpo.png", dpi=300, bbox_inches="tight")
    plt.close()
    
# Impacto do Escore CERAD nas Proporções Celulares

print("\n=== Gerando gráfico de proporções por Escore CERAD ===")

if 'CERAD.score' in adata_limpo.obs.columns:
    
    # GRÁFICO DE BARRAS EMPILHADAS
    
    # Removendo possíveis NAs na coluna CERAD para não sujar o gráfico
    df_cerad = adata_limpo.obs.dropna(subset=['CERAD.score']).copy()

    # Cruzando o CERAD com os clusters validados
    contagem_cerad = pd.crosstab( 
        df_cerad['CERAD.score'],
        df_cerad['Estado_Microglial'],
        normalize='index'
    ) * 100

    # Forçando a ordem lógica (biológica) das colunas (Clusters)
    contagem_cerad = contagem_cerad[ordem_desejada]

    # Plot
    contagem_cerad.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title("Distribuição dos Estados Microgliais (Leiden 0.15) por Escore CERAD")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("Escore CERAD (Densidade de Placas Neuríticas)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig("bin1_mutation/figures/todos_genotipos/Proporcao_clusters_CERAD_limpo.png", dpi=300, bbox_inches="tight")
    plt.close()
    
# Impacto do ADNC (Consenso Global) nas Proporções Celulares

print("\n=== Gerando gráfico de proporções por Nível de ADNC ===")

if 'ADNC' in adata_limpo.obs.columns:
    
    # GRÁFICO DE BARRAS EMPILHADAS
    
    # Removendo possíveis NAs na coluna ADNC
    df_adnc = adata_limpo.obs.dropna(subset=['ADNC']).copy()

    # Cruzando o ADNC com os clusters validados
    contagem_adnc = pd.crosstab( 
        df_adnc['ADNC'],
        df_adnc['Estado_Microglial'],
        normalize='index'
    ) * 100

    # Forçando a ordem lógica biológica das colunas (Clusters) e das linhas (Gravidade)
    contagem_adnc = contagem_adnc[ordem_desejada]
    ordem_adnc = ['Low', 'Intermediate', 'High']
    contagem_adnc = contagem_adnc.reindex(ordem_adnc)

    # Plot
    contagem_adnc.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title("Distribuição dos Estados Microgliais (Leiden 0.15) por Gravidade ADNC")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("ADNC (Gravidade Neuropatológica Global)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig("bin1_mutation/figures/todos_genotipos/Proporcao_clusters_ADNC_limpo.png", dpi=300, bbox_inches="tight")
    plt.close()

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

# PAINEL INTEGRADO: Abundância Celular por Genótipo BIN1 (Kruskal-Wallis)
# ===========================================================
print("\n=== Gerando Painel Integrado de Abundância por Genótipo ===")

n_plots = len(estados_celulares)
colunas = 2
linhas = int(np.ceil(n_plots / colunas))

# Criando a figura (Grid)
fig, axes = plt.subplots(linhas, colunas, figsize=(16, 6 * linhas))
axes = axes.flatten()

for i, estado in enumerate(estados_celulares):
    ax = axes[i]
    
    # Montando o DataFrame
    df_estado = pd.DataFrame({
        'Porcentagem': proporcao_doador[estado]
    })
    
    # Mesclando com os metadados do paciente
    df_estado = df_estado.merge(
        pacientes_unicos[['donor_id', 'Mutacao_BIN1']].set_index('donor_id'), 
        left_index=True, right_index=True
    )
    
    # Limpando e filtrando os 3 genótipos alvo
    df_estado = df_estado.dropna(subset=['Mutacao_BIN1'])
    df_estado['Mutacao_BIN1'] = df_estado['Mutacao_BIN1'].astype(str)
    df_teste = df_estado[df_estado['Mutacao_BIN1'].isin(ordem_genotipos)].copy()
    
    # Separando os grupos para o Kruskal-Wallis
    g0 = df_teste[df_teste['Mutacao_BIN1'] == '0/0']['Porcentagem']
    g1 = df_teste[df_teste['Mutacao_BIN1'] == '0/1']['Porcentagem']
    g2 = df_teste[df_teste['Mutacao_BIN1'] == '1/1']['Porcentagem']
    
    # Calculando p-valor
    stat, p_valor = kruskal(g0, g1, g2)
    
    # Plot do Boxplot
    sns.boxplot(
        data=df_teste,
        x='Mutacao_BIN1',
        y='Porcentagem',
        order=ordem_genotipos,
        color='lightgray',
        showfliers=False,
        ax=ax
    )
    
    # Plot do Stripplot
    sns.stripplot(
        data=df_teste,
        x='Mutacao_BIN1',
        y='Porcentagem',
        order=ordem_genotipos,
        palette=cores,
        size=7,
        jitter=True,
        alpha=0.7,
        hue='Mutacao_BIN1',
        legend=False,
        ax=ax
    )
    
    # Textos e Eixos
    nome_curto = estado.split('(')[0].strip()
    ax.set_title(f"{nome_curto}", fontsize=14)
    ax.set_xlabel("Genótipo BIN1 (rs6733839)", fontsize=12)
    ax.set_ylabel("Proporção no Doador (%)", fontsize=12)
    
    # Posicionamento dinâmico do p-valor na caixa de texto
    ymax = df_teste['Porcentagem'].max()
    pos_y = ymax * 0.90 if ymax > 0 else 0.5
    
    ax.text(
        x=1, y=pos_y,  # Centralizado no genótipo 0/1
        s=f"Kruskal-Wallis p = {p_valor:.4f}",
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5')
    )

# Removendo subplots vazios caso haja número ímpar de clusters
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Título Geral
plt.suptitle("Painel Integrado: Abundância Microglial por Genótipo BIN1 (rs6733839)", fontsize=18, y=1.02)
plt.tight_layout()

# Salvando a figura
plt.savefig("bin1_mutation/figures/todos_genotipos/Painel_Integrado_Abundancia_Genotipo.png", dpi=300, bbox_inches="tight")
plt.close()

print("Painel integrado de abundância por genótipo concluído!")


# PAINEL INTEGRADO: Abundância Celular - Modelo Recessivo (Mann-Whitney)
# ===========================================================
print("\n=== Gerando Painel Integrado de Abundância por Modelo Recessivo ===")

# Configurações do grid
n_plots = len(estados_celulares)
colunas = 2
linhas = int(np.ceil(n_plots / colunas))

fig, axes = plt.subplots(linhas, colunas, figsize=(16, 6 * linhas))
axes = axes.flatten()

ordem_agrupada = ['0/0 + 0/1', '1/1 (Homozigoto Recessivo)']
cores_agrupadas = {'0/0 + 0/1': '#440154', '1/1 (Homozigoto Recessivo)': '#21918c'}

for i, estado in enumerate(estados_celulares):
    ax = axes[i]
    
    # Montando o DataFrame específico
    df_estado = pd.DataFrame({
        'Porcentagem': proporcao_doador[estado]
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
    
    # Separando as distribuições matemáticas
    grupo_outros = df_teste[df_teste['Agrupamento_BIN1'] == '0/0 + 0/1']['Porcentagem']
    grupo_11 = df_teste[df_teste['Agrupamento_BIN1'] == '1/1 (Homozigoto Recessivo)']['Porcentagem']
    
    # Teste de Mann-Whitney U
    stat, p_valor = mannwhitneyu(grupo_outros, grupo_11, alternative='two-sided')
    
    # Plot do Boxplot
    sns.boxplot(
        data=df_teste,
        x='Agrupamento_BIN1',
        y='Porcentagem',
        order=ordem_agrupada,
        color='lightgray',
        showfliers=False,
        ax=ax
    )
    
    # Plot do Stripplot
    sns.stripplot(
        data=df_teste,
        x='Agrupamento_BIN1',
        y='Porcentagem',
        order=ordem_agrupada,
        hue='Agrupamento_BIN1',
        palette=cores_agrupadas,
        size=7,
        jitter=True,
        alpha=0.7,
        legend=False,
        ax=ax
    )
    
    # Ajustando textos e eixos do subplot
    nome_curto = estado.split('(')[0].strip()
    ax.set_title(f"{nome_curto}", fontsize=14)
    ax.set_xlabel("Modelo Recessivo BIN1", fontsize=12)
    ax.set_ylabel("Proporção no Doador (%)", fontsize=12)
    
    # Posicionamento dinâmico do p-valor na caixa de texto
    ymax = df_teste['Porcentagem'].max()
    pos_y = ymax * 0.90 if ymax > 0 else 0.5
    
    ax.text(
        x=0.5, y=pos_y,  # Posição 0.5 fica centralizada entre os dois grupos no eixo X
        s=f"Mann-Whitney p = {p_valor:.4f}",
        ha='center', va='center', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5')
    )

# Removendo subplots vazios caso haja número ímpar de clusters
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Título Geral para a figura inteira
plt.suptitle("Painel Integrado: Abundância Microglial por Modelo Recessivo BIN1\n(Homozigoto Recessivo vs Outros)", fontsize=18, y=1.05)
plt.tight_layout()

# Salvando a figura
plt.savefig("bin1_mutation/figures/todos_genotipos/Painel_Integrado_Recessivo.png", dpi=300, bbox_inches="tight")
plt.close()

print("Painel integrado do modelo recessivo concluído!")
    
# ============================================================
# INVESTIGAÇÃO DE VIÉS: Abundância Celular por Estágio de Braak (Painel Integrado)
# ============================================================
print("\n=== Gerando Painel Integrado de Stripplots por Estágio de Braak ===")

if 'Braak.stage' in adata_limpo.obs.columns:
    
    # Pegando todos os nomes dos clusters validados
    estados_celulares = proporcao_doador.columns.tolist()
    n_plots = len(estados_celulares)
    
    # Configurando o grid (2 colunas, linhas dinâmicas dependendo da quantidade de clusters)
    colunas = 2
    linhas = int(np.ceil(n_plots / colunas))
    
    # Criando a figura grande com os subplots
    fig, axes = plt.subplots(linhas, colunas, figsize=(16, 6 * linhas))
    axes = axes.flatten() # Achata a matriz de eixos para facilitar o loop
    
    for i, estado in enumerate(estados_celulares):
        ax = axes[i]
        
        # Montando o DataFrame específico
        df_investigacao = pd.DataFrame({
            'Porcentagem': proporcao_doador[estado]
        })
        
        # Mesclando com o Estágio de Braak de cada paciente
        df_investigacao = df_investigacao.merge(
            pacientes_unicos[['donor_id', 'Braak.stage']].set_index('donor_id'), 
            left_index=True, right_index=True
        )
        
        # Limpando NAs
        df_investigacao = df_investigacao.dropna(subset=['Braak.stage'])
        
        # Ordenando os estágios de Braak de forma cronológica/biológica
        ordem_braak = sorted(df_investigacao['Braak.stage'].unique())
        
        # O boxplot por baixo mostra a verdadeira mediana (imune ao viés)
        sns.boxplot(
            data=df_investigacao,
            x='Braak.stage',
            y='Porcentagem',
            order=ordem_braak,
            color='lightgray',
            showfliers=False,
            ax=ax # Aponta para o subplot atual
        )
        
        # O stripplot por cima mostra cada paciente (doador) como um ponto
        sns.stripplot(
            data=df_investigacao,
            x='Braak.stage',
            y='Porcentagem',
            order=ordem_braak,
            palette='viridis',
            size=7,
            jitter=True,
            alpha=0.7,
            hue='Braak.stage',
            legend=False,
            ax=ax # Aponta para o subplot atual
        )
        
        # Ajustando o título para não ficar gigante
        nome_curto = estado.split('(')[0].strip()
        
        ax.set_title(f"Distribuição de {nome_curto}", fontsize=14)
        ax.set_xlabel("Estágio de Braak", fontsize=12)
        ax.set_ylabel("Proporção no Doador (%)", fontsize=12)
        
    # Removendo eixos vazios caso o número de clusters seja ímpar
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Título geral para a figura inteira
    plt.suptitle("Investigação de Viés: Proporção dos Estados Microgliais por Estágio de Braak\n(Cada ponto = 1 Paciente)", fontsize=18, y=1.02)
    
    plt.tight_layout()
    
    # Salvando o painel completo
    plt.savefig("bin1_mutation/figures/todos_genotipos/Painel_Stripplots_Braak.png", dpi=300, bbox_inches="tight")
    plt.close()
        
    print("Painel integrado salvo com sucesso!")
else:
    print("Atenção: Coluna 'Braak.stage' não encontrada.")
    
    
# 2. PAINEL DE STRIPPLOTS (INVESTIGAÇÃO DE VIÉS - THAL.PHASE)
print("Gerando Painel Integrado de Stripplots por Fase de Thal...")
    
estados_celulares = proporcao_doador.columns.tolist()
n_plots = len(estados_celulares)

colunas = 2
linhas = int(np.ceil(n_plots / colunas))

fig, axes = plt.subplots(linhas, colunas, figsize=(16, 6 * linhas))
axes = axes.flatten() 

for i, estado in enumerate(estados_celulares):
    ax = axes[i]
    
    df_investigacao_thal = pd.DataFrame({
        'Porcentagem': proporcao_doador[estado]
    })
    
    df_investigacao_thal = df_investigacao_thal.merge(
        pacientes_unicos[['donor_id', 'Thal.phase']].set_index('donor_id'), 
        left_index=True, right_index=True
    )
    
    df_investigacao_thal = df_investigacao_thal.dropna(subset=['Thal.phase'])
    
    # Garantindo ordem biológica (Fase 0 a 5, etc)
    ordem_thal = sorted(df_investigacao_thal['Thal.phase'].unique())
    
    sns.boxplot(
        data=df_investigacao_thal,
        x='Thal.phase',
        y='Porcentagem',
        order=ordem_thal,
        color='lightgray',
        showfliers=False,
        ax=ax 
    )
    
    sns.stripplot(
        data=df_investigacao_thal,
        x='Thal.phase',
        y='Porcentagem',
        order=ordem_thal,
        palette='magma', # Mudando a paleta para diferenciar do Braak (viridis)
        size=7,
        jitter=True,
        alpha=0.7,
        hue='Thal.phase',
        legend=False,
        ax=ax 
    )
    
    nome_curto = estado.split('(')[0].strip()
    
    ax.set_title(f"Distribuição de {nome_curto}", fontsize=14)
    ax.set_xlabel("Fase de Thal", fontsize=12)
    ax.set_ylabel("Proporção no Doador (%)", fontsize=12)
    
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.suptitle("Investigação de Viés: Proporção dos Estados Microgliais por Fase de Thal\n(Cada ponto = 1 Paciente)", fontsize=18, y=1.02)

plt.tight_layout()
plt.savefig("bin1_mutation/figures/todos_genotipos/Painel_Stripplots_Thal.png", dpi=300, bbox_inches="tight")
plt.close()
    
print("Análises da Fase de Thal concluídas com sucesso!")


# PAINEL DE STRIPPLOTS (INVESTIGAÇÃO DE VIÉS - CERAD)

print("Gerando Painel Integrado de Stripplots por Escore CERAD...")

estados_celulares = proporcao_doador.columns.tolist()
n_plots = len(estados_celulares)

colunas = 2
linhas = int(np.ceil(n_plots / colunas))

fig, axes = plt.subplots(linhas, colunas, figsize=(16, 6 * linhas))
axes = axes.flatten() 

for i, estado in enumerate(estados_celulares):
    ax = axes[i]
    
    df_investigacao_cerad = pd.DataFrame({
        'Porcentagem': proporcao_doador[estado]
    })
    
    df_investigacao_cerad = df_investigacao_cerad.merge(
        pacientes_unicos[['donor_id', 'CERAD.score']].set_index('donor_id'), 
        left_index=True, right_index=True
    )
    
    df_investigacao_cerad = df_investigacao_cerad.dropna(subset=['CERAD.score'])
    
    # Garantindo ordem crescente/alfabética dos escores
    ordem_cerad = sorted(df_investigacao_cerad['CERAD.score'].unique())
    
    # O Boxplot base
    sns.boxplot(
        data=df_investigacao_cerad,
        x='CERAD.score',
        y='Porcentagem',
        order=ordem_cerad,
        color='lightgray',
        showfliers=False,
        ax=ax 
    )
    
    # O Stripplot sobreposto
    sns.stripplot(
        data=df_investigacao_cerad,
        x='CERAD.score',
        y='Porcentagem',
        order=ordem_cerad,
        palette='inferno', # Nova paleta
        size=7,
        jitter=True,
        alpha=0.7,
        hue='CERAD.score',
        legend=False,
        ax=ax 
    )
    
    nome_curto = estado.split('(')[0].strip()
    
    ax.set_title(f"Distribuição de {nome_curto}", fontsize=14)
    ax.set_xlabel("Escore CERAD", fontsize=12)
    ax.set_ylabel("Proporção no Doador (%)", fontsize=12)
    
# Limpando subplots vazios, se houver
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.suptitle("Investigação de Viés: Proporção dos Estados Microgliais por Escore CERAD\n(Cada ponto = 1 Paciente)", fontsize=18, y=1.02)

plt.tight_layout()
plt.savefig("bin1_mutation/figures/todos_genotipos/Painel_Stripplots_CERAD.png", dpi=300, bbox_inches="tight")
plt.close()
    
print("Análises do CERAD concluídas com sucesso!")

# PAINEL DE STRIPPLOTS (INVESTIGAÇÃO DE VIÉS - ADNC)
print("Gerando Painel Integrado de Stripplots por ADNC...")

estados_celulares = proporcao_doador.columns.tolist()
n_plots = len(estados_celulares)

colunas = 2
linhas = int(np.ceil(n_plots / colunas))

fig, axes = plt.subplots(linhas, colunas, figsize=(16, 6 * linhas))
axes = axes.flatten() 

for i, estado in enumerate(estados_celulares):
    ax = axes[i]
    
    df_investigacao_adnc = pd.DataFrame({
        'Porcentagem': proporcao_doador[estado]
    })
    
    df_investigacao_adnc = df_investigacao_adnc.merge(
        pacientes_unicos[['donor_id', 'ADNC']].set_index('donor_id'), 
        left_index=True, right_index=True
    )
    
    df_investigacao_adnc = df_investigacao_adnc.dropna(subset=['ADNC'])
    
    # O Boxplot base
    sns.boxplot(
        data=df_investigacao_adnc,
        x='ADNC',
        y='Porcentagem',
        order=ordem_adnc,
        color='lightgray',
        showfliers=False,
        ax=ax 
    )
    
    # O Stripplot sobreposto
    sns.stripplot(
        data=df_investigacao_adnc,
        x='ADNC',
        y='Porcentagem',
        order=ordem_adnc,
        palette='plasma', # Paleta diferente para distinguir visualmente (viridis -> magma -> plasma)
        size=7,
        jitter=True,
        alpha=0.7,
        hue='ADNC',
        legend=False,
        ax=ax 
    )
    
    nome_curto = estado.split('(')[0].strip()
    
    ax.set_title(f"Distribuição de {nome_curto}", fontsize=14)
    ax.set_xlabel("ADNC (Gravidade Neuropatológica)", fontsize=12)
    ax.set_ylabel("Proporção no Doador (%)", fontsize=12)
    
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
    
plt.suptitle("Investigação de Viés: Proporção dos Estados Microgliais por Consenso ADNC\n(Cada ponto = 1 Paciente)", fontsize=18, y=1.02)

plt.tight_layout()
plt.savefig("bin1_mutation/figures/todos_genotipos/Painel_Stripplots_ADNC.png", dpi=300, bbox_inches="tight")
plt.close()
    
print("Análises de ADNC concluídas com sucesso!")

# adata_limpo.write_h5ad("bin1_mutation/adata_limpo_anotado.h5ad", compression='gzip')