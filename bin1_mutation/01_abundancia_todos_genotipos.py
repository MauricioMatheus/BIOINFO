import os
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyclustree import clustree
from sklearn.metrics import adjusted_rand_score
from scipy.stats import kruskal # cálculo do p-valor baseado nas proporções.
from scipy.stats import mannwhitneyu

adata = sc.read_h5ad("bin1_mutation/BIN1_Mutation.h5ad")

# Resolução de Leiden atual

res_teste = "0.5"

# Parâmetros congelados do preprocessing/clustering final.
N_TOP_HVG = 3000
N_PCS = 50
N_NEIGHBORS = 15

pasta_saida = f"bin1_mutation/figures/resolucao_{res_teste}"
os.makedirs(pasta_saida, exist_ok=True)

# Tabelas de evidência para a anotação (as figuras continuam na pasta antiga).
pasta_resultados_anotacao = Path(
    f"bin1_mutation/results/resolucao_{res_teste}/anotacao"
)
pasta_resultados_anotacao.mkdir(parents=True, exist_ok=True)

# Auditoria/reprodutibilidade do preprocessing.
pasta_resultados_preprocessamento = Path(
    f"bin1_mutation/results/resolucao_{res_teste}/preprocessamento"
)
pasta_resultados_preprocessamento.mkdir(parents=True, exist_ok=True)

# Preencher SOMENTE depois de revisar as evidências da nova análise sem Harmony.
ANOTACAO_FINAL = {}


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
        plt.savefig(f"{pasta_saida}/heatmap_{nome_arquivo}_bin1.png", dpi=300, bbox_inches="tight")
        #plt.show()
    else:
        print(f"A coluna '{marcador}' não foi encontrada em adata.obs")

# ===========================================================
#ETAPA 2: Clustering (Leiden) e Fine Tuning
# ===========================================================

# ===========================================================
# PREPROCESSAMENTO DE EXPRESSÃO PARA PCA/KNN — SEM HARMONY
# ===========================================================
# A matriz de entrada deste projeto é esperada em contagens brutas.
# Antes de PCA/KNN, diagnosticamos a matriz para evitar tanto:
#   (1) PCA diretamente em counts; quanto
#   (2) dupla normalização/log-transformação.
#
# Se X for compatível com counts:
#   - preserva counts em adata.layers['counts'];
#   - normalize_total(CP10k);
#   - log1p;
#   - PCA -> KNN em X_pca.
#
# Se X não parecer counts, só continuamos automaticamente se houver
# metadado de log1p no objeto; caso contrário, paramos por segurança.

def diagnosticar_matriz_expressao(X, max_nonzero=200000):
    if hasattr(X, "data"):
        valores = np.asarray(X.data, dtype=float)
    else:
        valores = np.asarray(X, dtype=float).ravel()

    valores = valores[np.isfinite(valores)]
    valores = valores[valores != 0]

    if valores.size == 0:
        raise RuntimeError("A matriz de expressão não possui valores não-zero para diagnóstico.")

    if valores.size > max_nonzero:
        rng = np.random.default_rng(0)
        idx = rng.choice(valores.size, size=max_nonzero, replace=False)
        amostra = valores[idx]
    else:
        amostra = valores

    frac_inteiros = float(
        np.mean(np.isclose(amostra, np.round(amostra), atol=1e-6))
    )

    return {
        "frac_inteiros": frac_inteiros,
        "minimo": float(np.min(amostra)),
        "mediana": float(np.median(amostra)),
        "maximo": float(np.max(amostra)),
    }


diag = diagnosticar_matriz_expressao(adata.X)

print("\nDiagnóstico da matriz de entrada para clustering:")
print(f"  fração de valores compatíveis com inteiros: {diag['frac_inteiros']:.2%}")
print(
    f"  mínimo={diag['minimo']:.4f} | "
    f"mediana={diag['mediana']:.4f} | "
    f"máximo={diag['maximo']:.4f}"
)

matriz_compativel_counts = (
    diag["frac_inteiros"] >= 0.95
    and diag["minimo"] >= 0
)

if not matriz_compativel_counts:
    raise RuntimeError(
        "A matriz X de BIN1_Mutation.h5ad não parece conter contagens brutas. "
        "Como este pipeline final foi definido especificamente para counts, "
        "a execução foi interrompida para evitar dupla transformação ou "
        "preprocessing inconsistente."
    )

print(
    "Matriz compatível com counts. "
    "Preservando contagens em layers['counts']..."
)

# ------------------------------------------------------------------
# Preservação das contagens brutas
# ------------------------------------------------------------------
adata.layers["counts"] = adata.X.copy()

# O objeto-fonte foi confirmado sem log1p; ainda assim, removemos eventual
# metadado residual nesta cópia de trabalho antes de aplicar a transformação.
adata.uns.pop("log1p", None)

# ------------------------------------------------------------------
# Normalização por profundidade + log1p
# ------------------------------------------------------------------
print("Normalizando para 10.000 counts por célula (CP10k)...")
sc.pp.normalize_total(
    adata,
    target_sum=1e4,
)

print("Aplicando log1p...")
sc.pp.log1p(adata)

# ------------------------------------------------------------------
# Seleção pré-especificada de 3.000 genes altamente variáveis
# ------------------------------------------------------------------
#
# flavor='seurat' é usado sobre a matriz já normalizada/log1p.
# subset=False preserva TODOS os genes no AnnData; a coluna
# adata.var['highly_variable'] apenas marca os genes usados pela PCA.
#
# Não usamos batch_key='donor_id' aqui porque a proposta desta análise é
# NÃO remover/corrigir sistematicamente variação entre doadores antes do Milo.
print(f"Selecionando {N_TOP_HVG} genes altamente variáveis (HVGs)...")
sc.pp.highly_variable_genes(
    adata,
    flavor="seurat",
    n_top_genes=N_TOP_HVG,
    subset=False,
)

if "highly_variable" not in adata.var.columns:
    raise RuntimeError(
        "A seleção de HVGs terminou sem criar adata.var['highly_variable']."
    )

n_hvg_obtidos = int(adata.var["highly_variable"].sum())

if n_hvg_obtidos < 100:
    raise RuntimeError(
        f"Número inesperadamente baixo de HVGs: {n_hvg_obtidos}. "
        "Execução interrompida."
    )

print(f"HVGs selecionados: {n_hvg_obtidos}")

# Exporta a lista exata de HVGs para auditoria/reprodutibilidade.
colunas_hvg = [
    c for c in [
        "highly_variable",
        "means",
        "dispersions",
        "dispersions_norm",
    ]
    if c in adata.var.columns
]

df_hvg = adata.var.loc[
    adata.var["highly_variable"],
    colunas_hvg,
].copy()

df_hvg.insert(
    0,
    "gene",
    df_hvg.index.astype(str),
)

df_hvg.to_csv(
    pasta_resultados_preprocessamento / "HVGs_usados_na_PCA.csv",
    index=False,
)

# Figura diagnóstica dos HVGs.
try:
    sc.pl.highly_variable_genes(
        adata,
        show=False,
    )
    plt.savefig(
        f"{pasta_saida}/Preprocessamento_HVGs.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")
except Exception as exc:
    print(
        "AVISO: não foi possível gerar a figura diagnóstica de HVGs: "
        f"{exc}"
    )
    plt.close("all")

# Salva um resumo explícito do preprocessing.
pd.DataFrame(
    [
        {
            "Arquivo_entrada": "bin1_mutation/BIN1_Mutation.h5ad",
            "N_celulas_pos_filtro_AD": int(adata.n_obs),
            "N_genes_entrada": int(adata.n_vars),
            "Fracao_valores_inteiros": diag["frac_inteiros"],
            "Valor_naozero_minimo": diag["minimo"],
            "Valor_naozero_mediana": diag["mediana"],
            "Valor_naozero_maximo": diag["maximo"],
            "Normalizacao": "normalize_total",
            "Target_sum": 10000,
            "Transformacao": "log1p",
            "HVG_flavor": "seurat",
            "HVG_n_top_solicitado": N_TOP_HVG,
            "HVG_n_obtido": n_hvg_obtidos,
            "PCA_n_components": N_PCS,
            "KNN_n_neighbors": N_NEIGHBORS,
            "KNN_representation": "X_pca",
            "KNN_metric": "euclidean",
            "Harmony": False,
            "Resolucao_Leiden_principal": res_teste,
        }
    ]
).to_csv(
    pasta_resultados_preprocessamento / "Resumo_preprocessamento_final.csv",
    index=False,
)

# ------------------------------------------------------------------
# PCA apenas sobre os HVGs
# ------------------------------------------------------------------
print(
    f"\nCalculando PCA SEM Harmony sobre {n_hvg_obtidos} HVGs "
    f"({N_PCS} componentes)..."
)

sc.pp.pca(
    adata,
    n_comps=N_PCS,
    mask_var="highly_variable",
    random_state=0,
)

print("Construindo grafo de vizinhança SEM Harmony...")
print(f"X_pca disponível: {adata.obsm['X_pca'].shape}")

sc.pp.neighbors(
    adata,
    n_neighbors=N_NEIGHBORS,
    use_rep="X_pca",
    metric="euclidean",
    method="umap",
    random_state=0,
)

# Fine tuning ao redor da resolução principal escolhida (0.50).
# É obrigatório incluir 0.50, pois ela será usada em Estado_Microglial.
resolucoes = [0.45, 0.475, 0.5, 0.525, 0.55]
for res in resolucoes:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_{res}', flavor="igraph", random_state=0)

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
plt.savefig(f"{pasta_saida}/Fine Tuning da Estabilidade dos Clusters - Micróglias.png", dpi=300, bbox_inches="tight" )
#plt.show()

#=============================================================
# ETAPA 3: Concordância exploratória com a taxonomia SEA-AD (ARI)
#=============================================================

# Clustree e ARI são usados aqui como diagnósticos de estabilidade/concordância; não definem sozinhos a identidade biológica dos clusters.

# Se provou ineficiente para essa dataset!!

gabarito = (
    adata.obs['Supertype'].astype('object')
    .where(adata.obs['Supertype'].notna(), 'Desconhecido')
    .astype(str)
)
resultados_ari = {}

# Calculando ARI para cada resolução testada

for res in resolucoes:
    predicao = adata.obs[f'leiden_{res}'].astype(str)
    score = adjusted_rand_score(gabarito, predicao)
    resultados_ari[res] = score
    print(f"Resolução {res}: ARI = {score:.4f}")

# Encontrando a vencedora matematicamente

res_vencedora = max(resultados_ari, key=resultados_ari.get)
print(f"\nMaior ARI entre as resoluções testadas: {res_vencedora}")

# Heatmap de contingência da resolução com maior ARI entre as testadas

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
plt.savefig(f"{pasta_saida}/Heatmap_Contigencia_Best_Resolution.png", dpi=300, bbox_inches="tight")
#plt.show()

# A resolução principal definida para esta análise é 0.50.

#============================================================
# ETAPA 4: Visualização UMAP
#============================================================

sc.tl.umap(adata)

'''clusters = {
    '0': 'Micróglia Homeostática Basal (ADAM7-AS1+, DLEU7+)',
    '1': 'Micróglia Homeostática Fagocítica (FRMD4A+, PLXDC2+)',
    '2': 'Micróglia Associada à Doença - DAM (SPP1+ alto, FTH1+)',
    '3': 'Ruído Técnico / Doublets Neuronais (CADM2+, NRG3+)',
    '4': 'Micróglia Ativada por Estresse Metabólico (HIF1A+, SPP1+ mod)'
}'''
# Mapeando os nomes matemáticos para uma nova coluna biológica
adata.obs['Estado_Microglial'] = adata.obs[f'leiden_{res_teste}'].astype(str)

sc.pl.umap(
    adata,
    color='Estado_Microglial',
    title=f"Projeção UMAP dos Clusters Microgliais (Resolução {res_teste}, sem Harmony)",
    palette='Set1',
    show=False
)
plt.savefig(f"{pasta_saida}/UMAP_Anotado.png", dpi=300, bbox_inches="tight")
#plt.show()

#=============================================================
# ETAPA 5: Anotação Celular e Limpeza de Ruído 
#=============================================================

# Lista de genes que são puro ruído técnico no cérebro
# (MALAT1 é o principal, mas filtra-se mitocondriais e ribossomais por segurança)

genes_ruido = [gene for gene in adata.var_names if gene == "MALAT1" or gene.startswith(('MT-', 'RPS', "RPL"))]

#Criando uma matriz nova livre desse ruído

adata_limpo = adata[:, ~adata.var_names.isin(genes_ruido)].copy()

#adata_limpo = adata_limpo[adata_limpo.obs['Estado_Microglial'] != 'Ruído Técnico / Doublets Neuronais (CADM2+, NRG3+)'].copy()

#adata_limpo.obs['Estado_Microglial'] = adata_limpo.obs['Estado_Microglial'].cat.remove_unused_categories()

# A matriz X já foi normalizada (CP10k) e log-transformada ANTES da PCA.
# Portanto, NÃO normalizamos/log-transformamos novamente aqui.
#
# As contagens brutas, quando a entrada foi diagnosticada como counts,
# permanecem preservadas em adata_limpo.layers['counts'] após o subset de genes.
#
# .raw guarda uma cópia da matriz normalizada/log1p para referência Scanpy.
# Para pseudobulk de contagens, usar explicitamente layers['counts'].
adata_limpo.raw = adata_limpo.copy()

sc.tl.rank_genes_groups(
    adata_limpo,
    groupby=f'leiden_{res_teste}',
    method='wilcoxon',
    reference='rest',
    key_added='marcadores_wilcoxon',
    pts=True,
    tie_correct=True,
    use_raw=False,
)

# Dotplot para top 5 genes de cada cluster

# Organizando o dotplot
sc.tl.dendrogram(adata_limpo, groupby=f'leiden_{res_teste}')

sc.pl.rank_genes_groups_dotplot(
    adata_limpo,
    n_genes=5,
    key='marcadores_wilcoxon',
    groupby=f'leiden_{res_teste}',
    title=f"Top 5 Genes Marcadores por Cluster (Leiden {res_teste})",
    show=False 
)

plt.savefig(f"{pasta_saida}/top5_marcadores_limpo.png", dpi=300, bbox_inches="tight")
#plt.show()


# ============================================================
# ETAPA 5.5: EVIDÊNCIAS PARA ANOTAÇÃO BIOLÓGICA — SEM HARMONY
# ============================================================
# A identidade dos clusters é inferida independentemente do resultado BIN1/Milo.
# São combinados marcadores, programas gênicos, referências SEA-AD, possíveis
# outras linhagens e representatividade entre doadores. ANOTACAO_FINAL permanece
# vazia na primeira execução para evitar nomes automáticos excessivamente fortes.

print("\n=== Caracterização rigorosa para anotação biológica ===")
cluster_col = f'leiden_{res_teste}'

PROGRAMAS_MICROGLIAIS = {
    'Homeostase': ['P2RY12', 'P2RY13', 'TMEM119', 'CX3CR1', 'GPR34', 'SALL1', 'HEXB', 'CSF1R', 'SIGLEC10'],
    'Doenca_DAM_like': ['APOE', 'TREM2', 'LPL', 'CST7', 'GPNMB', 'SPP1', 'CD9', 'ITGAX', 'CTSB', 'CTSD', 'APOC1', 'LGALS3'],
    'Fagocitose_Lipidios': ['APOC1', 'LPL', 'FABP5', 'LIPA', 'CTSB', 'CTSD', 'FCER1G', 'TYROBP', 'AXL', 'MSR1'],
    'Apresentacao_Antigeno': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'CD74', 'B2M', 'CIITA'],
    'Interferon': ['ISG15', 'IFIT1', 'IFIT2', 'IFIT3', 'MX1', 'OAS1', 'OAS2', 'STAT1', 'IRF7'],
    'Inflamacao': ['IL1B', 'TNF', 'NFKBIA', 'PTGS2', 'CCL2', 'CXCL8', 'CCL3', 'CCL4'],
    'Proliferacao': ['MKI67', 'TOP2A', 'STMN1', 'TYMS', 'UBE2C', 'CENPF'],
    'Resposta_Imediata_Estresse': ['FOS', 'JUN', 'JUNB', 'DUSP1', 'ATF3', 'HSPA1A', 'HSPA1B'],
}

PAINEIS_CONTAMINACAO = {
    'Neuronal': ['SNAP25', 'SYT1', 'RBFOX3', 'SLC17A7', 'GAD1', 'GAD2', 'CAMK2A'],
    'Oligodendroglial': ['MBP', 'PLP1', 'MOG', 'MAG', 'MOBP', 'OLIG1', 'OLIG2'],
    'Astrocitario': ['GFAP', 'AQP4', 'SLC1A2', 'SLC1A3', 'ALDH1L1', 'SOX9'],
    'Endotelial': ['CLDN5', 'FLT1', 'KDR', 'PECAM1', 'VWF', 'RAMP2'],
    'Linfocitario': ['CD3D', 'CD3E', 'TRAC', 'CD247', 'MS4A1', 'CD79A', 'NKG7'],
}

REFERENCIAS_SEAAD = ['Class', 'Subclass', 'Supertype']


def genes_presentes(genes):
    universo = set(adata_limpo.var_names)
    return [g for g in genes if g in universo]


def zscore_colunas(df):
    z = df.copy()
    for col in z.columns:
        sd = z[col].std(ddof=0)
        z[col] = 0.0 if (sd == 0 or np.isnan(sd)) else (z[col] - z[col].mean()) / sd
    return z


def calcular_scores(painel, prefixo):
    colunas, status = [], []
    for nome, genes in painel.items():
        presentes = genes_presentes(genes)
        status.extend({'Painel': nome, 'Gene': g, 'Presente_no_dataset': g in presentes} for g in genes)
        if len(presentes) < 2:
            print(f"AVISO: {nome}: apenas {len(presentes)} gene(s) presente(s); score ignorado.")
            continue
        score = f'{prefixo}_{nome}'
        sc.tl.score_genes(adata_limpo, presentes, score_name=score, random_state=0, use_raw=False)
        colunas.append(score)
    return colunas, pd.DataFrame(status)


def resumo_scores(colunas, prefixo):
    if not colunas:
        return pd.DataFrame(), pd.DataFrame()
    base = adata_limpo.obs[[cluster_col] + colunas].copy()
    media = base.groupby(cluster_col, observed=True)[colunas].mean()
    mediana = base.groupby(cluster_col, observed=True)[colunas].median()
    media.columns = [c.replace(prefixo, '') for c in media.columns]
    mediana.columns = [c.replace(prefixo, '') for c in mediana.columns]
    media.index, mediana.index = media.index.astype(str), mediana.index.astype(str)
    return media, mediana


def salvar_heatmap(df, arquivo, titulo, cmap='RdBu_r', center=0, annot=True):
    if df.empty:
        return
    plt.figure(figsize=(max(8, len(df.columns) + 3), max(6, 0.55 * len(df.index) + 3)))
    sns.heatmap(df, cmap=cmap, center=center, annot=annot, fmt='.2f')
    plt.title(titulo)
    plt.ylabel(f'Cluster Leiden {res_teste}')
    plt.tight_layout()
    plt.savefig(f'{pasta_saida}/{arquivo}', dpi=300, bbox_inches='tight')
    plt.close()


# 1) Top 50 em CSV; Top 10 em figura. Top 50 serve como catálogo amplo de
# evidências, não como exigência de que todos os 50 genes definam o cluster.
clusters_anotacao = sorted(adata_limpo.obs[cluster_col].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
lista_marcadores = []
for cluster in clusters_anotacao:
    df = sc.get.rank_genes_groups_df(adata_limpo, group=cluster, key='marcadores_wilcoxon').copy()
    df.insert(0, 'Cluster_Leiden', cluster)
    lista_marcadores.append(df)
    df.head(50).to_csv(pasta_resultados_anotacao / f'Cluster_{cluster}_Top50_marcadores.csv', index=False)

marcadores_todos = pd.concat(lista_marcadores, ignore_index=True)
marcadores_todos.to_csv(pasta_resultados_anotacao / 'Marcadores_todos_clusters.csv', index=False)
marcadores_todos.groupby('Cluster_Leiden', group_keys=False).head(50).to_csv(
    pasta_resultados_anotacao / 'Top50_marcadores_por_cluster.csv', index=False
)

sc.pl.rank_genes_groups_dotplot(
    adata_limpo, n_genes=10, key='marcadores_wilcoxon', groupby=cluster_col,
    dendrogram=True, title=f'Top 10 genes marcadores — Leiden {res_teste} (sem Harmony)', show=False
)
plt.savefig(f'{pasta_saida}/ANOTACAO_DotPlot_Top10_Marcadores.png', dpi=300, bbox_inches='tight')
plt.close('all')

# 2) Programas microgliais.
scores_prog, status_prog = calcular_scores(PROGRAMAS_MICROGLIAIS, 'ScorePrograma')
media_prog, mediana_prog = resumo_scores(scores_prog, 'ScorePrograma_')
z_prog = zscore_colunas(media_prog)
media_prog.to_csv(pasta_resultados_anotacao / 'Scores_programas_microgliais_media.csv')
mediana_prog.to_csv(pasta_resultados_anotacao / 'Scores_programas_microgliais_mediana.csv')
z_prog.to_csv(pasta_resultados_anotacao / 'Scores_programas_microgliais_Z_por_cluster.csv')
salvar_heatmap(z_prog, 'ANOTACAO_Heatmap_Programas_Microgliais_Z.png', 'Programas microgliais — z-score entre clusters')

painel_prog = {k: genes_presentes(v) for k, v in PROGRAMAS_MICROGLIAIS.items()}
painel_prog = {k: v for k, v in painel_prog.items() if v}
if painel_prog:
    sc.pl.dotplot(adata_limpo, painel_prog, groupby=cluster_col, standard_scale='var', show=False,
                  title=f'Painéis microgliais — Leiden {res_teste} (sem Harmony)')
    plt.savefig(f'{pasta_saida}/ANOTACAO_DotPlot_Paineis_Microgliais.png', dpi=300, bbox_inches='tight')
    plt.close('all')

# 3) Possíveis outras linhagens/contaminação.
scores_cont, status_cont = calcular_scores(PAINEIS_CONTAMINACAO, 'ScoreContaminacao')
media_cont, mediana_cont = resumo_scores(scores_cont, 'ScoreContaminacao_')
z_cont = zscore_colunas(media_cont)
media_cont.to_csv(pasta_resultados_anotacao / 'Scores_contaminacao_media.csv')
mediana_cont.to_csv(pasta_resultados_anotacao / 'Scores_contaminacao_mediana.csv')
z_cont.to_csv(pasta_resultados_anotacao / 'Scores_contaminacao_Z_por_cluster.csv')
salvar_heatmap(z_cont, 'ANOTACAO_Heatmap_Contaminacao_Z.png', 'Possíveis outras linhagens — z-score entre clusters')

painel_cont = {k: genes_presentes(v) for k, v in PAINEIS_CONTAMINACAO.items()}
painel_cont = {k: v for k, v in painel_cont.items() if v}
if painel_cont:
    sc.pl.dotplot(adata_limpo, painel_cont, groupby=cluster_col, standard_scale='var', show=False,
                  title=f'Painéis de possíveis outras linhagens — Leiden {res_teste}')
    plt.savefig(f'{pasta_saida}/ANOTACAO_DotPlot_Paineis_Contaminacao.png', dpi=300, bbox_inches='tight')
    plt.close('all')

status_prog['Categoria'] = 'Programa_microglial'
status_cont['Categoria'] = 'Possivel_contaminacao'
pd.concat([status_prog, status_cont], ignore_index=True).to_csv(
    pasta_resultados_anotacao / 'Status_genes_paineis.csv', index=False
)

# 4) Hierarquia/taxonomia SEA-AD presente no próprio objeto.
resumos_ref = {}
for ref_col in REFERENCIAS_SEAAD:
    if ref_col not in adata_limpo.obs.columns:
        print(f"Referência SEA-AD '{ref_col}' ausente; pulando.")
        continue
    ref = adata_limpo.obs[ref_col].astype('object').where(adata_limpo.obs[ref_col].notna(), 'Desconhecido').astype(str)
    tab = pd.crosstab(ref, adata_limpo.obs[cluster_col].astype(str), normalize='columns') * 100
    tab.to_csv(pasta_resultados_anotacao / f'Composicao_SEAAD_{ref_col}_por_cluster.csv')
    plt.figure(figsize=(max(10, tab.shape[1] + 3), max(6, 0.45 * tab.shape[0] + 3)))
    sns.heatmap(tab, cmap='Blues', annot=(tab.shape[0] <= 20), fmt='.1f')
    plt.title(f'Leiden {res_teste} vs SEA-AD {ref_col} (%) — sem Harmony')
    plt.xlabel(f'Cluster Leiden {res_teste}'); plt.ylabel(f'SEA-AD {ref_col}'); plt.tight_layout()
    plt.savefig(f'{pasta_saida}/ANOTACAO_Heatmap_Leiden_vs_SEAAD_{ref_col}.png', dpi=300, bbox_inches='tight')
    plt.close()
    registros = [{'Cluster_Leiden': str(c), f'SEAAD_{ref_col}_dominante': tab[c].idxmax(),
                  f'SEAAD_{ref_col}_percentual': float(tab[c].max())} for c in tab.columns]
    resumos_ref[ref_col] = pd.DataFrame(registros).set_index('Cluster_Leiden')
    resumos_ref[ref_col].to_csv(pasta_resultados_anotacao / f'Resumo_SEAAD_{ref_col}_dominante.csv')

# 5) Representatividade donor-level.
contagem_cd = pd.crosstab(adata_limpo.obs['donor_id'].astype(str), adata_limpo.obs[cluster_col].astype(str))
contagem_cd.to_csv(pasta_resultados_anotacao / 'Contagem_celulas_doador_por_cluster.csv')
registros = []
for cluster in contagem_cd.columns:
    v = contagem_cd[cluster]; n = int(v.sum()); nd = int((v > 0).sum()); vmax = int(v.max())
    registros.append({
        'Cluster_Leiden': str(cluster), 'N_celulas': n,
        'Percentual_total_celulas': 100 * n / adata_limpo.n_obs,
        'N_doadores_com_cluster': nd,
        'Percentual_doadores_representados': 100 * nd / adata_limpo.obs['donor_id'].nunique(),
        'Mediana_celulas_por_doador': float(v.median()), 'Max_celulas_um_doador': vmax,
        'Percentual_cluster_maior_doador': 100 * vmax / n if n else np.nan,
    })
resumo_doadores = pd.DataFrame(registros).set_index('Cluster_Leiden')
resumo_doadores.to_csv(pasta_resultados_anotacao / 'Resumo_representatividade_por_doador.csv')

# 6) Tabela integrada, template manual e figura-resumo.
registros = []
for cluster in clusters_anotacao:
    genes = marcadores_todos.loc[marcadores_todos['Cluster_Leiden'] == cluster, 'names'].dropna().astype(str).head(10).tolist()
    r = {'Cluster_Leiden': cluster, 'Top3_marcadores': '; '.join(genes[:3]), 'Top10_marcadores': '; '.join(genes)}
    if cluster in resumo_doadores.index:
        r.update(resumo_doadores.loc[cluster].to_dict())
    if cluster in z_prog.index:
        r['Programa_relativamente_mais_alto'] = z_prog.loc[cluster].idxmax(); r['Z_programa_max'] = float(z_prog.loc[cluster].max())
    if cluster in z_cont.index:
        r['Sinal_contaminacao_relativamente_mais_alto'] = z_cont.loc[cluster].idxmax(); r['Z_contaminacao_max'] = float(z_cont.loc[cluster].max())
    for _, resumo_ref in resumos_ref.items():
        if cluster in resumo_ref.index:
            r.update(resumo_ref.loc[cluster].to_dict())
    registros.append(r)

resumo_anotacao = pd.DataFrame(registros)
resumo_anotacao.to_csv(pasta_resultados_anotacao / 'Resumo_evidencias_por_cluster.csv', index=False)
template = resumo_anotacao.copy()
template['Nome_biologico_final'] = ''; template['Confianca'] = ''; template['Justificativa'] = ''
template.to_csv(pasta_resultados_anotacao / 'Template_anotacao_manual.csv', index=False)

cols_fig = [c for c in ['Cluster_Leiden', 'N_celulas', 'N_doadores_com_cluster', 'Top3_marcadores',
                        'Programa_relativamente_mais_alto', 'SEAAD_Supertype_dominante',
                        'Sinal_contaminacao_relativamente_mais_alto'] if c in resumo_anotacao.columns]
fig, ax = plt.subplots(figsize=(18, max(5, 0.75 * len(resumo_anotacao) + 2.5)))
ax.axis('off')
tab_fig = ax.table(cellText=resumo_anotacao[cols_fig].astype(str).values, colLabels=cols_fig,
                   loc='center', cellLoc='left', colLoc='center')
tab_fig.auto_set_font_size(False); tab_fig.set_fontsize(8); tab_fig.scale(1, 1.6)
ax.set_title(f'Resumo integrado para anotação — Leiden {res_teste} (sem Harmony)', fontsize=14, pad=18)
plt.tight_layout(); plt.savefig(f'{pasta_saida}/ANOTACAO_Resumo_Integrado_Clusters.png', dpi=300, bbox_inches='tight'); plt.close()

# 7) UMAP descritivo: mantém o número do cluster e mostra evidências objetivas.
mapa_evidencia = {}
for _, r in resumo_anotacao.iterrows():
    cluster = str(r['Cluster_Leiden'])
    programa = r.get('Programa_relativamente_mais_alto', 'sem_score')
    supertype = r.get('SEAAD_Supertype_dominante', 'sem_Supertype')
    mapa_evidencia[cluster] = f"{cluster} | {programa} | {supertype}"
adata_limpo.obs['Rotulo_Anotacao_Descritiva'] = adata_limpo.obs[cluster_col].astype(str).map(mapa_evidencia).astype('category')
sc.pl.umap(adata_limpo, color='Rotulo_Anotacao_Descritiva', legend_loc='right margin', show=False,
           title=f'Leiden {res_teste} — evidências de anotação (sem Harmony)')
plt.savefig(f'{pasta_saida}/ANOTACAO_UMAP_Resumo_Evidencias.png', dpi=300, bbox_inches='tight'); plt.close('all')

# 8) UMAP com nomes biológicos somente após revisão manual.
if ANOTACAO_FINAL:
    esperados, fornecidos = set(clusters_anotacao), {str(k) for k in ANOTACAO_FINAL}
    if esperados != fornecidos:
        raise ValueError(f'ANOTACAO_FINAL incompleta. Faltantes={sorted(esperados-fornecidos)}; extras={sorted(fornecidos-esperados)}')
    adata_limpo.obs['Estado_Microglial_Biologico'] = adata_limpo.obs[cluster_col].astype(str).map(ANOTACAO_FINAL).astype('category')
    sc.pl.umap(adata_limpo, color='Estado_Microglial_Biologico', legend_loc='right margin', show=False,
               title=f'Anotação biológica final — Leiden {res_teste} (sem Harmony)')
    plt.savefig(f'{pasta_saida}/ANOTACAO_UMAP_Estados_Biologicos_Finais.png', dpi=300, bbox_inches='tight'); plt.close('all')
    print("ANOTACAO_FINAL aplicada em 'Estado_Microglial_Biologico'.")
else:
    print('ANOTACAO_FINAL vazio: nomes biológicos finais não foram forçados nesta execução.')

print(f'Resultados de anotação: {pasta_resultados_anotacao}')

# ============================================================
# Etapa 6: Impacto da Mutação nas Proporções Celulares
# ============================================================

# Atualizando a variável subclass por leiden_{res_teste}: Impacto da mutação nas proporções

contagem_celulas = pd.crosstab( # Cruzando o genótipo com os clusters validados
    adata_limpo.obs['Mutacao_BIN1'],
    adata_limpo.obs['Estado_Microglial'],
    normalize='index'
) * 100

# Forçando a ordem lógica (biológica) das colunas
'''ordem_desejada = [
    'Micróglia Homeostática Basal (ADAM7-AS1+, DLEU7+)', 
    'Micróglia Homeostática Fagocítica (FRMD4A+, PLXDC2+)', 
    'Micróglia Ativada por Estresse Metabólico (HIF1A+, SPP1+ mod)', 
    'Micróglia Associada à Doença - DAM (SPP1+ alto, FTH1+)' 
]'''
#contagem_celulas = contagem_celulas[ordem_desejada]

#Gráfico de barras empilhadas

contagem_celulas.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.title(f"Distribuição dos Estados Microgliais (Leiden {res_teste}) por Genótipo BIN1")
plt.ylabel("Proporção das células (%)")
plt.xlabel("Genótipo BIN1 (rs6733839)")
plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
plt.tight_layout()
plt.savefig(f"{pasta_saida}/Proporção_clusters_BIN1_limpo.png", dpi=300, bbox_inches="tight")
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
    #contagem_braak = contagem_braak[ordem_desejada]

    # Gráfico de barras empilhadas
    contagem_braak.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title(f"Distribuição dos Estados Microgliais (Leiden {res_teste}) por Estágio de Braak")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("Estágio de Braak (Progressão da Patologia)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{pasta_saida}/Proporcao_clusters_Braak_limpo.png", dpi=300, bbox_inches="tight")
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
    #contagem_thal = contagem_thal[ordem_desejada]

    # Plot
    contagem_thal.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title(f"Distribuição dos Estados Microgliais (Leiden {res_teste}) por Fase de Thal")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("Fase de Thal (Acúmulo de Beta-amiloide)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{pasta_saida}/Proporcao_clusters_Thal_limpo.png", dpi=300, bbox_inches="tight")
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
    #contagem_cerad = contagem_cerad[ordem_desejada]

    # Plot
    contagem_cerad.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title(f"Distribuição dos Estados Microgliais (Leiden {res_teste}) por Escore CERAD")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("Escore CERAD (Densidade de Placas Neuríticas)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{pasta_saida}/Proporcao_clusters_CERAD_limpo.png", dpi=300, bbox_inches="tight")
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
    #contagem_adnc = contagem_adnc[ordem_desejada]
    ordem_adnc = ['Low', 'Intermediate', 'High']
    contagem_adnc = contagem_adnc.reindex(ordem_adnc)

    # Plot
    contagem_adnc.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
    plt.title(f"Distribuição dos Estados Microgliais (Leiden {res_teste}) por Gravidade ADNC")
    plt.ylabel("Proporção das células (%)")
    plt.xlabel("ADNC (Gravidade Neuropatológica Global)")
    plt.legend(title="Cluster (Estado Celular)", bbox_to_anchor=(1.05, 1), loc ='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{pasta_saida}/Proporcao_clusters_ADNC_limpo.png", dpi=300, bbox_inches="tight")
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
    
    plt.figure(figsize=(10, 6))
    
    # Boxplot Base
    sns.boxplot(
        data=df_ancestralidade,
        x='Genótipo',
        y='Expressão',
        hue='Ancestralidade',
        palette='Set2',
        showfliers=False,       # Oculta os outliers do boxplot (para o stripplot mostrá-los)
        width=0.6,              # Deixa a caixa um pouco mais fina para ficar elegante
        boxprops={'alpha': 0.7} # Dá uma leve transparência na cor da caixa
    )
    
    # Stripplot (Pontos reais por cima)
    sns.stripplot(
        data=df_ancestralidade,
        x='Genótipo',
        y='Expressão',
        hue='Ancestralidade',
        dodge=True,             # Força os pontos a acompanharem a posição de cada caixa
        alpha=0.3,              # Pontos semi-transparentes para mostrar acúmulo (zeros)
        size=3,
        color='black',          # Todos os pontos pretos (contrasta com a caixa colorida)
        legend=False            # Impede que crie uma legenda duplicada
    )
    
    plt.title(f"Distribuição da Expressão de {gene} por Genótipo e Ancestralidade")
    plt.ylabel(f"Expressão Normalizada ({gene})")
    plt.xlabel("Genótipo BIN1 (rs6733839)")
    
    # Ajustando a legenda para pegar apenas as cores do boxplot
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:len(df_ancestralidade['Ancestralidade'].unique())], 
               labels[:len(df_ancestralidade['Ancestralidade'].unique())], 
               title="Ancestralidade", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{pasta_saida}/Expressão_{gene}_Ancestralidade_limpo.png", dpi=300, bbox_inches="tight")
    plt.close()
    #plt.show()
    
# ===========================================================
# ETAPA 7.5: Expressão de BIN1 por Cluster (Estado Microglial)
# ===========================================================

print("\n=== Gerando gráficos de expressão de BIN1 por cluster ===")

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

# Removendo NAs
df_bin1_cluster = df_bin1_cluster.dropna(subset=['Genótipo_Recessivo'])

plt.figure(figsize=(14, 7))

# Boxplot Base
sns.boxplot(
    data=df_bin1_cluster,
    x='Estado_Microglial',
    y='Expressão_BIN1',
    hue='Genótipo_Recessivo',
    palette={'0/0 + 0/1': '#bcbddc', '1/1 (Homozigoto Recessivo)': '#a1d99b'},
    showfliers=False,
    width=0.6,
    boxprops={'alpha': 0.8}
)

# Stripplot (Pontos reais por cima)
sns.stripplot(
    data=df_bin1_cluster,
    x='Estado_Microglial',
    y='Expressão_BIN1',
    hue='Genótipo_Recessivo',
    dodge=True,
    alpha=0.25,
    size=2.5,
    color='black',
    legend=False
)

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title("Expressão de BIN1 por Estado Microglial e Genótipo (Modelo Recessivo)")
plt.ylabel("Expressão Normalizada (BIN1)", fontsize=12)
plt.xlabel("")

# Corrigindo a legenda para evitar duplicação pelo stripplot
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Modelo Genético", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)

plt.tight_layout()
plt.savefig(f"{pasta_saida}/Boxplot_BIN1_Clusters_e_Recessivo.png", dpi=300, bbox_inches="tight")
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
plt.savefig(f"{pasta_saida}/Painel_Integrado_Abundancia_Genotipo.png", dpi=300, bbox_inches="tight")
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
plt.savefig(f"{pasta_saida}/Painel_Integrado_Recessivo.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(f"{pasta_saida}/Painel_Stripplots_Braak.png", dpi=300, bbox_inches="tight")
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
plt.savefig(f"{pasta_saida}/Painel_Stripplots_Thal.png", dpi=300, bbox_inches="tight")
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
plt.savefig(f"{pasta_saida}/Painel_Stripplots_CERAD.png", dpi=300, bbox_inches="tight")
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
plt.savefig(f"{pasta_saida}/Painel_Stripplots_ADNC.png", dpi=300, bbox_inches="tight")
plt.close()
    
print("Análises de ADNC concluídas com sucesso!")

adata_limpo.uns['Integracao_pipeline'] = 'Sem_Harmony'
adata_limpo.uns['Grafo_vizinhos_representacao'] = 'X_pca'
adata_limpo.uns['Grafo_vizinhos_n_neighbors'] = N_NEIGHBORS
adata_limpo.uns['Resolucao_Leiden_principal'] = res_teste
adata_limpo.uns['Preprocessamento_clustering'] = (
    f'counts -> normalize_total_1e4 -> log1p -> '
    f'{N_TOP_HVG}_HVGs_seurat -> PCA_{N_PCS}PCs -> '
    f'KNN_{N_NEIGHBORS}_euclidean; sem Harmony'
)
adata_limpo.uns['HVG_flavor'] = 'seurat'
adata_limpo.uns['HVG_n_top_solicitado'] = N_TOP_HVG
adata_limpo.uns['HVG_n_obtido'] = n_hvg_obtidos
adata_limpo.uns['PCA_n_components'] = N_PCS
adata_limpo.uns['Counts_preservados_em_layer'] = bool(
    'counts' in adata_limpo.layers
)

# O arquivo antigo não é sobrescrito: este é o novo objeto recomendado.
arquivo_saida_h5ad = (
    f"bin1_mutation/"
    f"adata_limpo_anotado_sem_harmony_res{res_teste}.h5ad"
)
adata_limpo.write_h5ad(arquivo_saida_h5ad, compression='gzip')
print(f"\nNovo objeto SEM Harmony salvo em: {arquivo_saida_h5ad}")