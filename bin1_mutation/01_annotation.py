# ============================================================
# 02_finalizar_anotacao_res0_5.py
#
# Finalização da anotação biológica do Leiden 0.50
# SEM recalcular preprocessing, PCA, KNN, Leiden ou UMAP.
#
# Este script:
#   1) carrega o .h5ad definitivo da resolução 0.50;
#   2) lê as tabelas de evidência já produzidas pelo script 01;
#   3) corrige a interpretação automática dos z-scores:
#        - programa: só chama "enriquecido" se Z >= 1.0;
#        - contaminação: só chama "sinal forte" se Z >= 1.5;
#   4) adiciona anotação biológica, confiança e flags de QC;
#   5) gera um UMAP final usando AS MESMAS coordenadas já existentes;
#   6) salva um novo .h5ad, sem sobrescrever o objeto anterior.
#
# IMPORTANTE:
# - os limiares de Z abaixo são heurísticos/descritivos, NÃO p-valores;
# - a anotação continua independente dos resultados de BIN1/Milo;
# - clusters donor-specific/contaminados são explicitamente sinalizados.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RESOLUCAO = "0.5"

ARQUIVO_ENTRADA = (
    "bin1_mutation/"
    "adata_limpo_anotado_sem_harmony_res0.5.h5ad"
)

ARQUIVO_SAIDA = (
    "bin1_mutation/"
    "adata_limpo_anotado_sem_harmony_res0.5_FINAL.h5ad"
)

PASTA_RESULTADOS = Path(
    f"bin1_mutation/results/resolucao_{RESOLUCAO}/anotacao"
)

PASTA_FIGURAS = Path(
    f"bin1_mutation/figures/resolucao_{RESOLUCAO}"
)

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

# Saídas adicionais solicitadas para apresentação/reunião.
PASTA_FIGURAS_BIN1 = PASTA_FIGURAS / "apresentacao_BIN1"
PASTA_RESULTADOS_BIN1 = PASTA_RESULTADOS / "apresentacao_BIN1"
PASTA_FIGURAS_BIN1.mkdir(parents=True, exist_ok=True)
PASTA_RESULTADOS_BIN1.mkdir(parents=True, exist_ok=True)

CLUSTER_COL = f"leiden_{RESOLUCAO}"

# Limiares DESCRITIVOS.
# Não correspondem a significância estatística.
LIMIAR_Z_PROGRAMA = 1.0
LIMIAR_Z_CONTAMINACAO = 1.5


# ============================================================
# ANOTAÇÃO BIOLÓGICA CONSERVADORA
# ============================================================
#
# Estes nomes combinam:
# - marcadores diferenciais;
# - programas gênicos;
# - referência SEA-AD;
# - representatividade por doador;
# - sinais de outras linhagens.
#
# Eles NÃO usam os resultados BIN1/Milo.
#
# Os nomes podem ser refinados futuramente sem alterar PCA/KNN/Milo.

ANOTACAO_FINAL = {
    "0": "Microglia/PVM ativada-fagocítica",
    "1": "Microglia ativada — resposta imediata/estresse",
    "2": "Provável contaminação/doublet neuronal-oligodendroglial",
    "3": "Microglia homeostática FRMD4A+/INPP5D+",
    "4": "Microglia homeostática/imunovigilância",
    "5": "Microglia inflamatória/metabólica",
    "6": "Microglia com assinatura neuronal — QC",
    "7": "Microglia homeostática/intermediária",
    "8": "Microglia ativada/apresentação de antígeno",
    "9": "Microglia ativada/interferon-inflamatória",
    "10": "Donor-specific / assinatura Y-linked — QC",
    "11": "PVM-like / MRC1+ — donor-enriched",
    "12": "Fagocítica/apresentação de antígeno — donor-specific",
}

CONFIANCA_FINAL = {
    "0": "Moderada",
    "1": "Moderada",
    "2": "Alta para flag de QC",
    "3": "Alta",
    "4": "Alta",
    "5": "Moderada-alta",
    "6": "Moderada-baixa",
    "7": "Moderada",
    "8": "Moderada",
    "9": "Moderada",
    "10": "Alta para flag de QC",
    "11": "Moderada-baixa",
    "12": "Baixa-moderada",
}

FLAG_QC_FINAL = {
    "0": "OK",
    "1": "Donor-enriched; interpretar com cautela",
    "2": "Forte sinal neuronal/oligodendroglial; provável contaminação/doublet",
    "3": "OK",
    "4": "OK",
    "5": "OK",
    "6": "Assinatura neuronal moderada; revisar",
    "7": "OK; alguma concentração por doador",
    "8": "Donor-enriched; interpretar com cautela",
    "9": "Donor-enriched; interpretar com cautela",
    "10": "Fortemente donor-specific; não tratar como estado populacional",
    "11": "Donor-enriched; interpretar com cautela",
    "12": "Fortemente donor-specific; não tratar como estado populacional",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ordenar_clusters(valores):
    """Ordena IDs de clusters numericamente quando possível."""
    return sorted(
        [str(v) for v in valores],
        key=lambda x: int(x) if x.isdigit() else x,
    )


def carregar_csv_indexado(caminho, index_col=0):
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo necessário não encontrado: {caminho}"
        )

    df = pd.read_csv(caminho, index_col=index_col)
    df.index = df.index.astype(str)
    return df


def listar_enriquecidos(linha, limiar):
    """
    Retorna todos os programas com Z >= limiar,
    ordenados do maior para o menor.

    Se nenhum atingir o limiar, retorna mensagem explícita em vez
    de escolher artificialmente o 'menos baixo'.
    """
    linha = pd.to_numeric(linha, errors="coerce").dropna()

    selecionados = linha[linha >= limiar].sort_values(ascending=False)

    if selecionados.empty:
        return "Sem enriquecimento relativo forte"

    return "; ".join(
        f"{nome} (Z={valor:.2f})"
        for nome, valor in selecionados.items()
    )


def maior_z(linha):
    linha = pd.to_numeric(linha, errors="coerce").dropna()

    if linha.empty:
        return np.nan, np.nan

    nome = linha.idxmax()
    valor = float(linha.max())

    return nome, valor



# ============================================================
# FUNÇÕES PARA AS FIGURAS ADICIONAIS DE BIN1
# ============================================================

def extrair_expressao_gene(adata_obj, gene):
    """Retorna a expressão em adata.X para um gene, em vetor 1D."""
    if gene not in adata_obj.var_names:
        raise KeyError(f"O gene {gene!r} não existe em adata.var_names.")

    x = adata_obj[:, gene].X

    if hasattr(x, "toarray"):
        valores = x.toarray().ravel()
    else:
        valores = np.asarray(x).ravel()

    return valores.astype(float)


def ordenar_clusters_numericamente(valores):
    return sorted(
        [str(v) for v in valores],
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )


# ============================================================
# 1. CARREGAMENTO DO OBJETO DEFINITIVO
# ============================================================

print("\n=== Carregando objeto Leiden 0.50 definitivo ===")

adata = sc.read_h5ad(ARQUIVO_ENTRADA)

if CLUSTER_COL not in adata.obs.columns:
    raise KeyError(
        f"A coluna {CLUSTER_COL!r} não existe em adata.obs."
    )

if "X_umap" not in adata.obsm:
    raise KeyError(
        "X_umap não existe no objeto. "
        "Este script não recalcula UMAP por princípio."
    )

clusters = ordenar_clusters(
    adata.obs[CLUSTER_COL].astype(str).unique()
)

esperados = set(clusters)

for nome, dicionario in [
    ("ANOTACAO_FINAL", ANOTACAO_FINAL),
    ("CONFIANCA_FINAL", CONFIANCA_FINAL),
    ("FLAG_QC_FINAL", FLAG_QC_FINAL),
]:
    fornecidos = {str(k) for k in dicionario}

    if fornecidos != esperados:
        raise ValueError(
            f"{nome} não corresponde exatamente aos clusters observados.\n"
            f"Faltantes: {sorted(esperados - fornecidos)}\n"
            f"Extras: {sorted(fornecidos - esperados)}"
        )

print(f"Shape: {adata.shape}")
print(f"Clusters Leiden {RESOLUCAO}: {clusters}")
print(f"Doadores: {adata.obs['donor_id'].nunique()}")


# ============================================================
# 2. LEITURA DAS EVIDÊNCIAS PRODUZIDAS PELO SCRIPT 01
# ============================================================

z_programas = carregar_csv_indexado(
    PASTA_RESULTADOS / "Scores_programas_microgliais_Z_por_cluster.csv"
)

z_contaminacao = carregar_csv_indexado(
    PASTA_RESULTADOS / "Scores_contaminacao_Z_por_cluster.csv"
)

representatividade = carregar_csv_indexado(
    PASTA_RESULTADOS / "Resumo_representatividade_por_doador.csv"
)

marcadores = pd.read_csv(
    PASTA_RESULTADOS / "Top50_marcadores_por_cluster.csv"
)

marcadores["Cluster_Leiden"] = (
    marcadores["Cluster_Leiden"].astype(str)
)

arquivo_seaad = (
    PASTA_RESULTADOS /
    "Resumo_SEAAD_Supertype_dominante.csv"
)

if arquivo_seaad.exists():
    seaad = carregar_csv_indexado(arquivo_seaad)
else:
    seaad = pd.DataFrame(index=clusters)
    print(
        "AVISO: resumo SEA-AD Supertype não encontrado; "
        "a coluna SEA-AD ficará vazia."
    )


# ============================================================
# 3. RESUMO REVISADO DOS Z-SCORES
# ============================================================

registros = []

for cluster in clusters:

    r = {
        "Cluster_Leiden": cluster,
        "Nome_biologico_final": ANOTACAO_FINAL[cluster],
        "Confianca": CONFIANCA_FINAL[cluster],
        "Flag_QC": FLAG_QC_FINAL[cluster],
    }

    # --------------------------------------------------------
    # Marcadores
    # --------------------------------------------------------
    genes_cluster = (
        marcadores.loc[
            marcadores["Cluster_Leiden"] == cluster,
            "names",
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    r["Top3_marcadores"] = "; ".join(genes_cluster[:3])
    r["Top10_marcadores"] = "; ".join(genes_cluster[:10])

    # --------------------------------------------------------
    # Programas microgliais
    # --------------------------------------------------------
    if cluster in z_programas.index:
        linha_prog = z_programas.loc[cluster]

        prog_max, z_prog_max = maior_z(linha_prog)

        r["Programa_com_maior_Z"] = prog_max
        r["Z_programa_max"] = z_prog_max

        r[
            f"Programas_enriquecidos_Z>={LIMIAR_Z_PROGRAMA:g}"
        ] = listar_enriquecidos(
            linha_prog,
            LIMIAR_Z_PROGRAMA,
        )

        if z_prog_max >= LIMIAR_Z_PROGRAMA:
            r["Interpretacao_programa"] = (
                f"{prog_max} relativamente enriquecido "
                f"(Z={z_prog_max:.2f})"
            )
        else:
            r["Interpretacao_programa"] = (
                "Sem programa fortemente enriquecido"
            )

    # --------------------------------------------------------
    # Sinais de outras linhagens
    # --------------------------------------------------------
    if cluster in z_contaminacao.index:
        linha_cont = z_contaminacao.loc[cluster]

        cont_max, z_cont_max = maior_z(linha_cont)

        r["Outra_linhagem_com_maior_Z"] = cont_max
        r["Z_outra_linhagem_max"] = z_cont_max

        r[
            f"Sinais_fortes_outras_linhagens_Z>={LIMIAR_Z_CONTAMINACAO:g}"
        ] = listar_enriquecidos(
            linha_cont,
            LIMIAR_Z_CONTAMINACAO,
        )

        if z_cont_max >= LIMIAR_Z_CONTAMINACAO:
            r["Interpretacao_outra_linhagem"] = (
                f"Sinal relativo forte: {cont_max} "
                f"(Z={z_cont_max:.2f})"
            )
        else:
            r["Interpretacao_outra_linhagem"] = (
                "Sem sinal relativo forte de outra linhagem"
            )

    # --------------------------------------------------------
    # Representatividade por doador
    # --------------------------------------------------------
    if cluster in representatividade.index:
        for coluna in [
            "N_celulas",
            "Percentual_total_celulas",
            "N_doadores_com_cluster",
            "Percentual_doadores_representados",
            "Mediana_celulas_por_doador",
            "Max_celulas_um_doador",
            "Percentual_cluster_maior_doador",
        ]:
            if coluna in representatividade.columns:
                r[coluna] = representatividade.loc[cluster, coluna]

    # --------------------------------------------------------
    # SEA-AD
    # --------------------------------------------------------
    if cluster in seaad.index:
        for coluna in seaad.columns:
            r[coluna] = seaad.loc[cluster, coluna]

    registros.append(r)


resumo_revisado = pd.DataFrame(registros)

resumo_revisado.to_csv(
    PASTA_RESULTADOS /
    "Resumo_evidencias_por_cluster_REVISADO.csv",
    index=False,
)

print(
    "\nResumo revisado salvo em:",
    PASTA_RESULTADOS /
    "Resumo_evidencias_por_cluster_REVISADO.csv",
)


# ============================================================
# 4. COLUNAS FINAIS NO AnnData
# ============================================================

cluster_str = adata.obs[CLUSTER_COL].astype(str)

adata.obs["Estado_Microglial_Biologico"] = (
    cluster_str
    .map(ANOTACAO_FINAL)
    .astype("category")
)

adata.obs["Confianca_Anotacao"] = (
    cluster_str
    .map(CONFIANCA_FINAL)
    .astype("category")
)

adata.obs["Flag_QC_Anotacao"] = (
    cluster_str
    .map(FLAG_QC_FINAL)
    .astype("category")
)

# Preserva explicitamente o Leiden numérico usado na interpretação.
adata.obs["Cluster_Leiden_0_5"] = (
    cluster_str.astype("category")
)


# ============================================================
# 5. UMAP FINAL
# ============================================================
#
# NÃO há sc.tl.umap() aqui.
# Portanto as coordenadas são exatamente as mesmas do objeto produzido
# pelo script 01. Só mudamos os rótulos e a paleta.

print("\n=== Gerando UMAP final com nomes biológicos ===")

sc.pl.umap(
    adata,
    color="Estado_Microglial_Biologico",
    legend_loc="right margin",
    palette="tab20",
    show=False,
    title=(
        "Estados microgliais — Leiden 0.50 "
        "(sem Harmony)"
    ),
)

plt.savefig(
    PASTA_FIGURAS /
    "ANOTACAO_UMAP_Estados_Biologicos_Finais.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")


# UMAP adicional apenas com flags de QC.
sc.pl.umap(
    adata,
    color="Flag_QC_Anotacao",
    legend_loc="right margin",
    palette="tab20",
    show=False,
    title=(
        "Flags de QC da anotação — Leiden 0.50 "
        "(sem Harmony)"
    ),
)

plt.savefig(
    PASTA_FIGURAS /
    "ANOTACAO_UMAP_Flags_QC.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")



# ============================================================
# 5.5. FIGURAS ADICIONAIS DE BIN1 PARA APRESENTAÇÃO
# ============================================================
#
# IMPORTANTE:
# - adata.X neste objeto contém expressão normalizada por CP10k + log1p;
# - o violin plot e o UMAP são DESCRITIVOS em nível celular;
# - a regressão de expressão por genótipo usa UMA observação por doador,
#   evitando pseudorreplicação por células;
# - na regressão, o genótipo é codificado como dosagem aditiva:
#       0/0 -> 0
#       0/1 -> 1
#       1/1 -> 2
#   Essa regressão de expressão é uma análise complementar e NÃO muda
#   o contraste recessivo usado no Milo.
# ============================================================

print("\n=== Gerando figuras adicionais de BIN1 para apresentação ===")

GENE_BIN1 = "BIN1"
exp_bin1 = extrair_expressao_gene(adata, GENE_BIN1)

df_bin1 = pd.DataFrame(
    {
        "donor_id": adata.obs["donor_id"].astype(str).to_numpy(),
        "Mutacao_BIN1": adata.obs["Mutacao_BIN1"].astype(str).to_numpy(),
        "Cluster_Leiden_0_5": adata.obs[CLUSTER_COL].astype(str).to_numpy(),
        "Estado_Microglial_Biologico": (
            adata.obs["Estado_Microglial_Biologico"].astype(str).to_numpy()
        ),
        "Expressao_BIN1_log1p_CP10k": exp_bin1,
    },
    index=adata.obs_names,
)

# ------------------------------------------------------------
# A) VIOLIN PLOT — expressão de BIN1 por estado microglial
# ------------------------------------------------------------

ordem_clusters = ordenar_clusters_numericamente(
    adata.obs[CLUSTER_COL].astype(str).unique()
)

ordem_estados = [
    ANOTACAO_FINAL[c]
    for c in ordem_clusters
]

resumo_estado = (
    df_bin1
    .groupby(
        ["Cluster_Leiden_0_5", "Estado_Microglial_Biologico"],
        observed=True,
    )
    .agg(
        N_celulas=("Expressao_BIN1_log1p_CP10k", "size"),
        Media=("Expressao_BIN1_log1p_CP10k", "mean"),
        Mediana=("Expressao_BIN1_log1p_CP10k", "median"),
        Fracao_expressando=(
            "Expressao_BIN1_log1p_CP10k",
            lambda x: float(np.mean(np.asarray(x, dtype=float) > 0)),
        ),
    )
    .reset_index()
)

resumo_estado.to_csv(
    PASTA_RESULTADOS_BIN1 / "BIN1_expressao_resumo_por_estado_microglial.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(12, 10))

sns.violinplot(
    data=df_bin1,
    y="Estado_Microglial_Biologico",
    x="Expressao_BIN1_log1p_CP10k",
    order=ordem_estados,
    inner="quartile",
    cut=0,
    density_norm="width",
    linewidth=0.8,
    ax=ax,
)

ax.set_title(
    "Expressão normalizada de BIN1 por estado microglial\n"
    "(CP10k + log1p; visualização descritiva em nível celular)"
)
ax.set_xlabel("Expressão normalizada de BIN1 (log1p CP10k)")
ax.set_ylabel("Estado microglial")
plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS_BIN1 / "Violin_BIN1_por_estado_microglial.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")


# ------------------------------------------------------------
# B) EXPRESSÃO DE BIN1 POR GENÓTIPO — donor-level
#    Scatter + regressão linear da dosagem 0/1/2
# ------------------------------------------------------------

geno_dosagem = {
    "0/0": 0,
    "0/1": 1,
    "1/1": 2,
}

donor_geno_n = (
    df_bin1
    .groupby("donor_id", observed=True)["Mutacao_BIN1"]
    .nunique(dropna=False)
)

if (donor_geno_n != 1).any():
    raise RuntimeError(
        "Mutacao_BIN1 não é constante dentro de todos os donor_id."
    )

df_donor_bin1 = (
    df_bin1
    .groupby("donor_id", observed=True)
    .agg(
        Mutacao_BIN1=("Mutacao_BIN1", "first"),
        Expressao_BIN1_media=("Expressao_BIN1_log1p_CP10k", "mean"),
        Expressao_BIN1_mediana=("Expressao_BIN1_log1p_CP10k", "median"),
        Fracao_celulas_BIN1_positivo=(
            "Expressao_BIN1_log1p_CP10k",
            lambda x: float(np.mean(np.asarray(x, dtype=float) > 0)),
        ),
        N_celulas=("Expressao_BIN1_log1p_CP10k", "size"),
    )
    .reset_index()
)

df_donor_bin1["Dosagem_BIN1"] = (
    df_donor_bin1["Mutacao_BIN1"].map(geno_dosagem)
)

if df_donor_bin1["Dosagem_BIN1"].isna().any():
    inesperados = sorted(
        df_donor_bin1.loc[
            df_donor_bin1["Dosagem_BIN1"].isna(),
            "Mutacao_BIN1",
        ].astype(str).unique()
    )
    raise RuntimeError(
        f"Genótipos BIN1 inesperados para regressão: {inesperados}"
    )

X_reg = sm.add_constant(
    df_donor_bin1[["Dosagem_BIN1"]].astype(float),
    has_constant="add",
)

y_reg = df_donor_bin1["Expressao_BIN1_media"].astype(float)

modelo = sm.OLS(
    y_reg,
    X_reg,
).fit(
    cov_type="HC3"
)

beta = float(modelo.params["Dosagem_BIN1"])
se_beta = float(modelo.bse["Dosagem_BIN1"])
p_beta = float(modelo.pvalues["Dosagem_BIN1"])
intercepto = float(modelo.params["const"])
r2 = float(modelo.rsquared)
n_donors_reg = int(len(df_donor_bin1))

pd.DataFrame(
    [
        {
            "Modelo": (
                "OLS donor-level: expressão média BIN1 ~ dosagem genotípica"
            ),
            "Codificacao_genotipo": "0/0=0; 0/1=1; 1/1=2",
            "N_doadores": n_donors_reg,
            "Intercepto": intercepto,
            "Beta_por_alelo": beta,
            "SE_HC3": se_beta,
            "PValue_HC3": p_beta,
            "R2": r2,
            "Observacao": (
                "Cada ponto é um doador; erro-padrão/p-valor HC3 robusto."
            ),
        }
    ]
).to_csv(
    PASTA_RESULTADOS_BIN1 / "BIN1_regressao_expressao_por_genotipo.csv",
    index=False,
)

df_donor_bin1.to_csv(
    PASTA_RESULTADOS_BIN1 / "BIN1_expressao_donor_level_por_genotipo.csv",
    index=False,
)

rng = np.random.default_rng(42)
x_real = df_donor_bin1["Dosagem_BIN1"].to_numpy(dtype=float)
x_jitter = x_real + rng.normal(
    loc=0.0,
    scale=0.045,
    size=len(x_real),
)

fig, ax = plt.subplots(figsize=(8.5, 6.5))

ax.scatter(
    x_jitter,
    y_reg.to_numpy(dtype=float),
    s=45,
    alpha=0.75,
)

x_linha = np.linspace(0, 2, 200)
y_linha = intercepto + beta * x_linha

ax.plot(
    x_linha,
    y_linha,
    linewidth=2,
)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["0/0", "0/1", "1/1"])
ax.set_xlabel("Genótipo BIN1 (dosagem do alelo: 0, 1, 2)")
ax.set_ylabel("Expressão média normalizada de BIN1 por doador")
ax.set_title(
    "Expressão de BIN1 por genótipo — regressão linear donor-level"
)

texto = (
    f"β por alelo = {beta:.4f}\n"
    f"SE(HC3) = {se_beta:.4f}\n"
    f"p = {p_beta:.3e}\n"
    f"R² = {r2:.3f}\n"
    f"n = {n_donors_reg} doadores"
)

ax.text(
    0.03,
    0.97,
    texto,
    transform=ax.transAxes,
    va="top",
    ha="left",
    bbox={
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.9,
    },
)

plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS_BIN1 / "Scatter_regressao_BIN1_por_genotipo_donor_level.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")


# ------------------------------------------------------------
# C) UMAP — expressão normalizada de BIN1
# ------------------------------------------------------------
#
# O limite superior da escala de cor é visualmente cortado no percentil 99
# para evitar que poucas células com expressão muito alta comprimam toda a
# escala. Nenhum valor do AnnData é alterado.

sc.pl.umap(
    adata,
    color=GENE_BIN1,
    use_raw=False,
    cmap="viridis",
    vmin=0,
    vmax="p99",
    show=False,
    title=(
        "UMAP — expressão normalizada de BIN1 "
        "(CP10k + log1p; escala de cor até p99)"
    ),
)

plt.savefig(
    PASTA_FIGURAS_BIN1 / "UMAP_expressao_normalizada_BIN1.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")

print(
    "Figuras adicionais de BIN1 salvas em:",
    PASTA_FIGURAS_BIN1,
)


# ============================================================
# 6. FIGURA-RESUMO FINAL
# ============================================================

colunas_figura = [
    "Cluster_Leiden",
    "Nome_biologico_final",
    "Confianca",
    "N_celulas",
    "N_doadores_com_cluster",
    "Top3_marcadores",
    "Interpretacao_programa",
    "Interpretacao_outra_linhagem",
    "Flag_QC",
]

colunas_figura = [
    c for c in colunas_figura
    if c in resumo_revisado.columns
]

tabela_fig = resumo_revisado[
    colunas_figura
].copy()

fig, ax = plt.subplots(
    figsize=(
        22,
        max(7, 0.9 * len(tabela_fig) + 3),
    )
)

ax.axis("off")

tab = ax.table(
    cellText=tabela_fig.astype(str).values,
    colLabels=tabela_fig.columns,
    loc="center",
    cellLoc="left",
    colLoc="center",
)

tab.auto_set_font_size(False)
tab.set_fontsize(7.5)
tab.scale(1, 1.7)

ax.set_title(
    "Anotação biológica final e QC — "
    f"Leiden {RESOLUCAO} (sem Harmony)",
    fontsize=14,
    pad=18,
)

plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS /
    "ANOTACAO_Resumo_Final_Clusters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")


# ============================================================
# 7. TABELA FINAL COM UMA LINHA POR CLUSTER
# ============================================================

tabela_final = resumo_revisado.copy()

tabela_final.to_csv(
    PASTA_RESULTADOS /
    "Anotacao_final_clusters_Leiden0.5.csv",
    index=False,
)


# ============================================================
# 8. PROVENIÊNCIA / METADADOS
# ============================================================

adata.uns["Anotacao_biologica"] = (
    "Manual/auditavel: marcadores + programas genicos + "
    "SEA-AD + representatividade por doador + QC de outras linhagens"
)

adata.uns["Anotacao_independente_BIN1_Milo"] = True

adata.uns["Limiar_descritivo_Z_programa"] = (
    LIMIAR_Z_PROGRAMA
)

adata.uns["Limiar_descritivo_Z_contaminacao"] = (
    LIMIAR_Z_CONTAMINACAO
)

adata.uns["Nota_zscore_anotacao"] = (
    "Z-scores comparam o score medio de cada programa entre clusters; "
    "nao sao p-valores nem medidas absolutas de identidade celular."
)


# ============================================================
# 9. SALVAMENTO
# ============================================================

adata.write_h5ad(
    ARQUIVO_SAIDA,
    compression="gzip",
)

print("\n=== Finalização concluída ===")
print("Objeto de entrada preservado:")
print(f"  {ARQUIVO_ENTRADA}")
print("Novo objeto anotado:")
print(f"  {ARQUIVO_SAIDA}")
print("UMAP original NÃO foi recalculado.")
print(
    "As coordenadas X_umap foram apenas reutilizadas "
    "com novos rótulos."
)