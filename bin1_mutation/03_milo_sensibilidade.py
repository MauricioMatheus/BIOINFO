import os

import numpy as np
import scanpy as sc
import pertpy as pt
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 03_milo_sensibilidade.py
#
# ANÁLISE DE SENSIBILIDADE DO MILO SEM HARMONY POR donor_id
#
# Objetivo:
# - manter a mesma coorte, os mesmos metadados, o mesmo modelo
#   recessivo e as mesmas três rotas estatísticas do 02_milo.py;
# - reconstruir SOMENTE o grafo KNN a partir de X_pca, sem usar
#   X_pca_harmony;
# - construir novas neighborhoods Milo nesse novo grafo;
# - comparar os resultados agregados com a análise principal.
#
# IMPORTANTE:
# As neighborhoods produzidas aqui NÃO devem ser comparadas
# diretamente por ID com as neighborhoods do 02_milo.py, porque
# a geometria do grafo foi reconstruída. A comparação apropriada
# é feita por:
#   1) estatísticas globais das rotas;
#   2) direção/magnitude dos efeitos;
#   3) distribuição dos candidatos entre os clusters Leiden 0.45;
#   4) padrões donor-level.
# =========================================================


# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
sc.settings.n_jobs = n_cpus

print(f"Usando {n_cpus} CPUs.")


# =========================================================
# PASTAS DE SAÍDA
# =========================================================

pasta_figuras = "figures/milo_sensibilidade_sem_harmony"
pasta_tabelas = "results/milo_sensibilidade_sem_harmony"

pasta_figuras_donor = f"{pasta_figuras}/donor_level"
pasta_tabelas_donor = f"{pasta_tabelas}/donor_level"

pasta_comparacao = f"{pasta_tabelas}/comparacao_com_harmony"
pasta_figuras_comparacao = f"{pasta_figuras}/comparacao_com_harmony"

for pasta in [
    pasta_figuras,
    pasta_tabelas,
    pasta_figuras_donor,
    pasta_tabelas_donor,
    pasta_comparacao,
    pasta_figuras_comparacao,
]:
    os.makedirs(pasta, exist_ok=True)


# =========================================================
# ETAPA 1: CARREGAMENTO E PREPARAÇÃO DOS METADADOS
# =========================================================

print("Carregando o objeto anotado...")

adata = sc.read_h5ad("adata_limpo_anotado.h5ad")


# ---------------------------------------------------------
# Nomes seguros para fórmulas no R
# ---------------------------------------------------------

adata.obs.rename(
    columns={
        "Age.at.death": "Age_at_death",
        "Braak.stage": "Braak_stage",
        "Thal.phase": "Thal_phase",
    },
    inplace=True,
)


# ---------------------------------------------------------
# Modelo recessivo do BIN1
# ---------------------------------------------------------

mapeamento_recessivo = {
    "0/0": "Nao_Recessivo",
    "0/1": "Nao_Recessivo",
    "1/1": "Recessivo",
}

adata.obs["Modelo_Recessivo"] = (
    adata.obs["Mutacao_BIN1"]
    .map(mapeamento_recessivo)
)


# ---------------------------------------------------------
# Padronização das faixas de idade
# ---------------------------------------------------------

mapeamento_idade = {
    "65 to 77 years old": "65_77",
    "78 to 89 years old": "78_89",
    "90+ years old": "90_plus",
}

adata.obs["Age_at_death"] = (
    adata.obs["Age_at_death"]
    .map(mapeamento_idade)
)


# ---------------------------------------------------------
# A anotação Leiden 0.45 será mantida APENAS como referência
# biológica/interpretativa.
#
# Não vamos recalcular Leiden aqui, pois queremos alterar
# somente o espaço usado para construir o KNN/Milo.
# ---------------------------------------------------------

if "Estado_Microglial" not in adata.obs.columns:
    raise ValueError(
        "A coluna 'Estado_Microglial' não foi encontrada em adata.obs."
    )


# =========================================================
# ETAPA 1.1: LIMPEZA GLOBAL DE NAs
# =========================================================

print(
    f"Células antes da limpeza para o Milo: "
    f"{adata.n_obs}"
)

adata = adata[
    ~adata.obs["Modelo_Recessivo"].isna()
    & ~adata.obs["ADNC"].isna()
    & ~adata.obs["Braak_stage"].isna()
    & ~adata.obs["Thal_phase"].isna()
    & ~adata.obs["Age_at_death"].isna()
    & ~adata.obs["donor_id"].isna()
    & ~adata.obs["Estado_Microglial"].isna()
].copy()

print(
    f"Células após a limpeza para o Milo: "
    f"{adata.n_obs}"
)


# =========================================================
# ETAPA 1.2: FATORES CATEGÓRICOS
# =========================================================

colunas_categoricas = [
    "Age_at_death",
    "ADNC",
    "Braak_stage",
    "Thal_phase",
    "Modelo_Recessivo",
]

for col in colunas_categoricas:
    adata.obs[col] = (
        adata.obs[col]
        .astype("category")
        .cat.remove_unused_categories()
    )

adata.obs["Estado_Microglial"] = (
    adata.obs["Estado_Microglial"]
    .astype("category")
    .cat.remove_unused_categories()
)


# ---------------------------------------------------------
# Níveis sintaticamente válidos para limma::makeContrasts()
# ---------------------------------------------------------

adata.obs["Braak_stage"] = (
    adata.obs["Braak_stage"]
    .cat.rename_categories(
        lambda x: str(x).replace(" ", "_")
    )
)

adata.obs["Thal_phase"] = (
    adata.obs["Thal_phase"]
    .cat.rename_categories(
        lambda x: str(x).replace(" ", "_")
    )
)


# ---------------------------------------------------------
# Ordem explícita do modelo recessivo
# ---------------------------------------------------------

adata.obs["Modelo_Recessivo"] = (
    adata.obs["Modelo_Recessivo"]
    .cat.reorder_categories(
        [
            "Nao_Recessivo",
            "Recessivo",
        ]
    )
)


print(
    "\n===== CATEGORIAS USADAS NO MODELO ====="
)

for col in colunas_categoricas:
    print(
        f"{col}: "
        f"{list(adata.obs[col].cat.categories)}"
    )

print(
    "Estado_Microglial (Leiden 0.45 de referência): "
    f"{list(adata.obs['Estado_Microglial'].cat.categories)}"
)

print(
    "=======================================\n"
)


# =========================================================
# ETAPA 1.3: CONSISTÊNCIA DOS METADADOS POR DOADOR
# =========================================================

covariaveis_doador = [
    "Age_at_death",
    "ADNC",
    "Braak_stage",
    "Thal_phase",
    "Modelo_Recessivo",
]

variacao_por_doador = (
    adata.obs
    .groupby(
        "donor_id",
        observed=True,
    )[covariaveis_doador]
    .nunique(
        dropna=False
    )
)

doadores_inconsistentes = (
    variacao_por_doador[
        variacao_por_doador
        .gt(1)
        .any(axis=1)
    ]
)

if not doadores_inconsistentes.empty:
    print(
        "\nERRO: existem doadores com mais de um "
        "valor clínico/genético."
    )

    print(
        doadores_inconsistentes
    )

    raise ValueError(
        "Metadados inconsistentes dentro de um mesmo donor_id."
    )


pacientes_unicos = (
    adata.obs
    .drop_duplicates(
        subset="donor_id"
    )
    .copy()
)

print(
    f"Número de doadores usados no Milo: "
    f"{len(pacientes_unicos)}"
)


print(
    "\n===== MODELO RECESSIVO POR DOADOR ====="
)

print(
    pacientes_unicos[
        "Modelo_Recessivo"
    ].value_counts()
)

print(
    "=======================================\n"
)


print(
    "===== IDADE POR DOADOR ====="
)

print(
    pacientes_unicos[
        "Age_at_death"
    ].value_counts()
)

print(
    "===========================\n"
)


grupos_presentes = set(
    pacientes_unicos[
        "Modelo_Recessivo"
    ]
    .dropna()
    .astype(str)
)

grupos_necessarios = {
    "Nao_Recessivo",
    "Recessivo",
}

if not grupos_necessarios.issubset(
    grupos_presentes
):
    raise ValueError(
        "O modelo recessivo não possui os grupos "
        "'Nao_Recessivo' e 'Recessivo'."
    )


# =========================================================
# ETAPA 1.4: METADADOS PARA INSPEÇÃO DONOR-LEVEL
# =========================================================

total_celulas_por_doador = (
    adata.obs[
        "donor_id"
    ]
    .astype(str)
    .value_counts()
)

metadados_doador = (
    pacientes_unicos[
        [
            "donor_id",
            "Modelo_Recessivo",
            "Age_at_death",
            "ADNC",
            "Braak_stage",
            "Thal_phase",
        ]
    ]
    .copy()
)

metadados_doador[
    "donor_id"
] = (
    metadados_doador[
        "donor_id"
    ]
    .astype(str)
)

metadados_doador[
    "Modelo_Recessivo"
] = (
    metadados_doador[
        "Modelo_Recessivo"
    ]
    .astype(str)
)


# =========================================================
# ETAPA 2: RECONSTRUÇÃO DO GRAFO SEM HARMONY
# =========================================================

print(
    "\n============================================"
)

print(
    "RECONSTRUINDO O GRAFO KNN SEM HARMONY"
)

print(
    "============================================\n"
)


# ---------------------------------------------------------
# X_pca é obrigatório para esta análise.
#
# O objetivo NÃO é recalcular a PCA em uma matriz diferente,
# e sim utilizar a representação PCA não corrigida já existente
# no .h5ad, isolando o efeito da etapa Harmony sobre o grafo.
# ---------------------------------------------------------

if "X_pca" not in adata.obsm:
    raise ValueError(
        "A representação 'X_pca' não foi encontrada em adata.obsm. "
        "Sem ela, esta análise de sensibilidade não deve prosseguir "
        "automaticamente, pois recalcular PCA aqui mudaria mais de "
        "uma etapa do pipeline."
    )


x_pca = np.asarray(
    adata.obsm[
        "X_pca"
    ]
)

if x_pca.shape[0] != adata.n_obs:
    raise ValueError(
        "X_pca não possui o mesmo número de células de adata."
    )

if not np.isfinite(
    x_pca
).all():
    raise ValueError(
        "X_pca contém valores NaN ou infinitos."
    )


print(
    f"X_pca encontrada: "
    f"{x_pca.shape[0]} células x "
    f"{x_pca.shape[1]} componentes."
)


if "X_pca_harmony" in adata.obsm:
    print(
        "X_pca_harmony também está presente, "
        "mas NÃO será usada nesta análise."
    )
else:
    print(
        "X_pca_harmony não foi encontrada. "
        "A análise continuará usando X_pca."
    )


# ---------------------------------------------------------
# Recupera os parâmetros do grafo original quando disponíveis.
#
# No pipeline principal, sc.pp.neighbors() foi executado sobre
# X_pca_harmony. Para a sensibilidade, tentamos manter o mesmo
# número de vizinhos, métrica e método, trocando apenas use_rep
# para X_pca.
# ---------------------------------------------------------

params_grafo_original = {}

if (
    "neighbors" in adata.uns
    and isinstance(
        adata.uns["neighbors"],
        dict,
    )
):
    params_grafo_original = (
        adata.uns[
            "neighbors"
        ]
        .get(
            "params",
            {},
        )
    )


n_neighbors = int(
    params_grafo_original.get(
        "n_neighbors",
        15,
    )
)

metric = params_grafo_original.get(
    "metric",
    "euclidean",
)

method = params_grafo_original.get(
    "method",
    "umap",
)


if not isinstance(
    metric,
    str,
):
    metric = "euclidean"

if method not in {
    "umap",
    "gauss",
}:
    method = "umap"


print(
    "Parâmetros usados no novo KNN:"
)

print(
    "  use_rep      = X_pca"
)

print(
    f"  n_neighbors  = {n_neighbors}"
)

print(
    f"  metric       = {metric}"
)

print(
    f"  method       = {method}"
)


# ---------------------------------------------------------
# Preserva o UMAP anterior apenas dentro deste objeto em memória.
# ---------------------------------------------------------

if "X_umap" in adata.obsm:
    adata.obsm[
        "X_umap_harmony_referencia"
    ] = (
        adata.obsm[
            "X_umap"
        ]
        .copy()
    )


neighbors_key_no_harmony = (
    "neighbors_no_harmony"
)


print(
    "\nCalculando novo grafo de vizinhos com X_pca..."
)

sc.pp.neighbors(
    adata,
    n_neighbors=n_neighbors,
    use_rep="X_pca",
    metric=metric,
    method=method,
    random_state=0,
    key_added=neighbors_key_no_harmony,
)


# ---------------------------------------------------------
# UMAP SEM HARMONY
#
# O UMAP não entra no teste edgeR.
# Ele é recalculado apenas para representar graficamente
# a geometria do novo KNN.
# ---------------------------------------------------------

print(
    "Calculando UMAP baseado no grafo sem Harmony..."
)

sc.tl.umap(
    adata,
    neighbors_key=neighbors_key_no_harmony,
    random_state=0,
)

adata.obsm[
    "X_umap_no_harmony"
] = (
    adata.obsm[
        "X_umap"
    ]
    .copy()
)


# ---------------------------------------------------------
# Salva a configuração do novo grafo
# ---------------------------------------------------------

config_grafo = pd.DataFrame(
    [
        {
            "Analise": "Sensibilidade sem Harmony",
            "Representacao": "X_pca",
            "Neighbors_key": neighbors_key_no_harmony,
            "N_cells": adata.n_obs,
            "N_pcs_disponiveis": x_pca.shape[1],
            "N_neighbors": n_neighbors,
            "Metric": metric,
            "Method": method,
            "Random_state": 0,
        }
    ]
)

config_grafo.to_csv(
    f"{pasta_tabelas}/Configuracao_grafo_sem_harmony.csv",
    index=False,
)


# =========================================================
# ETAPA 2.1: UMAPS DIAGNÓSTICOS SEM HARMONY
# =========================================================

print(
    "Salvando UMAPs diagnósticos sem Harmony..."
)


sc.pl.umap(
    adata,
    color="Estado_Microglial",
    title=(
        "UMAP sem Harmony - "
        "Leiden 0.45 como anotação de referência"
    ),
    show=False,
)

plt.savefig(
    f"{pasta_figuras}/"
    "UMAP_sem_Harmony_Leiden045_referencia.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    "all"
)


sc.pl.umap(
    adata,
    color="Modelo_Recessivo",
    title=(
        "UMAP sem Harmony - "
        "Modelo Recessivo BIN1"
    ),
    show=False,
)

plt.savefig(
    f"{pasta_figuras}/"
    "UMAP_sem_Harmony_Modelo_Recessivo.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(
    "all"
)


# =========================================================
# ETAPA 3: CONSTRUÇÃO DO MILO NO NOVO GRAFO
# =========================================================

print(
    "\nPreparando o objeto Milo para a "
    "análise sem Harmony..."
)

milo = pt.tl.Milo()

mdata = milo.load(
    adata
)


print(
    "Construindo neighborhoods Milo "
    "explicitamente no grafo sem Harmony..."
)

milo.make_nhoods(
    mdata[
        "rna"
    ],
    neighbors_key=neighbors_key_no_harmony,
    prop=0.1,
)


print(
    "Contando células por neighborhood "
    "e por doador..."
)

mdata = milo.count_nhoods(
    mdata,
    sample_col="donor_id",
)


for col in colunas_categoricas:
    if col in mdata[
        "milo"
    ].obs.columns:
        mdata[
            "milo"
        ].obs[col] = (
            mdata[
                "milo"
            ].obs[col]
            .astype(
                "category"
            )
            .cat
            .remove_unused_categories()
        )


mdata[
    "milo"
].X = sp.csr_matrix(
    mdata[
        "milo"
    ].X.astype(int)
)


print(
    "Dimensão da matriz Milo sem Harmony "
    "(doadores x neighborhoods): "
    f"{mdata['milo'].shape}"
)


# =========================================================
# ETAPA 3.1: ANOTAÇÃO DAS NEIGHBORHOODS
# =========================================================

print(
    "\nAnotando neighborhoods sem Harmony "
    "com o Leiden 0.45 de referência..."
)

milo.annotate_nhoods(
    mdata,
    anno_col="Estado_Microglial",
)


if (
    "nhood_annotation"
    in mdata[
        "milo"
    ].var.columns
):
    print(
        "\n===== NEIGHBORHOODS SEM HARMONY "
        "POR LEIDEN 0.45 ====="
    )

    print(
        mdata[
            "milo"
        ]
        .var[
            "nhood_annotation"
        ]
        .value_counts()
    )

    print(
        "=========================================\n"
    )


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def matriz_para_numpy(x):
    if sp.issparse(x):
        return x.toarray()

    return np.asarray(x)


def exportar_resultados(
    mdata_resultado,
    prefixo,
):
    """
    Salva:
    - todas as neighborhoods;
    - Top 20 por menor PValue;
    - Top 20 por maior |logFC|;
    - resumo por Leiden 0.45.
    """

    resultados = (
        mdata_resultado[
            "milo"
        ]
        .var
        .copy()
    )


    resultados.insert(
        0,
        "Neighborhood",
        resultados.index.astype(
            str
        ),
    )


    if (
        "frac_annotation"
        in mdata_resultado[
            "milo"
        ].varm
    ):
        frac = matriz_para_numpy(
            mdata_resultado[
                "milo"
            ].varm[
                "frac_annotation"
            ]
        )

        labels = [
            str(x)
            for x in
            mdata_resultado[
                "milo"
            ].uns[
                "annotation_labels"
            ]
        ]

        frac_df = pd.DataFrame(
            frac,
            index=resultados.index,
            columns=[
                f"frac_Leiden045_{x}"
                for x in labels
            ],
        )

        for col in frac_df.columns:
            resultados[
                col
            ] = (
                frac_df[
                    col
                ]
                .values
            )


    resultados[
        "abs_logFC"
    ] = (
        resultados[
            "logFC"
        ]
        .abs()
    )


    resultados_ordenados = (
        resultados
        .sort_values(
            by=[
                "PValue",
                "SpatialFDR",
            ]
        )
    )

    resultados_ordenados.to_csv(
        f"{pasta_tabelas}/"
        f"{prefixo}_todos_neighborhoods.csv",
        index=False,
    )


    top20_pvalue = (
        resultados
        .sort_values(
            "PValue"
        )
        .head(20)
    )

    top20_pvalue.to_csv(
        f"{pasta_tabelas}/"
        f"{prefixo}_Top20_menor_PValue.csv",
        index=False,
    )


    top20_logfc = (
        resultados
        .sort_values(
            "abs_logFC",
            ascending=False,
        )
        .head(20)
    )

    top20_logfc.to_csv(
        f"{pasta_tabelas}/"
        f"{prefixo}_Top20_maior_abs_logFC.csv",
        index=False,
    )


    if (
        "nhood_annotation"
        in resultados.columns
    ):
        resumo_cluster = (
            resultados
            .groupby(
                "nhood_annotation",
                observed=True,
            )
            .agg(
                n_neighborhoods=(
                    "Neighborhood",
                    "size",
                ),
                logFC_medio=(
                    "logFC",
                    "mean",
                ),
                logFC_mediano=(
                    "logFC",
                    "median",
                ),
                menor_PValue=(
                    "PValue",
                    "min",
                ),
                menor_FDR=(
                    "FDR",
                    "min",
                ),
                menor_SpatialFDR=(
                    "SpatialFDR",
                    "min",
                ),
            )
            .sort_values(
                "menor_PValue"
            )
        )

        resumo_cluster.to_csv(
            f"{pasta_tabelas}/"
            f"{prefixo}_resumo_por_Leiden045.csv"
        )


    return resultados


def gerar_graficos_estatisticos(
    resultados,
    prefixo,
    titulo,
):
    """
    Salva:
    - histograma de SpatialFDR;
    - histograma de logFC;
    - logFC de todas as neighborhoods ordenado.
    """

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        resultados[
            "SpatialFDR"
        ].dropna(),
        bins=50,
    )

    plt.xlabel(
        "Spatial FDR"
    )

    plt.ylabel(
        "Frequência de Neighborhoods"
    )

    plt.title(
        f"Spatial FDR - {titulo}\n"
        "Sensibilidade sem Harmony"
    )

    plt.axvline(
        x=0.1,
        color="red",
        linestyle="--",
    )

    plt.tight_layout()

    plt.savefig(
        f"{pasta_figuras}/"
        f"{prefixo}_SpatialFDR.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        resultados[
            "logFC"
        ].dropna(),
        bins=50,
    )

    plt.axvline(
        x=0,
        color="black",
        linestyle="--",
    )

    plt.xlabel(
        "logFC (Recessivo - Não Recessivo)"
    )

    plt.ylabel(
        "Frequência de Neighborhoods"
    )

    plt.title(
        f"Distribuição de logFC - {titulo}\n"
        "Sensibilidade sem Harmony"
    )

    plt.tight_layout()

    plt.savefig(
        f"{pasta_figuras}/"
        f"{prefixo}_Distribuicao_logFC.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    ordenados = (
        resultados[
            [
                "Neighborhood",
                "logFC",
            ]
        ]
        .dropna()
        .sort_values(
            "logFC"
        )
        .reset_index(
            drop=True
        )
    )


    plt.figure(
        figsize=(11, 6)
    )

    plt.scatter(
        np.arange(
            len(ordenados)
        ),
        ordenados[
            "logFC"
        ],
        s=12,
        alpha=0.7,
    )

    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
    )

    plt.xlabel(
        "Neighborhoods ordenadas por logFC"
    )

    plt.ylabel(
        "logFC (Recessivo - Não Recessivo)"
    )

    plt.title(
        f"logFC de todas as Neighborhoods - {titulo}\n"
        "Sensibilidade sem Harmony"
    )

    plt.tight_layout()

    plt.savefig(
        f"{pasta_figuras}/"
        f"{prefixo}_Todos_logFC_ordenados.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def gerar_inspecao_donor_level(
    mdata_resultado,
    resultados,
    prefixo,
    titulo,
    top_n_csv=20,
    top_n_plot=6,
):
    """
    Inspeção DESCRITIVA no nível do doador.

    A porcentagem é usada apenas para visualização.
    O modelo edgeR/Milo continua sendo o teste inferencial.
    """

    candidatos = (
        resultados
        .sort_values(
            "PValue"
        )
        .head(
            top_n_csv
        )
        .copy()
    )


    ids_candidatos = (
        candidatos[
            "Neighborhood"
        ]
        .astype(str)
        .tolist()
    )


    milo_adata = (
        mdata_resultado[
            "milo"
        ]
    )


    counts_df = pd.DataFrame(
        matriz_para_numpy(
            milo_adata.X
        ),
        index=milo_adata.obs_names.astype(
            str
        ),
        columns=milo_adata.var_names.astype(
            str
        ),
    )


    ids_validos = [
        nhood
        for nhood
        in ids_candidatos
        if nhood
        in counts_df.columns
    ]


    if not ids_validos:
        print(
            f"AVISO: nenhuma neighborhood candidata de {prefixo} "
            "foi encontrada na matriz donor x neighborhood."
        )

        return


    donor_long = (
        counts_df[
            ids_validos
        ]
        .rename_axis(
            "donor_id"
        )
        .reset_index()
        .melt(
            id_vars="donor_id",
            var_name="Neighborhood",
            value_name="N_cells_nhood",
        )
    )


    donor_long[
        "Total_cells_donor"
    ] = (
        donor_long[
            "donor_id"
        ]
        .map(
            total_celulas_por_doador
        )
    )


    donor_long[
        "Percent_cells_donor"
    ] = (
        100.0
        * donor_long[
            "N_cells_nhood"
        ]
        / donor_long[
            "Total_cells_donor"
        ]
    )


    donor_long = donor_long.merge(
        metadados_doador,
        on="donor_id",
        how="left",
        validate="many_to_one",
    )


    colunas_metricas = [
        "Neighborhood",
        "logFC",
        "PValue",
        "FDR",
        "SpatialFDR",
    ]


    for col in [
        "nhood_annotation",
        "nhood_annotation_frac",
        "Nhood_size",
    ]:
        if col in candidatos.columns:
            colunas_metricas.append(
                col
            )


    metricas = (
        candidatos[
            colunas_metricas
        ]
        .copy()
    )

    metricas[
        "Neighborhood"
    ] = (
        metricas[
            "Neighborhood"
        ]
        .astype(str)
    )


    donor_long = donor_long.merge(
        metricas,
        on="Neighborhood",
        how="left",
        validate="many_to_one",
    )


    donor_long.to_csv(
        f"{pasta_tabelas_donor}/"
        f"{prefixo}_Top20_donor_level.csv",
        index=False,
    )


    ids_plot = (
        ids_validos[
            :top_n_plot
        ]
    )


    n_cols = 3

    n_rows = int(
        np.ceil(
            len(
                ids_plot
            )
            / n_cols
        )
    )


    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            16,
            5 * n_rows,
        ),
    )


    axes = (
        np.atleast_1d(
            axes
        )
        .ravel()
    )

    rng = np.random.default_rng(
        42
    )


    for ax, nhood in zip(
        axes,
        ids_plot,
    ):
        df_nhood = (
            donor_long[
                donor_long[
                    "Neighborhood"
                ]
                == nhood
            ]
            .copy()
        )


        nao_rec = (
            df_nhood.loc[
                df_nhood[
                    "Modelo_Recessivo"
                ]
                == "Nao_Recessivo",
                "Percent_cells_donor",
            ]
            .to_numpy(
                dtype=float
            )
        )


        rec = (
            df_nhood.loc[
                df_nhood[
                    "Modelo_Recessivo"
                ]
                == "Recessivo",
                "Percent_cells_donor",
            ]
            .to_numpy(
                dtype=float
            )
        )


        ax.boxplot(
            [
                nao_rec,
                rec,
            ],
            positions=[
                0,
                1,
            ],
            widths=0.55,
            showfliers=False,
        )


        for pos, valores in enumerate(
            [
                nao_rec,
                rec,
            ]
        ):
            jitter = rng.normal(
                loc=pos,
                scale=0.04,
                size=len(
                    valores
                ),
            )

            ax.scatter(
                jitter,
                valores,
                s=26,
                alpha=0.7,
            )


        linha = (
            candidatos[
                candidatos[
                    "Neighborhood"
                ]
                .astype(str)
                == nhood
            ]
            .iloc[0]
        )


        anotacao = (
            str(
                linha[
                    "nhood_annotation"
                ]
            )
            if (
                "nhood_annotation"
                in candidatos.columns
            )
            else "NA"
        )


        ax.set_title(
            f"Nhood {nhood} | Leiden {anotacao}\n"
            f"logFC={linha['logFC']:.2f} | "
            f"P={linha['PValue']:.2e}",
            fontsize=10,
        )


        ax.set_xticks(
            [
                0,
                1,
            ]
        )

        ax.set_xticklabels(
            [
                "Não recessivo",
                "Recessivo",
            ],
            rotation=15,
        )

        ax.set_ylabel(
            "% das células do doador na neighborhood"
        )


    for ax in axes[
        len(
            ids_plot
        ):
    ]:
        fig.delaxes(
            ax
        )


    fig.suptitle(
        f"Inspeção donor-level - {titulo}\n"
        "Sensibilidade sem Harmony; cada ponto = 1 doador",
        fontsize=16,
    )


    plt.tight_layout()

    plt.savefig(
        f"{pasta_figuras_donor}/"
        f"{prefixo}_Top6_donor_level.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def gerar_grafos_milo(
    mdata_resultado,
    prefixo,
    titulo_grafo,
):
    """
    Salva:
    1) grafo inferencial com SpatialFDR <= 0.1;
    2) grafo exploratório mostrando logFC sem filtro de significância.
    """

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    milo.plot_nhood_graph(
        mdata_resultado,
        alpha=0.1,
        min_logFC=0,
        min_size=2,
        plot_edges=True,
        title=(
            f"{titulo_grafo}\n"
            "Sensibilidade sem Harmony"
        ),
        ax=ax,
    )

    plt.savefig(
        f"{pasta_figuras}/"
        f"{prefixo}_Grafo_Milo_SpatialFDR_0.1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    milo.plot_nhood_graph(
        mdata_resultado,
        alpha=1.0,
        min_logFC=0,
        min_size=2,
        plot_edges=True,
        title=(
            f"{titulo_grafo}\n"
            "SEM HARMONY - EXPLORATÓRIO: "
            "cores = logFC, sem filtro de significância"
        ),
        ax=ax,
    )

    plt.savefig(
        f"{pasta_figuras}/"
        f"{prefixo}_Grafo_logFC_exploratorio_sem_filtro.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def executar_rota(
    nome_rota,
    prefixo,
    design,
    titulo,
    titulo_grafo,
):
    print(
        "\n============================================"
    )

    print(
        nome_rota
    )

    print(
        "SENSIBILIDADE SEM HARMONY"
    )

    print(
        "============================================\n"
    )


    mdata_resultado = (
        mdata.copy()
    )


    milo.da_nhoods(
        mdata_resultado,
        design=design,
        model_contrasts=(
            "Modelo_RecessivoRecessivo "
            "- Modelo_RecessivoNao_Recessivo"
        ),
        solver="edger",
    )


    print(
        "Teste de abundância diferencial concluído: "
        f"{nome_rota}"
    )


    milo.build_nhood_graph(
        mdata_resultado
    )


    estatisticas = (
        mdata_resultado[
            "milo"
        ]
        .var[
            [
                "logFC",
                "PValue",
                "FDR",
                "SpatialFDR",
            ]
        ]
    )


    print(
        f"\n===== RESUMO {nome_rota} ====="
    )

    print(
        estatisticas.describe()
    )

    print(
        "================================\n"
    )


    n_significativas = (
        estatisticas[
            "SpatialFDR"
        ]
        .lt(0.1)
        .sum()
    )


    print(
        "Neighborhoods significativas "
        "(SpatialFDR < 0.1): "
        f"{n_significativas}"
    )


    resultados = exportar_resultados(
        mdata_resultado,
        prefixo,
    )


    gerar_graficos_estatisticos(
        resultados,
        prefixo,
        titulo,
    )


    gerar_inspecao_donor_level(
        mdata_resultado,
        resultados,
        prefixo,
        titulo,
    )


    gerar_grafos_milo(
        mdata_resultado,
        prefixo,
        titulo_grafo,
    )


    resumo_rota = {
        "Rota": nome_rota,
        "N_neighborhoods": len(
            resultados
        ),
        "Min_PValue": resultados[
            "PValue"
        ].min(),
        "Min_FDR": resultados[
            "FDR"
        ].min(),
        "Min_SpatialFDR": resultados[
            "SpatialFDR"
        ].min(),
        "N_SpatialFDR_menor_0_1": (
            resultados[
                "SpatialFDR"
            ]
            .lt(0.1)
            .sum()
        ),
        "LogFC_min": resultados[
            "logFC"
        ].min(),
        "LogFC_max": resultados[
            "logFC"
        ].max(),
    }


    return (
        mdata_resultado,
        resultados,
        resumo_rota,
    )


# =========================================================
# ETAPA 4: ROTA 0 — BIN1 + IDADE
# =========================================================

mdata_r0, resultados_r0, resumo_r0 = (
    executar_rota(
        nome_rota=(
            "ROTA 0: BIN1 + IDADE"
        ),
        prefixo=(
            "Rota0_BIN1_Idade_SemHarmony"
        ),
        design=(
            "~ Modelo_Recessivo "
            "+ Age_at_death"
        ),
        titulo=(
            "BIN1 + Idade"
        ),
        titulo_grafo=(
            "Impacto do BIN1 (Modelo Recessivo)\n"
            "Ajustado apenas por Idade"
        ),
    )
)


# =========================================================
# ETAPA 5: ROTA 1 — BIN1 + IDADE + ADNC
# =========================================================

mdata_r1, resultados_r1, resumo_r1 = (
    executar_rota(
        nome_rota=(
            "ROTA 1: BIN1 + IDADE + ADNC"
        ),
        prefixo=(
            "Rota1_BIN1_Idade_ADNC_SemHarmony"
        ),
        design=(
            "~ Modelo_Recessivo "
            "+ Age_at_death "
            "+ ADNC"
        ),
        titulo=(
            "BIN1 + Idade + ADNC"
        ),
        titulo_grafo=(
            "Impacto do BIN1 (Modelo Recessivo)\n"
            "Controlado por Idade e ADNC"
        ),
    )
)


# =========================================================
# ETAPA 6: ROTA 2 — BIN1 + IDADE + BRAAK + THAL
# =========================================================

mdata_r2, resultados_r2, resumo_r2 = (
    executar_rota(
        nome_rota=(
            "ROTA 2: BIN1 + IDADE + BRAAK + THAL"
        ),
        prefixo=(
            "Rota2_BIN1_Idade_Braak_Thal_SemHarmony"
        ),
        design=(
            "~ Modelo_Recessivo "
            "+ Age_at_death "
            "+ Braak_stage "
            "+ Thal_phase"
        ),
        titulo=(
            "BIN1 + Idade + Braak + Thal"
        ),
        titulo_grafo=(
            "Impacto do BIN1 (Modelo Recessivo)\n"
            "Controlado por Idade, Braak e Thal"
        ),
    )
)


# =========================================================
# ETAPA 7: RESUMO DAS TRÊS ROTAS SEM HARMONY
# =========================================================

comparacao_rotas_sem_harmony = pd.DataFrame(
    [
        resumo_r0,
        resumo_r1,
        resumo_r2,
    ]
)


comparacao_rotas_sem_harmony.to_csv(
    f"{pasta_tabelas}/"
    "Resumo_comparativo_das_rotas_sem_Harmony.csv",
    index=False,
)


print(
    "\n============================================"
)

print(
    "RESUMO DAS ROTAS - SEM HARMONY"
)

print(
    "============================================"
)

print(
    comparacao_rotas_sem_harmony.to_string(
        index=False
    )
)


# =========================================================
# ETAPA 8: TOP 20 DENTRO DA ANÁLISE SEM HARMONY
# =========================================================

resultados_por_rota = {
    "Rota0_BIN1_Idade_SemHarmony": resultados_r0,
    "Rota1_BIN1_Idade_ADNC_SemHarmony": resultados_r1,
    "Rota2_BIN1_Idade_Braak_Thal_SemHarmony": resultados_r2,
}

registros_top20 = []


for nome, df in resultados_por_rota.items():
    top = (
        df
        .sort_values(
            "PValue"
        )
        .head(20)
        .copy()
        .reset_index(
            drop=True
        )
    )


    top[
        "Rank_PValue"
    ] = np.arange(
        1,
        len(top) + 1,
    )

    top[
        "Rota"
    ] = nome


    colunas = [
        "Rota",
        "Rank_PValue",
        "Neighborhood",
        "logFC",
        "PValue",
        "FDR",
        "SpatialFDR",
    ]


    for col in [
        "nhood_annotation",
        "nhood_annotation_frac",
        "Nhood_size",
    ]:
        if col in top.columns:
            colunas.append(
                col
            )


    registros_top20.append(
        top[
            colunas
        ]
    )


candidatos_top20_sem_harmony = pd.concat(
    registros_top20,
    ignore_index=True,
)


ocorrencias_sem_harmony = (
    candidatos_top20_sem_harmony
    .groupby(
        "Neighborhood"
    )
    .agg(
        Ocorrencias_Top20=(
            "Rota",
            "nunique",
        ),
        Rotas=(
            "Rota",
            lambda x: "; ".join(
                sorted(
                    set(x)
                )
            ),
        ),
        Melhor_rank=(
            "Rank_PValue",
            "min",
        ),
        Menor_PValue=(
            "PValue",
            "min",
        ),
        Menor_SpatialFDR=(
            "SpatialFDR",
            "min",
        ),
        Maior_abs_logFC=(
            "logFC",
            lambda x: np.max(
                np.abs(x)
            ),
        ),
    )
    .sort_values(
        [
            "Ocorrencias_Top20",
            "Menor_PValue",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


ocorrencias_sem_harmony.to_csv(
    f"{pasta_tabelas}/"
    "Candidatos_recorrentes_Top20_sem_Harmony_resumo.csv"
)


candidatos_top20_sem_harmony = (
    candidatos_top20_sem_harmony
    .merge(
        ocorrencias_sem_harmony[
            [
                "Ocorrencias_Top20"
            ]
        ],
        left_on="Neighborhood",
        right_index=True,
        how="left",
    )
)


candidatos_top20_sem_harmony.to_csv(
    f"{pasta_tabelas}/"
    "Candidatos_recorrentes_Top20_sem_Harmony_detalhado.csv",
    index=False,
)


print(
    "\n===== CANDIDATOS RECORRENTES "
    "ENTRE AS ROTAS SEM HARMONY ====="
)

print(
    ocorrencias_sem_harmony
    .head(20)
    .to_string()
)

print(
    "=============================================\n"
)


# =========================================================
# ETAPA 9: COMPARAÇÃO COM A ANÁLISE PRINCIPAL (HARMONY)
# =========================================================

print(
    "\n============================================"
)

print(
    "COMPARAÇÃO COM A ANÁLISE PRINCIPAL (HARMONY)"
)

print(
    "============================================\n"
)


caminho_resumo_harmony = (
    "results/milo_analysis/"
    "Resumo_comparativo_das_rotas.csv"
)


if os.path.exists(
    caminho_resumo_harmony
):
    resumo_harmony = pd.read_csv(
        caminho_resumo_harmony
    )


    comparacao_harmony = resumo_harmony.merge(
        comparacao_rotas_sem_harmony,
        on="Rota",
        how="inner",
        suffixes=(
            "_Harmony",
            "_SemHarmony",
        ),
    )


    comparacao_harmony.to_csv(
        f"{pasta_comparacao}/"
        "Comparacao_Harmony_vs_SemHarmony_resumo.csv",
        index=False,
    )


    print(
        comparacao_harmony.to_string(
            index=False
        )
    )


    if not comparacao_harmony.empty:
        x = np.arange(
            len(
                comparacao_harmony
            )
        )

        largura = 0.36


        plt.figure(
            figsize=(10, 6)
        )


        plt.bar(
            x - largura / 2,
            comparacao_harmony[
                "Min_SpatialFDR_Harmony"
            ],
            width=largura,
            label="Com Harmony",
        )


        plt.bar(
            x + largura / 2,
            comparacao_harmony[
                "Min_SpatialFDR_SemHarmony"
            ],
            width=largura,
            label="Sem Harmony",
        )


        plt.axhline(
            y=0.1,
            color="black",
            linestyle="--",
        )


        plt.xticks(
            x,
            [
                "Rota 0",
                "Rota 1",
                "Rota 2",
            ][
                :len(
                    x
                )
            ],
        )


        plt.ylabel(
            "Menor SpatialFDR"
        )

        plt.title(
            "Sensibilidade ao Harmony: "
            "menor SpatialFDR por rota"
        )

        plt.legend()

        plt.tight_layout()


        plt.savefig(
            f"{pasta_figuras_comparacao}/"
            "Comparacao_Min_SpatialFDR_Harmony_vs_SemHarmony.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


else:
    print(
        "AVISO: o arquivo da análise principal não foi encontrado:"
    )

    print(
        caminho_resumo_harmony
    )

    print(
        "A análise sem Harmony foi concluída normalmente, "
        "mas a comparação automática global não será gerada."
    )


# =========================================================
# ETAPA 9.1: COMPARAÇÃO POR LEIDEN 0.45
# =========================================================

# IMPORTANTE:
# Os IDs das neighborhoods NÃO são comparáveis entre os dois
# grafos. A comparação é feita no nível da anotação de referência
# Leiden 0.45.

rotas_comparacao = [
    {
        "nome": "Rota0_BIN1_Idade",
        "sem_harmony": "Rota0_BIN1_Idade_SemHarmony",
    },
    {
        "nome": "Rota1_BIN1_Idade_ADNC",
        "sem_harmony": "Rota1_BIN1_Idade_ADNC_SemHarmony",
    },
    {
        "nome": "Rota2_BIN1_Idade_Braak_Thal",
        "sem_harmony": "Rota2_BIN1_Idade_Braak_Thal_SemHarmony",
    },
]


for rota in rotas_comparacao:
    caminho_cluster_harmony = (
        "results/milo_analysis/"
        f"{rota['nome']}_resumo_por_Leiden045.csv"
    )

    caminho_cluster_sem = (
        f"{pasta_tabelas}/"
        f"{rota['sem_harmony']}_resumo_por_Leiden045.csv"
    )


    if (
        os.path.exists(
            caminho_cluster_harmony
        )
        and os.path.exists(
            caminho_cluster_sem
        )
    ):
        cluster_harmony = pd.read_csv(
            caminho_cluster_harmony
        )

        cluster_sem = pd.read_csv(
            caminho_cluster_sem
        )


        comparacao_cluster = cluster_harmony.merge(
            cluster_sem,
            on="nhood_annotation",
            how="outer",
            suffixes=(
                "_Harmony",
                "_SemHarmony",
            ),
        )


        comparacao_cluster[
            "Direcao_mediana_consistente"
        ] = (
            np.sign(
                comparacao_cluster[
                    "logFC_mediano_Harmony"
                ]
            )
            == np.sign(
                comparacao_cluster[
                    "logFC_mediano_SemHarmony"
                ]
            )
        )


        comparacao_cluster.to_csv(
            f"{pasta_comparacao}/"
            f"{rota['nome']}_Comparacao_por_Leiden045.csv",
            index=False,
        )


# =========================================================
# ETAPA 9.2: COMPARAÇÃO DA COMPOSIÇÃO DOS TOP 20
# =========================================================

for rota in rotas_comparacao:
    caminho_top_harmony = (
        "results/milo_analysis/"
        f"{rota['nome']}_Top20_menor_PValue.csv"
    )

    caminho_top_sem = (
        f"{pasta_tabelas}/"
        f"{rota['sem_harmony']}_Top20_menor_PValue.csv"
    )


    if (
        os.path.exists(
            caminho_top_harmony
        )
        and os.path.exists(
            caminho_top_sem
        )
    ):
        top_harmony = pd.read_csv(
            caminho_top_harmony
        )

        top_sem = pd.read_csv(
            caminho_top_sem
        )


        if (
            "nhood_annotation"
            in top_harmony.columns
            and "nhood_annotation"
            in top_sem.columns
        ):
            cont_harmony = (
                top_harmony[
                    "nhood_annotation"
                ]
                .astype(str)
                .value_counts()
                .rename(
                    "N_Top20_Harmony"
                )
            )


            cont_sem = (
                top_sem[
                    "nhood_annotation"
                ]
                .astype(str)
                .value_counts()
                .rename(
                    "N_Top20_SemHarmony"
                )
            )


            comparacao_top = pd.concat(
                [
                    cont_harmony,
                    cont_sem,
                ],
                axis=1,
            ).fillna(
                0
            )


            comparacao_top[
                "N_Top20_Harmony"
            ] = (
                comparacao_top[
                    "N_Top20_Harmony"
                ]
                .astype(int)
            )


            comparacao_top[
                "N_Top20_SemHarmony"
            ] = (
                comparacao_top[
                    "N_Top20_SemHarmony"
                ]
                .astype(int)
            )


            comparacao_top.index.name = (
                "Leiden045_referencia"
            )


            comparacao_top.to_csv(
                f"{pasta_comparacao}/"
                f"{rota['nome']}_Comparacao_Top20_por_Leiden045.csv"
            )


# =========================================================
# FINALIZAÇÃO
# =========================================================

print(
    "\n============================================"
)

print(
    "ANÁLISE DE SENSIBILIDADE SEM HARMONY CONCLUÍDA"
)

print(
    "============================================"
)

print(
    f"Figuras principais: "
    f"{pasta_figuras}"
)

print(
    f"Figuras donor-level: "
    f"{pasta_figuras_donor}"
)

print(
    f"Tabelas principais: "
    f"{pasta_tabelas}"
)

print(
    f"Tabelas donor-level: "
    f"{pasta_tabelas_donor}"
)

print(
    f"Comparações com Harmony: "
    f"{pasta_comparacao}"
)

print(
    "\nIMPORTANTE: não comparar IDs de neighborhoods "
    "diretamente entre o 02_milo.py e este script."
)

print(
    "Os grafos são diferentes; a comparação deve ser feita "
    "pelas estatísticas das rotas, direção dos efeitos, "
    "Leiden 0.45 de referência e padrões donor-level."
)