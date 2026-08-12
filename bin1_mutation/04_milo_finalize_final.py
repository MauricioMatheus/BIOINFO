# ============================================================
# 04_milo_finalize_final.py
#
# MILO FINAL — ETAPA 3
# SpatialFDR + anotação + comparação das 3 rotas
#
# EXECUTAR NO AMBIENTE:
#   conda activate milo_bin1
#
# Este script:
#   - reconstrói deterministicamente as MESMAS neighborhoods
#     sobre o KNN congelado;
#   - verifica o SHA-256;
#   - importa os resultados edgeR 4.10.x;
#   - calcula SpatialFDR reproduzindo Pertpy 0.10.0;
#   - agrega anotação final/Leiden/QC;
#   - gera tabelas, resumos, recorrência e gráficos;
#   - não executa edgeR.
# ============================================================

import os

for _var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[_var] = "1"

os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHONHASHSEED"] = "0"

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pertpy as pt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = (
    SCRIPT_DIR
    / "adata_limpo_anotado_sem_harmony_res0.5_FINAL.h5ad"
)

PASTA_BASE = (
    SCRIPT_DIR
    / "results"
    / "milo_final_edger410"
)

PASTA_EXPORT = PASTA_BASE / "export"
PASTA_EDGER = PASTA_BASE / "edger"
PASTA_FINAL = PASTA_BASE / "final"
PASTA_AUDITORIA = PASTA_BASE / "auditoria"

PASTA_FIGURAS = (
    SCRIPT_DIR
    / "figures"
    / "milo_final_edger410"
)

PASTA_DONOR = (
    PASTA_FINAL
    / "donor_level"
)

PASTA_FIGURAS_DONOR = (
    PASTA_FIGURAS
    / "donor_level"
)

for _pasta in [
    PASTA_FINAL,
    PASTA_AUDITORIA,
    PASTA_FIGURAS,
    PASTA_DONOR,
    PASTA_FIGURAS_DONOR,
]:
    _pasta.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIGURAÇÃO
# ============================================================

NHOOD_PROP = 0.10
NHOOD_SEED = 0
SPATIAL_FDR_LIMIAR = 0.10
PURITY_LIMIAR = 0.60

HASH_ESPERADO = (
    "c06ce07f1ded7c9d96e932901a84e864"
    "f1487ec62ca952ee9152ec341aa08033"
)

N_CELULAS_ESPERADO = 32657
N_DOADORES_ESPERADO = 75
N_NHOODS_ESPERADO = 2201

COL_LEIDEN = "leiden_0.5"
COL_BIO = "Estado_Microglial_Biologico"
COL_CONF = "Confianca_Anotacao"
COL_QC = "Flag_QC_Anotacao"

ROTAS = {
    "Rota0_BIN1_Idade": (
        "ROTA 0: BIN1 + IDADE"
    ),
    "Rota1_BIN1_Idade_ADNC": (
        "ROTA 1: BIN1 + IDADE + ADNC"
    ),
    "Rota2_BIN1_Idade_Braak_Thal": (
        "ROTA 2: BIN1 + IDADE + BRAAK + THAL"
    ),
}


# ============================================================
# FUNÇÕES
# ============================================================

def salvar_json(
    caminho,
    dados,
):
    def conv(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(
            f"Tipo não serializável: {type(obj)}"
        )

    Path(caminho).write_text(
        json.dumps(
            dados,
            ensure_ascii=False,
            indent=2,
            default=conv,
        ),
        encoding="utf-8",
    )


def nivel_seguro_r(valor):
    valor = str(valor).strip()
    valor = re.sub(
        r"\s+",
        "_",
        valor,
    )
    valor = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        valor,
    )
    valor = re.sub(
        r"_+",
        "_",
        valor,
    ).strip("_")
    return valor


def assinatura_neighborhoods(
    rna,
    milo_adata,
):
    nhoods = sp.csr_matrix(
        rna.obsm["nhoods"]
    )

    nhoods.sort_indices()
    nhoods.eliminate_zeros()

    h = hashlib.sha256()

    h.update(
        np.asarray(
            nhoods.shape,
            dtype=np.int64,
        ).tobytes()
    )

    h.update(
        np.asarray(
            nhoods.indptr,
            dtype=np.int64,
        ).tobytes()
    )

    h.update(
        np.asarray(
            nhoods.indices,
            dtype=np.int64,
        ).tobytes()
    )

    h.update(
        np.asarray(
            nhoods.data,
            dtype=np.int8,
        ).tobytes()
    )

    if "index_cell" in milo_adata.var.columns:
        for nome in (
            milo_adata.var["index_cell"]
            .astype(str)
        ):
            h.update(
                nome.encode("utf-8")
            )
            h.update(b"\0")

    return h.hexdigest()


def spatial_fdr_pertpy_0100(
    pvalues,
    kth_distance,
):
    """
    Reprodução da função Milo._graph_spatial_fdr do Pertpy 0.10.0:

        w = 1 / kth_distance
        inf -> 0
        ordenar por PValue
        (sum(w) * p / cumsum(w))
        cumulative minimum de trás para frente
        recolocar na ordem original
        limitar em 1
    """

    p = pd.Series(
        pd.to_numeric(
            pvalues,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
    )

    d = pd.Series(
        pd.to_numeric(
            kth_distance,
            errors="coerce",
        ).to_numpy(
            dtype=float
        )
    )

    w = 1.0 / d
    w[
        np.isinf(
            w
        )
    ] = 0.0

    keep = (
        ~p.isna()
        & ~w.isna()
    )

    p_keep = p.loc[
        keep
    ]

    w_keep = w.loc[
        keep
    ]

    order = np.argsort(
        p_keep.to_numpy()
    )

    p_sorted = (
        p_keep
        .iloc[order]
        .to_numpy(
            dtype=float
        )
    )

    w_sorted = (
        w_keep
        .iloc[order]
        .to_numpy(
            dtype=float
        )
    )

    if (
        len(w_sorted) == 0
        or w_sorted.sum() <= 0
    ):
        raise RuntimeError(
            "Pesos inválidos para SpatialFDR."
        )

    cumul_w = np.cumsum(
        w_sorted
    )

    if np.any(
        cumul_w <= 0
    ):
        raise RuntimeError(
            "cumsum dos pesos contém zero antes do fim."
        )

    bruto = (
        w_sorted.sum()
        * p_sorted
        / cumul_w
    )

    monotono = np.minimum.accumulate(
        bruto[
            ::-1
        ]
    )[
        ::-1
    ]

    monotono = np.minimum(
        monotono,
        1.0,
    )

    ajustado_keep = np.empty(
        len(order),
        dtype=float,
    )

    ajustado_keep[
        order
    ] = monotono

    saida = np.full(
        len(p),
        np.nan,
        dtype=float,
    )

    idx_keep = np.flatnonzero(
        keep.to_numpy()
    )

    saida[
        idx_keep
    ] = ajustado_keep

    return saida


def preparar_metadados(
    adata,
):
    renomear = {}

    if (
        "Age_at_death" not in adata.obs.columns
        and "Age.at.death" in adata.obs.columns
    ):
        renomear[
            "Age.at.death"
        ] = "Age_at_death"

    if (
        "Braak_stage" not in adata.obs.columns
        and "Braak.stage" in adata.obs.columns
    ):
        renomear[
            "Braak.stage"
        ] = "Braak_stage"

    if (
        "Thal_phase" not in adata.obs.columns
        and "Thal.phase" in adata.obs.columns
    ):
        renomear[
            "Thal.phase"
        ] = "Thal_phase"

    if renomear:
        adata.obs.rename(
            columns=renomear,
            inplace=True,
        )

    mapa_bin1 = {
        "0/0": "Nao_Recessivo",
        "0/1": "Nao_Recessivo",
        "1/1": "Recessivo",
    }

    adata.obs["Modelo_Recessivo"] = (
        adata.obs["Mutacao_BIN1"]
        .astype(str)
        .map(mapa_bin1)
    )

    mapa_idade = {
        "65 to 77 years old": "65_77",
        "78 to 89 years old": "78_89",
        "90+ years old": "90_plus",
        "65_77": "65_77",
        "78_89": "78_89",
        "90_plus": "90_plus",
    }

    adata.obs["Age_at_death"] = (
        adata.obs["Age_at_death"]
        .astype(str)
        .map(mapa_idade)
    )

    for col in [
        "Braak_stage",
        "Thal_phase",
    ]:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .map(nivel_seguro_r)
        )

    adata.obs["Modelo_Recessivo"] = pd.Categorical(
        adata.obs["Modelo_Recessivo"],
        categories=[
            "Nao_Recessivo",
            "Recessivo",
        ],
        ordered=True,
    )

    adata.obs["Age_at_death"] = pd.Categorical(
        adata.obs["Age_at_death"],
        categories=[
            "65_77",
            "78_89",
            "90_plus",
        ],
        ordered=True,
    )

    adata.obs["ADNC"] = pd.Categorical(
        adata.obs["ADNC"].astype(str),
        categories=[
            "Low",
            "Intermediate",
            "High",
        ],
        ordered=True,
    )

    for col in [
        "Braak_stage",
        "Thal_phase",
    ]:
        cats = sorted(
            adata.obs[col]
            .astype(str)
            .unique()
        )

        adata.obs[col] = pd.Categorical(
            adata.obs[col].astype(str),
            categories=cats,
            ordered=True,
        )

    for col in [
        COL_LEIDEN,
        COL_BIO,
        COL_CONF,
        COL_QC,
    ]:
        adata.obs[col] = (
            adata.obs[col]
            .astype(str)
            .astype("category")
        )

    return adata


def mapa_unico(
    adata,
    coluna,
):
    tabela = (
        adata.obs[
            [
                COL_LEIDEN,
                coluna,
            ]
        ]
        .astype(str)
        .drop_duplicates()
    )

    n = (
        tabela
        .groupby(
            COL_LEIDEN
        )[
            coluna
        ]
        .nunique()
    )

    if (n != 1).any():
        raise RuntimeError(
            f"{coluna} não é unívoca por Leiden."
        )

    return dict(
        zip(
            tabela[COL_LEIDEN].astype(str),
            tabela[coluna].astype(str),
        )
    )


def gerar_graficos_basicos(
    df,
    prefixo,
    titulo,
):
    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.hist(
        df["SpatialFDR"].dropna(),
        bins=50,
    )

    plt.axvline(
        SPATIAL_FDR_LIMIAR,
        linestyle="--",
    )

    plt.xlabel(
        "Spatial FDR"
    )

    plt.ylabel(
        "Frequência de neighborhoods"
    )

    plt.title(
        f"Spatial FDR — {titulo}"
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_FIGURAS
        / f"{prefixo}_SpatialFDR.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.hist(
        df["logFC"].dropna(),
        bins=50,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "logFC (Recessivo - Não Recessivo)"
    )

    plt.ylabel(
        "Frequência de neighborhoods"
    )

    plt.title(
        f"Distribuição de logFC — {titulo}"
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_FIGURAS
        / f"{prefixo}_Distribuicao_logFC.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def gerar_donor_level(
    counts_donor_x_nhood,
    donor_meta,
    resultados,
    prefixo,
    titulo,
):
    top = (
        resultados
        .sort_values(
            "PValue"
        )
        .head(20)
        .copy()
    )

    ids = (
        top["Neighborhood"]
        .astype(str)
        .tolist()
    )

    ids = [
        x
        for x in ids
        if x in counts_donor_x_nhood.columns
    ]

    if not ids:
        return

    total_cells = (
        counts_donor_x_nhood
        .max(
            axis=1
        )
    )

    # Para a visualização percentual, o denominador correto é o total
    # de células do donor no objeto, calculado separadamente no chamador.
    # Esta função recebe essa coluna já presente em donor_meta.
    donor_long = (
        counts_donor_x_nhood[
            ids
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

    donor_long = (
        donor_long
        .merge(
            donor_meta,
            on="donor_id",
            how="left",
            validate="many_to_one",
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

    metricas_cols = [
        "Neighborhood",
        "logFC",
        "PValue",
        "FDR",
        "SpatialFDR",
        "nhood_Leiden_0_5",
        "nhood_annotation",
        "nhood_QC_cluster",
    ]

    metricas = (
        top[
            [
                c
                for c in metricas_cols
                if c in top.columns
            ]
        ]
        .copy()
    )

    metricas["Neighborhood"] = (
        metricas["Neighborhood"]
        .astype(str)
    )

    donor_long = (
        donor_long
        .merge(
            metricas,
            on="Neighborhood",
            how="left",
            validate="many_to_one",
        )
    )

    donor_long.to_csv(
        PASTA_DONOR
        / f"{prefixo}_Top20_donor_level.csv",
        index=False,
    )

    ids_plot = ids[:6]

    n_cols = 3
    n_rows = int(
        np.ceil(
            len(ids_plot)
            / n_cols
        )
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            17,
            5.5 * n_rows,
        ),
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    rng = np.random.default_rng(
        42
    )

    for ax, nhood in zip(
        axes,
        ids_plot,
    ):
        d = donor_long[
            donor_long["Neighborhood"]
            == nhood
        ]

        nao = (
            d.loc[
                d["Modelo_Recessivo"]
                == "Nao_Recessivo",
                "Percent_cells_donor",
            ]
            .to_numpy(
                dtype=float
            )
        )

        rec = (
            d.loc[
                d["Modelo_Recessivo"]
                == "Recessivo",
                "Percent_cells_donor",
            ]
            .to_numpy(
                dtype=float
            )
        )

        ax.boxplot(
            [
                nao,
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
                nao,
                rec,
            ]
        ):
            jitter = rng.normal(
                loc=pos,
                scale=0.04,
                size=len(valores),
            )

            ax.scatter(
                jitter,
                valores,
                s=26,
                alpha=0.7,
            )

        linha = (
            top[
                top["Neighborhood"]
                .astype(str)
                == nhood
            ]
            .iloc[0]
        )

        ax.set_title(
            (
                f"Nhood {nhood}\n"
                f"logFC={linha['logFC']:.2f} | "
                f"P={linha['PValue']:.2e} | "
                f"SFDR={linha['SpatialFDR']:.2e}"
            ),
            fontsize=9,
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
            rotation=12,
        )

        ax.set_ylabel(
            "% das células do donor"
        )

    for ax in axes[
        len(ids_plot):
    ]:
        fig.delaxes(
            ax
        )

    fig.suptitle(
        (
            f"Donor-level — {titulo}\n"
            "Visualização descritiva"
        ),
        fontsize=15,
    )

    plt.tight_layout()

    plt.savefig(
        PASTA_FIGURAS_DONOR
        / f"{prefixo}_Top6_donor_level.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 1. RECONSTRUÇÃO DETERMINÍSTICA
# ============================================================

print("=" * 78)
print("MILO FINAL — ETAPA 3 — SpatialFDR E CONSOLIDAÇÃO")
print("=" * 78)

if pt.__version__ != "0.10.0":
    raise RuntimeError(
        "Pertpy 0.10.0 é obrigatório nesta etapa."
    )

adata = sc.read_h5ad(
    ARQUIVO_ENTRADA
)

if adata.n_obs != N_CELULAS_ESPERADO:
    raise RuntimeError(
        "Número de células inesperado."
    )

adata = preparar_metadados(
    adata
)

if "nhoods" in adata.obsm:
    del adata.obsm["nhoods"]

for col in [
    "nhood_ixs_random",
    "nhood_ixs_refined",
    "nhood_kth_distance",
]:
    if col in adata.obs.columns:
        del adata.obs[col]

if "nhood_neighbors_key" in adata.uns:
    del adata.uns[
        "nhood_neighbors_key"
    ]

milo = pt.tl.Milo()
mdata = milo.load(
    adata
)

milo.make_nhoods(
    mdata["rna"],
    prop=NHOOD_PROP,
    seed=NHOOD_SEED,
)

mdata = milo.count_nhoods(
    mdata,
    sample_col="donor_id",
)

m = mdata["milo"]
rna = mdata["rna"]

if m.shape != (
    N_DOADORES_ESPERADO,
    N_NHOODS_ESPERADO,
):
    raise RuntimeError(
        f"Matriz Milo inesperada: {m.shape}"
    )

assinatura = assinatura_neighborhoods(
    rna,
    m,
)

print(
    f"Neighborhood SHA-256: {assinatura}"
)

if assinatura != HASH_ESPERADO:
    raise RuntimeError(
        "Neighborhoods divergiram do conjunto congelado."
    )

print(
    "[OK] Neighborhoods idênticas."
)


# ============================================================
# 2. ANOTAÇÃO
# ============================================================

milo.annotate_nhoods(
    mdata,
    anno_col=COL_BIO,
)

if "annotation_labels" in m.uns:
    m.uns["annotation_labels"] = np.asarray(
        [
            str(x)
            for x in m.uns["annotation_labels"]
        ],
        dtype=str,
    )

mapa_bio = mapa_unico(
    adata,
    COL_BIO,
)

mapa_conf = mapa_unico(
    adata,
    COL_CONF,
)

mapa_qc = mapa_unico(
    adata,
    COL_QC,
)

leiden = (
    rna.obs[COL_LEIDEN]
    .astype(str)
)

labels_leiden = sorted(
    leiden.unique(),
    key=lambda x: (
        int(x)
        if str(x).isdigit()
        else str(x)
    ),
)

dummies = (
    pd.get_dummies(
        leiden
    )
    .reindex(
        columns=labels_leiden,
        fill_value=0,
    )
)

contagens_leiden = (
    rna.obsm["nhoods"]
    .T
    .dot(
        sp.csr_matrix(
            dummies.to_numpy(
                dtype=np.int64
            )
        )
    )
    .toarray()
)

frac_leiden = (
    contagens_leiden
    / contagens_leiden.sum(
        axis=1
    )[
        :,
        np.newaxis,
    ]
)

idx = np.argmax(
    frac_leiden,
    axis=1,
)

leiden_dom = np.asarray(
    labels_leiden,
    dtype=object,
)[idx]

leiden_frac = frac_leiden[
    np.arange(
        len(idx)
    ),
    idx,
]

m.var["nhood_Leiden_0_5"] = pd.Categorical(
    leiden_dom,
    categories=labels_leiden,
)

m.var["nhood_Leiden_0_5_frac"] = (
    leiden_frac
)

serie_dom = pd.Series(
    leiden_dom,
    index=m.var_names,
    dtype=str,
)

m.var["nhood_bio_por_Leiden_dominante"] = (
    serie_dom
    .map(mapa_bio)
    .to_numpy()
)

m.var["nhood_confianca_cluster"] = (
    serie_dom
    .map(mapa_conf)
    .to_numpy()
)

m.var["nhood_QC_cluster"] = (
    serie_dom
    .map(mapa_qc)
    .to_numpy()
)

m.var["nhood_mista"] = (
    (
        m.var["nhood_annotation_frac"]
        < PURITY_LIMIAR
    )
    |
    (
        m.var["nhood_Leiden_0_5_frac"]
        < PURITY_LIMIAR
    )
)

m.var["Nhood_size"] = np.asarray(
    rna.obsm["nhoods"]
    .sum(axis=0)
).ravel().astype(int)


# ============================================================
# 3. DONOR-LEVEL BASE
# ============================================================

doadores = (
    adata.obs[
        [
            "donor_id",
            "Modelo_Recessivo",
            "Age_at_death",
            "ADNC",
            "Braak_stage",
            "Thal_phase",
        ]
    ]
    .drop_duplicates(
        subset="donor_id"
    )
    .copy()
)

for col in doadores.columns:
    doadores[col] = (
        doadores[col]
        .astype(str)
    )

total_cells = (
    adata.obs["donor_id"]
    .astype(str)
    .value_counts()
)

doadores["Total_cells_donor"] = (
    doadores["donor_id"]
    .map(total_cells)
)

X_counts = m.X

if sp.issparse(X_counts):
    X_counts = X_counts.toarray()

counts_donor_x_nhood = pd.DataFrame(
    np.asarray(
        X_counts,
        dtype=np.int64,
    ),
    index=m.obs_names.astype(str),
    columns=m.var_names.astype(str),
)


# ============================================================
# 4. RESULTADOS edgeR + SpatialFDR
# ============================================================

resultados_rotas = {}
resumos = []

for prefixo, titulo in ROTAS.items():
    print(
        f"\n[ROTA] {titulo}"
    )

    caminho = (
        PASTA_EDGER
        / f"{prefixo}_edgeR_raw.csv"
    )

    if not caminho.exists():
        raise FileNotFoundError(
            f"Resultado edgeR ausente:\n{caminho}"
        )

    edge = pd.read_csv(
        caminho
    )

    edge["Neighborhood"] = (
        edge["Neighborhood"]
        .astype(str)
    )

    if len(edge) != N_NHOODS_ESPERADO:
        raise RuntimeError(
            f"{prefixo}: esperado 2201 resultados."
        )

    if edge["Neighborhood"].duplicated().any():
        raise RuntimeError(
            f"{prefixo}: Neighborhood duplicada."
        )

    edge = (
        edge
        .set_index(
            "Neighborhood"
        )
        .loc[
            m.var_names.astype(str)
        ]
        .copy()
    )

    edge_version = (
        edge["EdgeR_version"]
        .astype(str)
        .unique()
        .tolist()
    )

    if (
        len(edge_version) != 1
        or not edge_version[0].startswith(
            "4.10."
        )
    ):
        raise RuntimeError(
            f"{prefixo}: edgeR não é 4.10.x: "
            f"{edge_version}"
        )

    if not (
        edge["Robust"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
        .all()
    ):
        raise RuntimeError(
            f"{prefixo}: Robust != TRUE."
        )

    if not (
        edge["Legacy"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "false",
                "0",
            ]
        )
        .all()
    ):
        raise RuntimeError(
            f"{prefixo}: Legacy != FALSE."
        )

    edge["SpatialFDR"] = (
        spatial_fdr_pertpy_0100(
            edge["PValue"],
            m.var["kth_distance"],
        )
    )

    meta = m.var[
        [
            "index_cell",
            "kth_distance",
            "Nhood_size",
            "nhood_annotation",
            "nhood_annotation_frac",
            "nhood_Leiden_0_5",
            "nhood_Leiden_0_5_frac",
            "nhood_bio_por_Leiden_dominante",
            "nhood_confianca_cluster",
            "nhood_QC_cluster",
            "nhood_mista",
        ]
    ].copy()

    meta.index = (
        meta.index.astype(str)
    )

    final = (
        edge
        .join(
            meta,
            how="left",
            validate="one_to_one",
        )
    )

    final.insert(
        0,
        "Neighborhood",
        final.index.astype(str),
    )

    final["abs_logFC"] = (
        final["logFC"]
        .abs()
    )

    final[
        "Significativa_SpatialFDR_0_1"
    ] = (
        final["SpatialFDR"]
        < SPATIAL_FDR_LIMIAR
    )

    final.to_csv(
        PASTA_FINAL
        / f"{prefixo}_todos_neighborhoods.csv",
        index=False,
    )

    (
        final
        .sort_values(
            "PValue"
        )
        .head(20)
        .to_csv(
            PASTA_FINAL
            / f"{prefixo}_Top20_menor_PValue.csv",
            index=False,
        )
    )

    (
        final[
            final["SpatialFDR"]
            < SPATIAL_FDR_LIMIAR
        ]
        .sort_values(
            [
                "SpatialFDR",
                "PValue",
            ]
        )
        .to_csv(
            PASTA_FINAL
            / f"{prefixo}_significativas_SpatialFDR_0_1.csv",
            index=False,
        )
    )

    resumo_cluster = (
        final
        .groupby(
            "nhood_Leiden_0_5",
            observed=True,
        )
        .agg(
            n_neighborhoods=(
                "Neighborhood",
                "size",
            ),
            n_significativas=(
                "Significativa_SpatialFDR_0_1",
                "sum",
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
        .reset_index()
    )

    resumo_cluster.to_csv(
        PASTA_FINAL
        / f"{prefixo}_resumo_por_Leiden0_5.csv",
        index=False,
    )

    resumo_bio = (
        final
        .groupby(
            "nhood_annotation",
            observed=True,
        )
        .agg(
            n_neighborhoods=(
                "Neighborhood",
                "size",
            ),
            n_significativas=(
                "Significativa_SpatialFDR_0_1",
                "sum",
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
        .reset_index()
    )

    resumo_bio.to_csv(
        PASTA_FINAL
        / f"{prefixo}_resumo_por_estado_biologico.csv",
        index=False,
    )

    n_sig = int(
        final[
            "Significativa_SpatialFDR_0_1"
        ].sum()
    )

    resumo = {
        "Rota": prefixo,
        "Nome": titulo,
        "N_neighborhoods": len(final),
        "Min_PValue": float(
            final["PValue"].min()
        ),
        "Min_FDR": float(
            final["FDR"].min()
        ),
        "Min_SpatialFDR": float(
            final["SpatialFDR"].min()
        ),
        "N_SpatialFDR_menor_0_1": n_sig,
        "LogFC_min": float(
            final["logFC"].min()
        ),
        "LogFC_max": float(
            final["logFC"].max()
        ),
        "EdgeR": edge_version[0],
        "Robust": True,
        "Legacy": False,
    }

    resumos.append(
        resumo
    )

    resultados_rotas[
        prefixo
    ] = final.copy()

    gerar_graficos_basicos(
        final,
        prefixo,
        titulo,
    )

    gerar_donor_level(
        counts_donor_x_nhood,
        doadores,
        final,
        prefixo,
        titulo,
    )

    # Grafo Milo: opcional, sem impacto estatístico.
    try:
        for col in [
            "logFC",
            "PValue",
            "FDR",
            "SpatialFDR",
        ]:
            m.var[col] = (
                final
                .set_index(
                    "Neighborhood"
                )
                .loc[
                    m.var_names.astype(str),
                    col,
                ]
                .to_numpy()
            )

        milo.build_nhood_graph(
            mdata,
            basis="X_umap",
        )

        fig, ax = plt.subplots(
            figsize=(
                10,
                10,
            )
        )

        milo.plot_nhood_graph(
            mdata,
            alpha=SPATIAL_FDR_LIMIAR,
            min_logFC=0,
            min_size=2,
            plot_edges=True,
            title=(
                f"{titulo}\n"
                "SpatialFDR < 0.10"
            ),
            ax=ax,
        )

        plt.savefig(
            PASTA_FIGURAS
            / f"{prefixo}_Grafo_Milo_SpatialFDR_0.1.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            "all"
        )

    except Exception as exc:
        print(
            f"AVISO gráfico Milo {prefixo}: {exc}"
        )
        plt.close(
            "all"
        )

    print(
        f"[OK] {prefixo}: "
        f"{n_sig} neighborhoods com SpatialFDR < 0.10"
    )


# ============================================================
# 5. COMPARAÇÃO DAS ROTAS
# ============================================================

resumo_df = pd.DataFrame(
    resumos
)

resumo_df.to_csv(
    PASTA_FINAL
    / "Resumo_comparativo_das_rotas.csv",
    index=False,
)

print(
    "\nResumo comparativo:"
)

print(
    resumo_df.to_string(
        index=False
    )
)


# ============================================================
# 6. TOP20 RECORRENTES
# ============================================================

top_partes = []

for prefixo, df in resultados_rotas.items():
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

    top["Rota"] = prefixo

    top["Rank_PValue"] = np.arange(
        1,
        len(top) + 1,
    )

    top_partes.append(
        top
    )

top20 = pd.concat(
    top_partes,
    ignore_index=True,
)

rec_top = (
    top20
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
            lambda x: float(
                np.max(
                    np.abs(
                        np.asarray(
                            x,
                            dtype=float,
                        )
                    )
                )
            ),
        ),
        Leiden_0_5=(
            "nhood_Leiden_0_5",
            "first",
        ),
        Estado_biologico=(
            "nhood_annotation",
            "first",
        ),
        Flag_QC=(
            "nhood_QC_cluster",
            "first",
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

rec_top.to_csv(
    PASTA_FINAL
    / "Candidatos_recorrentes_Top20_resumo.csv"
)

top20.to_csv(
    PASTA_FINAL
    / "Candidatos_recorrentes_Top20_detalhado.csv",
    index=False,
)


# ============================================================
# 7. SIGNIFICATIVAS RECORRENTES
# ============================================================

sig_partes = []

for prefixo, df in resultados_rotas.items():
    sig = (
        df[
            df["SpatialFDR"]
            < SPATIAL_FDR_LIMIAR
        ]
        .copy()
    )

    if sig.empty:
        continue

    sig["Rota"] = prefixo
    sig_partes.append(
        sig
    )

if sig_partes:
    sig_all = pd.concat(
        sig_partes,
        ignore_index=True,
    )

    sig_all.to_csv(
        PASTA_FINAL
        / "Neighborhoods_significativas_todas_rotas_detalhado.csv",
        index=False,
    )

    sig_rec = (
        sig_all
        .groupby(
            "Neighborhood"
        )
        .agg(
            N_rotas_significativa=(
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
            Menor_SpatialFDR=(
                "SpatialFDR",
                "min",
            ),
            Menor_PValue=(
                "PValue",
                "min",
            ),
            Maior_abs_logFC=(
                "logFC",
                lambda x: float(
                    np.max(
                        np.abs(
                            np.asarray(
                                x,
                                dtype=float,
                            )
                        )
                    )
                ),
            ),
            Leiden_0_5=(
                "nhood_Leiden_0_5",
                "first",
            ),
            Estado_biologico=(
                "nhood_annotation",
                "first",
            ),
            Flag_QC=(
                "nhood_QC_cluster",
                "first",
            ),
        )
        .sort_values(
            [
                "N_rotas_significativa",
                "Menor_SpatialFDR",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    sig_rec.to_csv(
        PASTA_FINAL
        / "Neighborhoods_significativas_recorrentes_resumo.csv"
    )

else:
    pd.DataFrame().to_csv(
        PASTA_FINAL
        / "Neighborhoods_significativas_recorrentes_resumo.csv",
        index=False,
    )


# ============================================================
# 8. STATUS FINAL
# ============================================================

status = {
    "Concluido": True,
    "N_rotas": 3,
    "N_neighborhoods": N_NHOODS_ESPERADO,
    "SHA256_neighborhoods": assinatura,
    "SpatialFDR_limiar": SPATIAL_FDR_LIMIAR,
    "SpatialFDR_implementacao": (
        "Reprodução de Pertpy 0.10.0 Milo._graph_spatial_fdr"
    ),
    "EdgeR": sorted(
        set(
            resumo_df["EdgeR"]
            .astype(str)
        )
    ),
    "Robust": True,
    "Legacy": False,
    "Harmony": False,
    "PCA_recalculada": False,
    "KNN_recalculado": False,
    "UMAP_recalculado": False,
    "Anotacao_biologica_refeita": False,
}

salvar_json(
    PASTA_AUDITORIA
    / "Execucao_final_status.json",
    status,
)

print("\n" + "=" * 78)
print("MILO FINAL edgeR 4.10.x CONCLUÍDO")
print("=" * 78)
print(
    f"Resultados finais: {PASTA_FINAL}"
)
print(
    f"Figuras: {PASTA_FIGURAS}"
)