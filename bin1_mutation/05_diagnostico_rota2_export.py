# ============================================================
# 05_diagnostico_rota2_export.py
#
# Exporta a matriz donor x neighborhood e os metadados da Rota 2
# SEM executar edgeR.
#
# Objetivo:
#   permitir que o edgeR seja testado diretamente via Rscript,
#   fora do rpy2/Python, para localizar o SIGBUS.
#
# IMPORTANTE:
#   - usa o mesmo objeto final sem Harmony;
#   - usa o mesmo KNN já existente;
#   - usa prop=0.10 e seed=0;
#   - exige o mesmo SHA-256 das neighborhoods já observado
#     nas Rotas 0/1/2.
# ============================================================

import os

for var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ[var] = "1"

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pertpy as pt


SCRIPT_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = (
    SCRIPT_DIR
    / "adata_limpo_anotado_sem_harmony_res0.5_FINAL.h5ad"
)

PASTA_SAIDA = (
    SCRIPT_DIR
    / "results"
    / "diagnostico_rota2_edger"
)

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True,
)

NHOOD_PROP = 0.10
NHOOD_SEED = 0

HASH_ESPERADO = (
    "c06ce07f1ded7c9d96e932901a84e864"
    "f1487ec62ca952ee9152ec341aa08033"
)


def nivel_seguro_r(valor):
    valor = str(valor).strip()
    valor = re.sub(r"\s+", "_", valor)
    valor = re.sub(r"[^A-Za-z0-9_]+", "_", valor)
    valor = re.sub(r"_+", "_", valor).strip("_")
    return valor


def assinatura_neighborhoods(rna, milo_adata):
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


print("=" * 72)
print("EXPORTAÇÃO DIAGNÓSTICA DA ROTA 2 — SEM edgeR")
print("=" * 72)

if pt.__version__ != "0.10.0":
    raise RuntimeError(
        f"Pertpy esperado: 0.10.0; encontrado: {pt.__version__}"
    )

if not ARQUIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_ENTRADA}"
    )

adata = sc.read_h5ad(
    ARQUIVO_ENTRADA
)

print(f"Objeto: {ARQUIVO_ENTRADA}")
print(f"Shape: {adata.shape}")
print(f"Doadores: {adata.obs['donor_id'].nunique()}")

if adata.shape != (32657, 22015):
    raise RuntimeError(
        f"Shape inesperado: {adata.shape}"
    )

if adata.obs["donor_id"].nunique() != 75:
    raise RuntimeError(
        "Número de doadores diferente de 75."
    )

if "connectivities" not in adata.obsp:
    raise KeyError(
        "connectivities ausente."
    )

if "distances" not in adata.obsp:
    raise KeyError(
        "distances ausente."
    )

if (
    adata.uns["neighbors"]["params"].get("use_rep")
    != "X_pca"
):
    raise RuntimeError(
        "O grafo ativo não usa X_pca."
    )

if int(
    adata.uns["neighbors"]["params"].get("n_neighbors")
) != 15:
    raise RuntimeError(
        "O grafo ativo não usa 15 vizinhos."
    )

# ------------------------------------------------------------
# Metadados — mesma lógica do worker
# ------------------------------------------------------------

renomear = {}

if (
    "Age_at_death" not in adata.obs.columns
    and "Age.at.death" in adata.obs.columns
):
    renomear["Age.at.death"] = "Age_at_death"

if (
    "Braak_stage" not in adata.obs.columns
    and "Braak.stage" in adata.obs.columns
):
    renomear["Braak.stage"] = "Braak_stage"

if (
    "Thal_phase" not in adata.obs.columns
    and "Thal.phase" in adata.obs.columns
):
    renomear["Thal.phase"] = "Thal_phase"

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

necessarias = [
    "donor_id",
    "Modelo_Recessivo",
    "Age_at_death",
    "Braak_stage",
    "Thal_phase",
]

if adata.obs[necessarias].isna().any().any():
    raise RuntimeError(
        "Existem NAs nos metadados da Rota 2."
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

for col in [
    "Braak_stage",
    "Thal_phase",
]:
    categorias = sorted(
        adata.obs[col]
        .astype(str)
        .unique()
    )

    adata.obs[col] = pd.Categorical(
        adata.obs[col].astype(str),
        categories=categorias,
        ordered=True,
    )

# Consistência donor-level
covariaveis = [
    "Modelo_Recessivo",
    "Age_at_death",
    "Braak_stage",
    "Thal_phase",
]

variacao = (
    adata.obs
    .groupby(
        "donor_id",
        observed=True,
    )[covariaveis]
    .nunique(
        dropna=False,
    )
)

if variacao.gt(1).any(axis=1).any():
    raise RuntimeError(
        "Metadados inconsistentes dentro de donor_id."
    )

# ------------------------------------------------------------
# Milo sem edgeR
# ------------------------------------------------------------

print("\nConstruindo neighborhoods...")

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

assinatura = assinatura_neighborhoods(
    rna,
    m,
)

print(f"Neighborhoods: {m.n_vars}")
print(f"SHA-256: {assinatura}")

if m.n_vars != 2201:
    raise RuntimeError(
        f"Esperávamos 2201 neighborhoods; encontradas {m.n_vars}."
    )

if assinatura != HASH_ESPERADO:
    raise RuntimeError(
        "O SHA-256 das neighborhoods NÃO corresponde ao hash "
        "observado nas Rotas 0/1/2."
    )

print("[OK] Neighborhoods idênticas às rotas anteriores.")

# ------------------------------------------------------------
# Metadados no MESMO ORDER dos doadores da matriz Milo
# ------------------------------------------------------------

doadores = (
    adata.obs[
        [
            "donor_id",
            "Modelo_Recessivo",
            "Age_at_death",
            "Braak_stage",
            "Thal_phase",
        ]
    ]
    .drop_duplicates(
        subset="donor_id"
    )
    .copy()
)

doadores["donor_id"] = (
    doadores["donor_id"]
    .astype(str)
)

for col in [
    "Modelo_Recessivo",
    "Age_at_death",
    "Braak_stage",
    "Thal_phase",
]:
    doadores[col] = (
        doadores[col]
        .astype(str)
    )

doadores = (
    doadores
    .set_index("donor_id")
    .loc[
        m.obs_names.astype(str)
    ]
)

# ------------------------------------------------------------
# Exporta counts: neighborhoods nas linhas, doadores nas colunas
# ------------------------------------------------------------

X = m.X

if sp.issparse(X):
    X = X.toarray()

X = np.asarray(
    X,
    dtype=np.int64,
)

counts = pd.DataFrame(
    X.T,
    index=m.var_names.astype(str),
    columns=m.obs_names.astype(str),
)

counts.index.name = "Neighborhood"

counts.to_csv(
    PASTA_SAIDA
    / "rota2_counts_neighborhood_x_donor.tsv",
    sep="\t",
)

doadores.to_csv(
    PASTA_SAIDA
    / "rota2_metadata_donor.tsv",
    sep="\t",
)

nhood_meta = pd.DataFrame(
    index=m.var_names.astype(str)
)

nhood_meta.index.name = "Neighborhood"

if "index_cell" in m.var.columns:
    nhood_meta["index_cell"] = (
        m.var["index_cell"]
        .astype(str)
        .to_numpy()
    )

if "kth_distance" in m.var.columns:
    nhood_meta["kth_distance"] = (
        m.var["kth_distance"]
        .to_numpy()
    )

nhood_meta.to_csv(
    PASTA_SAIDA
    / "rota2_neighborhood_metadata.tsv",
    sep="\t",
)

print("\nArquivos exportados:")
print(
    PASTA_SAIDA
    / "rota2_counts_neighborhood_x_donor.tsv"
)
print(
    PASTA_SAIDA
    / "rota2_metadata_donor.tsv"
)
print(
    PASTA_SAIDA
    / "rota2_neighborhood_metadata.tsv"
)
print("\nNenhum teste edgeR foi executado neste script.")