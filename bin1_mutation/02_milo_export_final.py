# ============================================================
# 02_milo_export_final.py
#
# MILO FINAL — ETAPA 1
# Exportação congelada das neighborhoods para edgeR 4.10.x
#
# EXECUTAR NO AMBIENTE:
#   conda activate milo_bin1
#
# O script:
#   - carrega o objeto pré-Milo definitivo;
#   - NÃO recalcula normalização/PCA/KNN/Leiden/UMAP;
#   - constrói as neighborhoods sobre o KNN existente;
#   - usa prop=0.10 e seed=0;
#   - exige o SHA-256 já validado;
#   - exporta counts donor x neighborhood, metadados donor-level
#     e metadados das neighborhoods;
#   - NÃO executa edgeR.
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
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import pertpy as pt
import anndata
import mudata
import rpy2


SCRIPT_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = (
    SCRIPT_DIR
    / "adata_limpo_anotado_sem_harmony_res0.5_FINAL.h5ad"
)

PASTA_BASE = (
    SCRIPT_DIR
    / "results"
    / "milo_final_edger410_reuniao"
)

PASTA_EXPORT = PASTA_BASE / "export"
PASTA_AUDITORIA = PASTA_BASE / "auditoria"

for _pasta in [
    PASTA_BASE,
    PASTA_EXPORT,
    PASTA_AUDITORIA,
]:
    _pasta.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PARÂMETROS CONGELADOS
# ============================================================

NHOOD_PROP = 0.10
NHOOD_SEED = 0
PURITY_LIMIAR = 0.60

N_CELULAS_ESPERADO = 32657
N_GENES_ESPERADO = 22015
N_DOADORES_ESPERADO = 75
N_NAO_RECESSIVOS_ESPERADO = 64
N_RECESSIVOS_ESPERADO = 11
N_NEIGHBORS_ESPERADO = 15
N_NHOODS_ESPERADO = 2201

HASH_ESPERADO = (
    "c06ce07f1ded7c9d96e932901a84e864"
    "f1487ec62ca952ee9152ec341aa08033"
)

COL_LEIDEN = "leiden_0.5"
COL_BIO = "Estado_Microglial_Biologico"
COL_CONF = "Confianca_Anotacao"
COL_QC = "Flag_QC_Anotacao"


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



def detectar_coluna_sexo(obs):
    """
    Detecta uma coluna de sexo/gênero biológico nos metadados.
    A análise Milo usará uma variável padronizada chamada 'Sex'.
    """
    candidatos = [
        "Sex",
        "sex",
        "SEX",
        "Gender",
        "gender",
        "GENDER",
    ]

    presentes = [
        col
        for col in candidatos
        if col in obs.columns
    ]

    if not presentes:
        candidatos_textuais = [
            col
            for col in obs.columns
            if (
                "sex" in str(col).lower()
                or "gender" in str(col).lower()
            )
        ]

        raise KeyError(
            "Nenhuma coluna de sexo/gênero foi encontrada. "
            f"Colunas candidatas por texto: {candidatos_textuais}"
        )

    return presentes[0]


def normalizar_sexo(valor):
    if pd.isna(valor):
        return np.nan

    s = str(valor).strip().lower()

    feminino = {
        "f",
        "female",
        "feminino",
        "woman",
        "mulher",
        "xx",
    }

    masculino = {
        "m",
        "male",
        "masculino",
        "man",
        "homem",
        "xy",
    }

    if s in feminino:
        return "Female"

    if s in masculino:
        return "Male"

    return np.nan


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


def mapa_unico_por_cluster(
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
            f"{coluna!r} não é unívoca por Leiden."
        )

    return dict(
        zip(
            tabela[COL_LEIDEN].astype(str),
            tabela[coluna].astype(str),
        )
    )


# ============================================================
# INÍCIO
# ============================================================

print("=" * 78)
print("MILO FINAL — ETAPA 1 — EXPORTAÇÃO PARA edgeR 4.10.x")
print("=" * 78)
print(f"Objeto: {ARQUIVO_ENTRADA}")
print(f"Pertpy: {pt.__version__}")
print(f"Python: {sys.version.split()[0]}")
print("=" * 78)

if pt.__version__ != "0.10.0":
    raise RuntimeError(
        "Esta exportação foi validada para Pertpy 0.10.0."
    )

if not ARQUIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"Objeto final não encontrado:\n{ARQUIVO_ENTRADA}"
    )


# ============================================================
# 1. OBJETO E GRAFO
# ============================================================

print("\n[1] Carregando objeto final...")
adata = sc.read_h5ad(
    ARQUIVO_ENTRADA
)

print(f"Shape: {adata.shape}")

if adata.shape != (
    N_CELULAS_ESPERADO,
    N_GENES_ESPERADO,
):
    raise RuntimeError(
        f"Shape inesperado: {adata.shape}"
    )

if "donor_id" not in adata.obs.columns:
    raise KeyError(
        "donor_id ausente."
    )

if (
    adata.obs["donor_id"].nunique()
    != N_DOADORES_ESPERADO
):
    raise RuntimeError(
        "Número de doadores diferente de 75."
    )

for chave in [
    "X_pca",
    "X_umap",
]:
    if chave not in adata.obsm:
        raise KeyError(
            f"{chave} ausente."
        )

for chave in [
    "connectivities",
    "distances",
]:
    if chave not in adata.obsp:
        raise KeyError(
            f"{chave} ausente."
        )

if "neighbors" not in adata.uns:
    raise KeyError(
        "neighbors ausente de adata.uns."
    )

params = dict(
    adata.uns["neighbors"].get(
        "params",
        {},
    )
)

use_rep = params.get(
    "use_rep",
    "X_pca",
)

n_neighbors = int(
    params.get(
        "n_neighbors",
        -1,
    )
)

integracao = str(
    adata.uns.get(
        "Integracao_pipeline",
        "NA",
    )
)

resolucao = str(
    adata.uns.get(
        "Resolucao_Leiden_principal",
        "NA",
    )
)

print(
    f"use_rep={use_rep}; "
    f"n_neighbors={n_neighbors}; "
    f"integração={integracao}; "
    f"Leiden={resolucao}"
)

if use_rep != "X_pca":
    raise RuntimeError(
        "Grafo não está registrado em X_pca."
    )

if n_neighbors != N_NEIGHBORS_ESPERADO:
    raise RuntimeError(
        "KNN não possui 15 vizinhos."
    )

if integracao != "Sem_Harmony":
    raise RuntimeError(
        "Objeto não está marcado como Sem_Harmony."
    )

if resolucao not in {
    "0.5",
    "0.50",
}:
    raise RuntimeError(
        "Leiden principal não é 0.50."
    )

for col in [
    COL_LEIDEN,
    COL_BIO,
    COL_CONF,
    COL_QC,
]:
    if col not in adata.obs.columns:
        raise KeyError(
            f"Anotação final ausente: {col}"
        )

    if adata.obs[col].isna().any():
        raise RuntimeError(
            f"{col} contém NA."
        )

print("[OK] Objeto e grafo final validados.")


# ============================================================
# 2. METADADOS
# ============================================================

print("\n[2] Preparando metadados donor-level...")

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

coluna_sexo_original = detectar_coluna_sexo(
    adata.obs
)

print(
    f"Coluna de sexo detectada: {coluna_sexo_original!r}"
)

adata.obs["Sex"] = (
    adata.obs[coluna_sexo_original]
    .map(normalizar_sexo)
)

if adata.obs["Sex"].isna().any():
    valores_problematicos = sorted(
        adata.obs.loc[
            adata.obs["Sex"].isna(),
            coluna_sexo_original,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "Não foi possível padronizar todos os valores de sexo. "
        f"Valores problemáticos: {valores_problematicos}"
    )

obrigatorias = [
    "donor_id",
    "Mutacao_BIN1",
    "Age_at_death",
    "Sex",
    "ADNC",
    "Braak_stage",
    "Thal_phase",
]

faltantes = [
    col
    for col in obrigatorias
    if col not in adata.obs.columns
]

if faltantes:
    raise KeyError(
        f"Metadados ausentes: {faltantes}"
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

colunas_modelo = [
    "donor_id",
    "Modelo_Recessivo",
    "Age_at_death",
    "Sex",
    "ADNC",
    "Braak_stage",
    "Thal_phase",
]

if (
    adata.obs[
        colunas_modelo
    ]
    .isna()
    .any()
    .any()
):
    print(
        adata.obs[
            colunas_modelo
        ]
        .isna()
        .sum()
    )
    raise RuntimeError(
        "Há NA nos metadados; nenhuma célula será removida."
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

sex_presentes = set(
    adata.obs["Sex"]
    .astype(str)
    .unique()
)

if not sex_presentes.issubset({"Female", "Male"}):
    raise RuntimeError(
        f"Categorias de sexo inesperadas: {sorted(sex_presentes)}"
    )

if len(sex_presentes) < 2:
    raise RuntimeError(
        "A covariável Sex possui apenas uma categoria na coorte."
    )

adata.obs["Sex"] = pd.Categorical(
    adata.obs["Sex"].astype(str),
    categories=[
        "Female",
        "Male",
    ],
    ordered=True,
)

adnc_presentes = set(
    adata.obs["ADNC"]
    .astype(str)
    .unique()
)

adnc_ordem = [
    x
    for x in [
        "Low",
        "Intermediate",
        "High",
    ]
    if x in adnc_presentes
]

if set(adnc_ordem) != adnc_presentes:
    raise RuntimeError(
        f"Categorias ADNC inesperadas: "
        f"{sorted(adnc_presentes)}"
    )

adata.obs["ADNC"] = pd.Categorical(
    adata.obs["ADNC"].astype(str),
    categories=adnc_ordem,
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

cov_donor = [
    "Modelo_Recessivo",
    "Age_at_death",
    "Sex",
    "ADNC",
    "Braak_stage",
    "Thal_phase",
]

variacao = (
    adata.obs
    .groupby(
        "donor_id",
        observed=True,
    )[cov_donor]
    .nunique(
        dropna=False,
    )
)

inconsistentes = (
    variacao[
        variacao
        .gt(1)
        .any(axis=1)
    ]
)

if not inconsistentes.empty:
    print(inconsistentes)
    raise RuntimeError(
        "Há covariáveis inconsistentes dentro de donor_id."
    )

doadores = (
    adata.obs[
        [
            "donor_id",
            *cov_donor,
        ]
    ]
    .drop_duplicates(
        subset="donor_id"
    )
    .copy()
)

if len(doadores) != N_DOADORES_ESPERADO:
    raise RuntimeError(
        "Número donor-level inesperado."
    )

contagem_bin1 = (
    doadores["Modelo_Recessivo"]
    .astype(str)
    .value_counts()
)

n_nao = int(
    contagem_bin1.get(
        "Nao_Recessivo",
        0,
    )
)

n_rec = int(
    contagem_bin1.get(
        "Recessivo",
        0,
    )
)

if (
    n_nao != N_NAO_RECESSIVOS_ESPERADO
    or n_rec != N_RECESSIVOS_ESPERADO
):
    raise RuntimeError(
        f"Grupos inesperados: "
        f"Não-recessivo={n_nao}, recessivo={n_rec}"
    )

print(
    f"[OK] 75 doadores: "
    f"{n_nao} não recessivos / {n_rec} recessivos."
)

print("\nDistribuição donor-level de sexo:")
print(
    doadores["Sex"]
    .astype(str)
    .value_counts()
)

tabela_bin1_sexo = pd.crosstab(
    doadores["Modelo_Recessivo"].astype(str),
    doadores["Sex"].astype(str),
)

print("\nModelo recessivo x sexo:")
print(tabela_bin1_sexo)

tabela_bin1_sexo.to_csv(
    PASTA_AUDITORIA
    / "Crosstab_Modelo_Recessivo_x_Sex.csv"
)


# ============================================================
# 3. NEIGHBORHOODS
# ============================================================

print("\n[3] Construindo neighborhoods sobre o KNN existente...")

# Remove apenas resíduos Milo caso existam.
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

m.X = sp.csr_matrix(
    m.X,
    dtype=np.int64,
)

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
    f"Matriz Milo: {m.shape}"
)
print(
    f"SHA-256: {assinatura}"
)

if assinatura != HASH_ESPERADO:
    raise RuntimeError(
        "SHA-256 das neighborhoods não corresponde "
        "ao valor congelado."
    )

print(
    "[OK] 2201 neighborhoods idênticas às execuções anteriores."
)


# ============================================================
# 4. ANOTAÇÃO DAS NEIGHBORHOODS
# ============================================================

print("\n[4] Importando anotação final para as neighborhoods...")

milo.annotate_nhoods(
    mdata,
    anno_col=COL_BIO,
)

# Corrige annotation_labels para tipo simples.
if "annotation_labels" in m.uns:
    m.uns["annotation_labels"] = np.asarray(
        [
            str(x)
            for x in m.uns["annotation_labels"]
        ],
        dtype=str,
    )

mapa_bio = mapa_unico_por_cluster(
    adata,
    COL_BIO,
)
mapa_conf = mapa_unico_por_cluster(
    adata,
    COL_CONF,
)
mapa_qc = mapa_unico_por_cluster(
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

somas = contagens_leiden.sum(
    axis=1
)

if np.any(somas == 0):
    raise RuntimeError(
        "Neighborhood vazia detectada."
    )

frac_leiden = (
    contagens_leiden
    / somas[
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

m.varm["frac_Leiden_0_5"] = (
    frac_leiden
)

m.uns["Leiden_0_5_labels"] = np.asarray(
    labels_leiden,
    dtype=str,
)

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

nhood_size = np.asarray(
    rna.obsm["nhoods"]
    .sum(axis=0)
).ravel()

m.var["Nhood_size"] = (
    nhood_size.astype(int)
)

print(
    "Neighborhoods mistas (<60%): "
    f"{int(m.var['nhood_mista'].sum())}"
)


# ============================================================
# 5. EXPORTAÇÃO
# ============================================================

print("\n[5] Exportando arquivos congelados...")

# Counts: neighborhoods x donors
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

# donor metadata no MESMO order das colunas
doadores_export = (
    doadores
    .assign(
        donor_id=lambda x: x[
            "donor_id"
        ].astype(str)
    )
    .set_index(
        "donor_id"
    )
    .loc[
        counts.columns
    ]
    .copy()
)

for col in doadores_export.columns:
    doadores_export[col] = (
        doadores_export[col]
        .astype(str)
    )

counts_path = (
    PASTA_EXPORT
    / "milo_counts_neighborhood_x_donor.tsv"
)

metadata_path = (
    PASTA_EXPORT
    / "milo_metadata_donor.tsv"
)

nhood_path = (
    PASTA_EXPORT
    / "milo_neighborhood_metadata.tsv"
)

counts.to_csv(
    counts_path,
    sep="\t",
)

doadores_export.to_csv(
    metadata_path,
    sep="\t",
)

nhood_meta = pd.DataFrame(
    index=m.var_names.astype(str)
)

nhood_meta.index.name = "Neighborhood"

for col in [
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
]:
    if col in m.var.columns:
        nhood_meta[col] = (
            m.var[col]
            .astype(str)
            .to_numpy()
            if m.var[col].dtype.name in {
                "category",
                "object",
                "bool",
            }
            else m.var[col].to_numpy()
        )

# Garante numericidade das colunas essenciais.
nhood_meta["kth_distance"] = pd.to_numeric(
    nhood_meta["kth_distance"],
    errors="raise",
)

nhood_meta["Nhood_size"] = pd.to_numeric(
    nhood_meta["Nhood_size"],
    errors="raise",
).astype(int)

nhood_meta["nhood_annotation_frac"] = pd.to_numeric(
    nhood_meta["nhood_annotation_frac"],
    errors="raise",
)

nhood_meta["nhood_Leiden_0_5_frac"] = pd.to_numeric(
    nhood_meta["nhood_Leiden_0_5_frac"],
    errors="raise",
)

nhood_meta.to_csv(
    nhood_path,
    sep="\t",
)

# Matriz sparse cell x nhood para auditoria/reuso.
membership_path = (
    PASTA_EXPORT
    / "milo_membership_cell_x_neighborhood.npz"
)

sp.save_npz(
    membership_path,
    sp.csr_matrix(
        rna.obsm["nhoods"]
    ),
)

pd.Series(
    rna.obs_names.astype(str),
    name="cell_id",
).to_csv(
    PASTA_EXPORT
    / "milo_cell_ids.txt",
    index=False,
    header=False,
)

pd.Series(
    m.var_names.astype(str),
    name="Neighborhood",
).to_csv(
    PASTA_EXPORT
    / "milo_neighborhood_ids.txt",
    index=False,
    header=False,
)

# Auditoria da anotação por Leiden
pd.DataFrame(
    [
        {
            "Cluster_Leiden_0_5": cluster,
            "Estado_Microglial_Biologico": mapa_bio[cluster],
            "Confianca_Anotacao": mapa_conf[cluster],
            "Flag_QC_Anotacao": mapa_qc[cluster],
        }
        for cluster in sorted(
            mapa_bio,
            key=lambda x: (
                int(x)
                if str(x).isdigit()
                else str(x)
            ),
        )
    ]
).to_csv(
    PASTA_AUDITORIA
    / "Anotacao_clusters_importada.csv",
    index=False,
)

ambiente = {
    "Python": sys.version.split()[0],
    "Pertpy": pt.__version__,
    "Scanpy": sc.__version__,
    "AnnData": anndata.__version__,
    "MuData": mudata.__version__,
    "rpy2": rpy2.__version__,
    "Sistema": platform.platform(),
    "Arquivo_entrada": str(
        ARQUIVO_ENTRADA
    ),
    "N_celulas": adata.n_obs,
    "N_genes": adata.n_vars,
    "N_doadores": len(doadores_export),
    "N_nhoods": len(nhood_meta),
    "Nhood_prop": NHOOD_PROP,
    "Nhood_seed": NHOOD_SEED,
    "SHA256_neighborhoods": assinatura,
    "KNN_n_neighbors": n_neighbors,
    "KNN_use_rep": use_rep,
    "Integracao": integracao,
    "Leiden_principal": resolucao,
    "Coluna_sexo_original": coluna_sexo_original,
    "Categorias_Sex": sorted(sex_presentes),
    "EdgeR_executado_nesta_etapa": False,
}

salvar_json(
    PASTA_AUDITORIA
    / "Etapa1_export_status.json",
    ambiente,
)

print("\nArquivos exportados:")
print(counts_path)
print(metadata_path)
print(nhood_path)
print(membership_path)

print("\n" + "=" * 78)
print("ETAPA 1 CONCLUÍDA")
print("Nenhum teste edgeR foi executado.")
print("=" * 78)