#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_figuras_poster.py

Figuras finais para o pôster do projeto BIN1 / microglia / Milo.

OBJETIVO
--------
Este script NÃO refaz a inferência estatística e NÃO altera o pipeline final.
Ele apenas:

1) audita a coorte antes/depois do filtro ADNC;
2) reconstrói deterministicamente as neighborhoods Milo APENAS para recuperar
   as coordenadas delas no UMAP, exigindo o mesmo hash do pipeline congelado;
3) sobrepõe as neighborhoods significativas da rota G2 ao UMAP anotado;
4) gera um dotplot compacto com marcadores/programas já usados na anotação;
5) exporta PNG (600 dpi), PDF e tabelas de auditoria.

IMPORTANTE
----------
- O UMAP NÃO é recalculado.
- PCA/KNN/Leiden NÃO são recalculados.
- edgeR/SpatialFDR NÃO são recalculados.
- A rota G2 é lida do CSV final já produzido.
- A reconstrução Milo serve somente para recuperar a geometria/posição das
  neighborhoods e é abortada se o SHA-256 não coincidir com o conjunto final.

Ambiente esperado do projeto final:
    pertpy == 0.10.0

Execute a partir do diretório do projeto, por exemplo:
    conda activate milo_bin1
    python -u bin1_mutation/09_figuras_poster.py
"""

from __future__ import annotations

import hashlib
import json
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
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# =============================================================================
# 0. CAMINHOS E PARÂMETROS CONGELADOS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

ARQUIVO_BRUTO = SCRIPT_DIR / "BIN1_Mutation.h5ad"
ARQUIVO_FINAL = SCRIPT_DIR / "adata_limpo_anotado_sem_harmony_res0.5_FINAL.h5ad"

PASTA_MILO_FINAL = (
    SCRIPT_DIR
    / "results"
    / "milo_final_edger410_reuniao"
    / "final"
)

ARQUIVO_G2 = (
    PASTA_MILO_FINAL
    / "RotaG2_BIN1_Idade_Sexo_Braak_todos_neighborhoods.csv"
)

PASTA_SAIDA_FIGURAS = SCRIPT_DIR / "figures" / "poster_final"
PASTA_SAIDA_RESULTADOS = SCRIPT_DIR / "results" / "poster_final"

PASTA_SAIDA_FIGURAS.mkdir(parents=True, exist_ok=True)
PASTA_SAIDA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Identidade do pipeline final.
N_CELULAS_FINAL_ESPERADO = 32657
N_DOADORES_FINAL_ESPERADO = 75
N_NHOODS_ESPERADO = 2201
N_G2_SIG_ESPERADO = 55
N_G2_POS_ESPERADO = 43
N_G2_NEG_ESPERADO = 12

# O valor 84 é usado apenas como REFERÊNCIA de auditoria para a coorte bruta.
# O script NÃO o força: ele conta diretamente donor_id no arquivo bruto.
N_DOADORES_BRUTOS_REFERENCIA = 84

NHOOD_PROP = 0.10
NHOOD_SEED = 0
SPATIAL_FDR_LIMIAR = 0.10

HASH_NHOODS_ESPERADO = (
    "c06ce07f1ded7c9d96e932901a84e864"
    "f1487ec62ca952ee9152ec341aa08033"
)

COL_DONOR = "donor_id"
COL_ADNC = "ADNC"
COL_LEIDEN = "leiden_0.5"
COL_BIO = "Estado_Microglial_Biologico"

ADNC_MANTER = ["Low", "Intermediate", "High"]

# Figuras para pôster.
DPI_PNG = 600


# =============================================================================
# 1. RÓTULOS E MARCADORES COMPACTOS
# =============================================================================

# Rótulos abreviados somente para o eixo Y do dotplot/pôster.
ROTULOS_CLUSTER_CURTOS = {
    "0": "0  Activated-phagocytic / PVM",
    "1": "1  Activated immediate-response / stress",
    "2": "2  Probable neuronal-oligodendroglial contamination / doublet",
    "3": "3  Homeostatic FRMD4A+ / INPP5D+",
    "4": "4  Homeostatic / immunosurveillance",
    "5": "5  Inflammatory / metabolic",
    "6": "6  Neuronal-signature microglia - QC",
    "7": "7  Homeostatic / intermediate",
    "8": "8  Activated / antigen presentation",
    "9": "9  Activated / interferon-inflammatory",
    "10": "10  Donor-specific / Y-linked - QC",
    "11": "11  PVM-like / MRC1+ - donor-enriched",
    "12": "12  Phagocytic / antigen presentation - donor-specific",
}

# Painel ENXUTO, definido a partir das evidências que já existiam antes do Milo:
# - programas microgliais do script 01;
# - marcadores explicitamente usados na anotação final/relatório;
# - genes de QC para deixar os clusters problemáticos visualmente justificáveis.
#
# Não há seleção baseada em p-valor/logFC do Milo.
PAINEL_DOTPLOT = {
    "Homeostatic": [
        "P2RY12",
        "TMEM119",
        "INPP5D",
        "FRMD4A",
    ],
    "Activated / phagocytic / PVM": [
        "CD163",
        "GPNMB",
        "APOC1",
        "LPL",
        "MRC1",
    ],
    "Antigen / interferon": [
        "HLA-DRA",
        "CD74",
        "ISG15",
        "IFIT3",
    ],
    "Inflammatory / metabolic / stress": [
        "HAMP",
        "SLC11A1",
        "NFKBIA",
        "FOS",
    ],
    "QC": [
        "SNAP25",
        "MBP",
        "PRKY",
    ],
}


# =============================================================================
# 2. UTILITÁRIOS
# =============================================================================

def exigir_arquivo(caminho: Path, descricao: str) -> None:
    if not caminho.exists():
        raise FileNotFoundError(
            f"{descricao} não encontrado:\n  {caminho}\n"
            "Confirme que este script está dentro da pasta bin1_mutation."
        )


def salvar_figura(fig: plt.Figure, nome_base: str) -> None:
    """Salva a mesma figura em PNG de alta resolução e PDF."""
    png = PASTA_SAIDA_FIGURAS / f"{nome_base}.png"
    pdf = PASTA_SAIDA_FIGURAS / f"{nome_base}.pdf"

    fig.savefig(
        png,
        dpi=DPI_PNG,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        pdf,
        dpi=DPI_PNG,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"[OK] PNG: {png}")
    print(f"[OK] PDF: {pdf}")


def ordenar_clusters(valores) -> list[str]:
    return sorted(
        [str(v) for v in valores],
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )


def assinatura_neighborhoods(rna, milo_adata) -> str:
    """Mesma assinatura usada no pipeline Milo final."""
    nhoods = sp.csr_matrix(rna.obsm["nhoods"])
    nhoods.sort_indices()
    nhoods.eliminate_zeros()

    h = hashlib.sha256()
    h.update(np.asarray(nhoods.shape, dtype=np.int64).tobytes())
    h.update(np.asarray(nhoods.indptr, dtype=np.int64).tobytes())
    h.update(np.asarray(nhoods.indices, dtype=np.int64).tobytes())
    h.update(np.asarray(nhoods.data, dtype=np.int8).tobytes())

    if "index_cell" in milo_adata.var.columns:
        for nome in milo_adata.var["index_cell"].astype(str):
            h.update(nome.encode("utf-8"))
            h.update(b"\0")

    return h.hexdigest()


def validar_objeto_final(adata: sc.AnnData) -> None:
    if adata.n_obs != N_CELULAS_FINAL_ESPERADO:
        raise RuntimeError(
            f"Objeto final tem {adata.n_obs} células; "
            f"esperado {N_CELULAS_FINAL_ESPERADO}."
        )

    if COL_DONOR not in adata.obs.columns:
        raise KeyError(f"Coluna {COL_DONOR!r} ausente no objeto final.")

    n_donors = adata.obs[COL_DONOR].astype(str).nunique()
    if n_donors != N_DOADORES_FINAL_ESPERADO:
        raise RuntimeError(
            f"Objeto final tem {n_donors} doadores; "
            f"esperado {N_DOADORES_FINAL_ESPERADO}."
        )

    for col in [COL_LEIDEN, COL_BIO]:
        if col not in adata.obs.columns:
            raise KeyError(f"Coluna {col!r} ausente no objeto final.")

    if "X_umap" not in adata.obsm:
        raise KeyError(
            "X_umap ausente no objeto final. Este script NÃO recalcula UMAP."
        )

    if "connectivities" not in adata.obsp or "distances" not in adata.obsp:
        raise KeyError(
            "Grafo KNN final não encontrado em adata.obsp. "
            "Este script NÃO recalcula KNN."
        )


def detectar_cores_clusters(adata: sc.AnnData, clusters: list[str]) -> dict[str, tuple]:
    """Tenta reutilizar a paleta já armazenada pelo Scanpy."""
    chave = f"{COL_LEIDEN}_colors"

    # Cores existentes no objeto.
    if chave in adata.uns:
        serie = adata.obs[COL_LEIDEN]
        if hasattr(serie.dtype, "categories"):
            categorias = [str(x) for x in serie.cat.categories]
        else:
            categorias = ordenar_clusters(serie.astype(str).unique())

        cores = list(adata.uns[chave])
        if len(cores) >= len(categorias):
            mapa = {
                cat: to_rgba(cores[i])
                for i, cat in enumerate(categorias)
            }
            if all(c in mapa for c in clusters):
                return {c: mapa[c] for c in clusters}

    # Fallback determinístico apenas para visualização.
    cmap = plt.get_cmap("tab20")
    return {
        c: cmap(i % 20)
        for i, c in enumerate(clusters)
    }


# =============================================================================
# 3. AUDITORIA DA COORTE + MINI-FLUXO PARA METHODS
# =============================================================================

def auditar_coorte() -> dict:
    print("\n" + "=" * 78)
    print("[1/5] AUDITORIA DA COORTE")
    print("=" * 78)

    exigir_arquivo(ARQUIVO_BRUTO, "Arquivo bruto")
    exigir_arquivo(ARQUIVO_FINAL, "Objeto final")

    # backed='r' evita carregar toda a matriz de expressão para a RAM.
    bruto = sc.read_h5ad(ARQUIVO_BRUTO, backed="r")

    if COL_DONOR not in bruto.obs.columns:
        raise KeyError(f"{COL_DONOR!r} ausente no arquivo bruto.")
    if COL_ADNC not in bruto.obs.columns:
        raise KeyError(f"{COL_ADNC!r} ausente no arquivo bruto.")

    n_cells_pre = int(bruto.n_obs)
    donor_pre = bruto.obs[COL_DONOR].astype(str)
    n_donors_pre = int(donor_pre.nunique())

    adnc_str = bruto.obs[COL_ADNC].astype(str)
    mask_ad = adnc_str.isin(ADNC_MANTER)

    n_cells_pos = int(mask_ad.sum())
    n_donors_pos = int(
        bruto.obs.loc[mask_ad, COL_DONOR].astype(str).nunique()
    )

    adnc_cells = (
        adnc_str.value_counts(dropna=False)
        .rename_axis("ADNC")
        .reset_index(name="N_cells")
    )

    donors_unique = bruto.obs[[COL_DONOR, COL_ADNC]].copy()
    donors_unique[COL_DONOR] = donors_unique[COL_DONOR].astype(str)
    donors_unique[COL_ADNC] = donors_unique[COL_ADNC].astype(str)

    variacao_adnc = donors_unique.groupby(COL_DONOR)[COL_ADNC].nunique(dropna=False)
    if (variacao_adnc > 1).any():
        raise RuntimeError(
            "ADNC varia dentro de pelo menos um donor_id no arquivo bruto."
        )

    donors_unique = donors_unique.drop_duplicates(subset=COL_DONOR)
    adnc_donors = (
        donors_unique[COL_ADNC]
        .value_counts(dropna=False)
        .rename_axis("ADNC")
        .reset_index(name="N_donors")
    )

    # Fecha o backed file antes de carregar o objeto final em memória.
    if getattr(bruto, "file", None) is not None:
        try:
            bruto.file.close()
        except Exception:
            pass

    final = sc.read_h5ad(ARQUIVO_FINAL, backed="r")
    n_cells_final = int(final.n_obs)
    n_donors_final = int(final.obs[COL_DONOR].astype(str).nunique())

    if getattr(final, "file", None) is not None:
        try:
            final.file.close()
        except Exception:
            pass

    # Checagens críticas.
    if n_cells_pos != N_CELULAS_FINAL_ESPERADO:
        raise RuntimeError(
            "O filtro ADNC aplicado diretamente ao arquivo bruto resultou em "
            f"{n_cells_pos} células; esperado {N_CELULAS_FINAL_ESPERADO}."
        )

    if n_donors_pos != N_DOADORES_FINAL_ESPERADO:
        raise RuntimeError(
            "O filtro ADNC aplicado diretamente ao arquivo bruto resultou em "
            f"{n_donors_pos} doadores; esperado {N_DOADORES_FINAL_ESPERADO}."
        )

    if n_cells_final != N_CELULAS_FINAL_ESPERADO or n_donors_final != N_DOADORES_FINAL_ESPERADO:
        raise RuntimeError(
            "O objeto final não corresponde à coorte final esperada."
        )

    referencia_84_confirmada = n_donors_pre == N_DOADORES_BRUTOS_REFERENCIA

    resumo = {
        "dataset_label": "SEA-AD microglial dataset",
        "raw_cells": n_cells_pre,
        "raw_donors": n_donors_pre,
        "reference_raw_donors_84_confirmed": bool(referencia_84_confirmada),
        "filter": "ADNC in {Low, Intermediate, High}; Not AD excluded",
        "post_filter_cells": n_cells_pos,
        "post_filter_donors": n_donors_pos,
        "final_object_cells": n_cells_final,
        "final_object_donors": n_donors_final,
        "excluded_cells": n_cells_pre - n_cells_pos,
        "excluded_donors": n_donors_pre - n_donors_pos,
    }

    pd.DataFrame([resumo]).to_csv(
        PASTA_SAIDA_RESULTADOS / "Cohort_audit_summary.csv",
        index=False,
    )
    adnc_cells.to_csv(
        PASTA_SAIDA_RESULTADOS / "Cohort_ADNC_cells_pre_filter.csv",
        index=False,
    )
    adnc_donors.to_csv(
        PASTA_SAIDA_RESULTADOS / "Cohort_ADNC_donors_pre_filter.csv",
        index=False,
    )

    with open(
        PASTA_SAIDA_RESULTADOS / "Cohort_audit_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    print(f"Células pré-filtro : {n_cells_pre:,}")
    print(f"Doadores pré-filtro: {n_donors_pre}")
    print(f"Células pós-filtro : {n_cells_pos:,}")
    print(f"Doadores pós-filtro: {n_donors_pos}")

    if referencia_84_confirmada:
        print("[OK] A memória 84 -> 75 doadores foi confirmada diretamente no .h5ad.")
    else:
        print(
            "[ATENÇÃO] O arquivo bruto NÃO contém exatamente 84 donor_id únicos. "
            f"Encontrado: {n_donors_pre}. Use o valor auditado no pôster."
        )

    gerar_fluxo_coorte(resumo)
    return resumo


def gerar_fluxo_coorte(resumo: dict) -> None:
    """Mini-figura horizontal para incorporar ao Methods se houver espaço."""
    fig, ax = plt.subplots(figsize=(12.0, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    caixas = [
        (
            0.03,
            0.21,
            0.25,
            0.58,
            "SEA-AD microglial dataset\n"
            f"{resumo['raw_cells']:,} cells | {resumo['raw_donors']} donors",
        ),
        (
            0.375,
            0.21,
            0.25,
            0.58,
            "Clinical filter\n"
            "ADNC Low / Intermediate / High\n"
            "Not AD excluded",
        ),
        (
            0.72,
            0.21,
            0.25,
            0.58,
            "Final analysis cohort\n"
            f"{resumo['post_filter_cells']:,} cells | {resumo['post_filter_donors']} donors\n"
            "11 homozygous risk | 64 non-recessive",
        ),
    ]

    for x, y, w, h, texto in caixas:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1.4,
            facecolor="white",
            edgecolor="0.35",
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            texto,
            ha="center",
            va="center",
            fontsize=11,
        )

    for x1, x2 in [(0.285, 0.37), (0.63, 0.715)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 0.50),
                (x2, 0.50),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.4,
                color="0.30",
            )
        )

    ax.set_title(
        "Cohort definition",
        fontsize=14,
        fontweight="bold",
        pad=8,
    )
    salvar_figura(fig, "Fig_Methods_Cohort_Flow")


# =============================================================================
# 4. RECONSTRUÇÃO GEOMÉTRICA DAS NEIGHBORHOODS (SEM TESTE ESTATÍSTICO)
# =============================================================================

def reconstruir_milo_para_visualizacao(adata: sc.AnnData):
    print("\n" + "=" * 78)
    print("[2/5] RECONSTRUÇÃO DAS NEIGHBORHOODS PARA POSICIONAMENTO NO UMAP")
    print("=" * 78)

    if pt.__version__ != "0.10.0":
        raise RuntimeError(
            "Para reproduzir exatamente as neighborhoods finais, este script exige "
            f"pertpy 0.10.0. Versão encontrada: {pt.__version__}."
        )

    # Remove somente resíduos Milo; NÃO toca no KNN/UMAP existente.
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
        del adata.uns["nhood_neighbors_key"]

    milo = pt.tl.Milo()
    mdata = milo.load(adata)

    # Usa EXATAMENTE o grafo KNN existente no AnnData.
    milo.make_nhoods(
        mdata["rna"],
        prop=NHOOD_PROP,
        seed=NHOOD_SEED,
    )

    # count_nhoods cria a modalidade 'milo'; NÃO executa edgeR.
    mdata = milo.count_nhoods(
        mdata,
        sample_col=COL_DONOR,
    )

    m = mdata["milo"]
    rna = mdata["rna"]

    if m.shape != (N_DOADORES_FINAL_ESPERADO, N_NHOODS_ESPERADO):
        raise RuntimeError(
            f"Matriz Milo reconstruída = {m.shape}; esperado "
            f"({N_DOADORES_FINAL_ESPERADO}, {N_NHOODS_ESPERADO})."
        )

    hash_atual = assinatura_neighborhoods(rna, m)
    print(f"SHA-256 reconstruído: {hash_atual}")

    if hash_atual != HASH_NHOODS_ESPERADO:
        raise RuntimeError(
            "ABORTANDO: as neighborhoods reconstruídas não são idênticas às finais.\n"
            f"Esperado: {HASH_NHOODS_ESPERADO}\n"
            f"Obtido  : {hash_atual}"
        )

    print("[OK] Neighborhoods idênticas às do pipeline final.")

    # Apenas posicionamento visual no X_umap JÁ existente.
    milo.build_nhood_graph(
        mdata,
        basis="X_umap",
    )

    if "X_milo_graph" not in m.varm:
        raise KeyError("Pertpy não criou mdata['milo'].varm['X_milo_graph'].")

    coords = np.asarray(m.varm["X_milo_graph"], dtype=float)
    if coords.shape != (N_NHOODS_ESPERADO, 2):
        raise RuntimeError(
            f"Coordenadas Milo inesperadas: {coords.shape}."
        )

    # Nhood_size para dimensionamento dos círculos.
    m.var["Nhood_size"] = np.asarray(
        rna.obsm["nhoods"].sum(axis=0)
    ).ravel().astype(int)

    return mdata


# =============================================================================
# 5. G2 SOBREPOSTO AO UMAP ANOTADO
# =============================================================================

def carregar_validar_g2(mdata) -> pd.DataFrame:
    exigir_arquivo(ARQUIVO_G2, "CSV final da rota G2")

    g2 = pd.read_csv(ARQUIVO_G2)

    obrigatorias = {
        "Neighborhood",
        "logFC",
        "SpatialFDR",
    }
    faltantes = obrigatorias - set(g2.columns)
    if faltantes:
        raise KeyError(
            f"CSV G2 sem colunas obrigatórias: {sorted(faltantes)}"
        )

    if len(g2) != N_NHOODS_ESPERADO:
        raise RuntimeError(
            f"CSV G2 possui {len(g2)} linhas; esperado {N_NHOODS_ESPERADO}."
        )

    g2["Neighborhood"] = g2["Neighborhood"].astype(str)
    g2["logFC"] = pd.to_numeric(g2["logFC"], errors="raise")
    g2["SpatialFDR"] = pd.to_numeric(g2["SpatialFDR"], errors="raise")

    m = mdata["milo"]
    ids_milo = pd.Index(m.var_names.astype(str), name="Neighborhood")

    if set(g2["Neighborhood"]) != set(ids_milo):
        faltam_no_csv = sorted(set(ids_milo) - set(g2["Neighborhood"]))[:10]
        extras_csv = sorted(set(g2["Neighborhood"]) - set(ids_milo))[:10]
        raise RuntimeError(
            "IDs de neighborhoods do CSV G2 não correspondem ao Milo reconstruído.\n"
            f"Exemplos ausentes no CSV: {faltam_no_csv}\n"
            f"Exemplos extras no CSV: {extras_csv}"
        )

    g2 = (
        g2.set_index("Neighborhood")
        .loc[ids_milo]
        .copy()
    )

    coords = np.asarray(m.varm["X_milo_graph"], dtype=float)
    g2["UMAP1_nhood"] = coords[:, 0]
    g2["UMAP2_nhood"] = coords[:, 1]
    g2["Nhood_size_plot"] = np.asarray(m.var["Nhood_size"], dtype=int)

    sig = g2["SpatialFDR"] < SPATIAL_FDR_LIMIAR
    n_sig = int(sig.sum())
    n_pos = int((sig & (g2["logFC"] > 0)).sum())
    n_neg = int((sig & (g2["logFC"] < 0)).sum())

    print(f"G2 significativas: {n_sig}")
    print(f"  positivas: {n_pos}")
    print(f"  negativas: {n_neg}")

    if (n_sig, n_pos, n_neg) != (
        N_G2_SIG_ESPERADO,
        N_G2_POS_ESPERADO,
        N_G2_NEG_ESPERADO,
    ):
        raise RuntimeError(
            "O arquivo G2 não corresponde ao resultado final esperado.\n"
            f"Encontrado: sig={n_sig}, pos={n_pos}, neg={n_neg}\n"
            f"Esperado : sig={N_G2_SIG_ESPERADO}, "
            f"pos={N_G2_POS_ESPERADO}, neg={N_G2_NEG_ESPERADO}"
        )

    g2.reset_index().to_csv(
        PASTA_SAIDA_RESULTADOS / "G2_with_UMAP_coordinates_for_poster.csv",
        index=False,
    )

    return g2


def escala_tamanho_nhood(valores: np.ndarray) -> np.ndarray:
    """Transforma Nhood_size em área de marcador moderada e legível."""
    valores = np.asarray(valores, dtype=float)
    raiz = np.sqrt(np.clip(valores, 0, None))

    minimo = float(np.min(raiz))
    maximo = float(np.max(raiz))

    if maximo <= minimo:
        return np.full_like(raiz, 90.0)

    norm = (raiz - minimo) / (maximo - minimo)
    return 55.0 + 135.0 * norm


def plotar_g2_sobre_umap(
    adata: sc.AnnData,
    g2: pd.DataFrame,
    incluir_legenda_estados: bool,
    nome_base: str,
) -> None:
    coords = np.asarray(adata.obsm["X_umap"], dtype=float)
    leiden = adata.obs[COL_LEIDEN].astype(str)
    clusters = ordenar_clusters(leiden.unique())
    cores = detectar_cores_clusters(adata, clusters)

    fig, ax = plt.subplots(figsize=(9.8, 7.8))

    # Fundo: células do UMAP final, com cores originais suavizadas.
    for cluster in clusters:
        mask = leiden.to_numpy() == cluster
        rgba = np.array(cores[cluster], dtype=float)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.2,
            color=[rgba],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    # IDs dos clusters no centro geométrico (mediana é robusta a outliers).
    for cluster in clusters:
        mask = leiden.to_numpy() == cluster
        centro = np.median(coords[mask], axis=0)
        ax.text(
            centro[0],
            centro[1],
            cluster,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="circle,pad=0.20",
                facecolor="white",
                edgecolor="0.25",
                linewidth=0.7,
                alpha=0.88,
            ),
            zorder=4,
        )

    sig = g2[g2["SpatialFDR"] < SPATIAL_FDR_LIMIAR].copy()

    vmax = float(np.max(np.abs(sig["logFC"].to_numpy(dtype=float))))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    tamanhos = escala_tamanho_nhood(sig["Nhood_size_plot"].to_numpy())

    scatter = ax.scatter(
        sig["UMAP1_nhood"],
        sig["UMAP2_nhood"],
        c=sig["logFC"],
        cmap="coolwarm",
        norm=norm,
        s=tamanhos,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(
        "G2 logFC\n(homozygous risk vs non-recessive)",
        fontsize=9,
    )
    cbar.ax.tick_params(labelsize=8)

    # Legenda do tamanho das neighborhoods.
    # Para evitar um salto grande entre os exemplos da legenda, usamos
    # três tamanhos representativos: mínimo, mediana e máximo.
    valores_base = sig["Nhood_size_plot"].to_numpy(dtype=float)
    valores_ref = np.array([
        int(np.min(valores_base)),
        int(np.median(valores_base)),
        int(np.max(valores_base)),
    ], dtype=int)

    # Remove duplicatas caso min/mediana/max coincidam em conjuntos pequenos.
    valores_ref = np.unique(valores_ref)
    tamanhos_ref = escala_tamanho_nhood(valores_ref)

    handles_size = [
        ax.scatter(
            [],
            [],
            s=float(s),
            facecolor="0.75",
            edgecolor="white",
            linewidth=0.7,
            label=f"{int(v)} cells",
        )
        for v, s in zip(valores_ref, tamanhos_ref)
    ]

    leg_size = ax.legend(
        handles=handles_size,
        title="Neighborhood size (cells)",
        loc="lower left",
        frameon=True,
        fontsize=7.5,
        title_fontsize=8,
        borderpad=0.5,
        labelspacing=0.7,
    )
    ax.add_artist(leg_size)

    if incluir_legenda_estados:
        handles_states = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5,
                markerfacecolor=cores[c],
                markeredgecolor="none",
                label=ROTULOS_CLUSTER_CURTOS.get(c, c),
            )
            for c in clusters
        ]
        ax.legend(
            handles=handles_states,
            title="Microglial states",
            loc="center left",
            bbox_to_anchor=(1.04, 0.50),
            frameon=False,
            fontsize=7.2,
            title_fontsize=8,
        )

    # Título e subtítulo separados para deixar explícito o que significa G2.
    fig.suptitle(
        "G2 differential-abundance neighborhoods on the annotated microglial UMAP",
        fontsize=12,
        fontweight="bold",
        y=0.975,
    )
    ax.set_title(
        "G2: genotype + age + sex + Braak",
        fontsize=9.5,
        fontweight="normal",
        pad=5,
    )

    # A explicação das cores sai de dentro do eixo para não colidir com a
    # legenda de tamanho das neighborhoods. Ela passa a funcionar como uma
    # caption limpa imediatamente abaixo da figura.
    fig.text(
        0.50,
        0.035,
        (
            "G2 significant neighborhoods (SpatialFDR < 0.10): "
            "red = enriched and blue = depleted in homozygous risk-allele carriers; "
            "circle size reflects neighborhood size."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="0.22",
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Reserva espaço real para título/subtítulo/caption e, na versão FULL,
    # para a legenda externa dos 13 estados microgliais.
    fig.subplots_adjust(
        top=0.88,
        bottom=0.14,
        right=0.80 if incluir_legenda_estados else 0.92,
    )

    salvar_figura(fig, nome_base)



# =============================================================================
# 6. UMAP FINAL ANOTADO EM INGLÊS
# =============================================================================

def plotar_umap_anotado_ingles(adata: sc.AnnData) -> None:
    """
    Exporta o UMAP final com os 13 estados em inglês usando EXATAMENTE:
    - as coordenadas X_umap do objeto final;
    - os clusters Leiden 0.50 do objeto final;
    - a mesma função/paleta usada no overlay G2.

    Portanto, cluster 0 tem a mesma cor no UMAP simples e no overlay G2,
    cluster 1 tem a mesma cor nos dois, e assim sucessivamente.
    Nenhuma análise é recalculada.
    """
    coords = np.asarray(adata.obsm["X_umap"], dtype=float)
    leiden = adata.obs[COL_LEIDEN].astype(str)
    clusters = ordenar_clusters(leiden.unique())
    cores = detectar_cores_clusters(adata, clusters)

    fig, ax = plt.subplots(figsize=(11.6, 8.4))

    for cluster in clusters:
        mask = leiden.to_numpy() == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.7,
            color=[cores[cluster]],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    for cluster in clusters:
        mask = leiden.to_numpy() == cluster
        centro = np.median(coords[mask], axis=0)

        ax.text(
            centro[0],
            centro[1],
            cluster,
            ha="center",
            va="center",
            fontsize=10.6,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="circle,pad=0.22",
                facecolor="white",
                edgecolor="0.25",
                linewidth=0.8,
                alpha=0.92,
            ),
            zorder=4,
        )

    handles_states = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=cores[c],
            markeredgecolor="none",
            label=ROTULOS_CLUSTER_CURTOS.get(c, c),
        )
        for c in clusters
    ]

    ax.legend(
        handles=handles_states,
        title="Microglial states",
        loc="center left",
        bbox_to_anchor=(1.015, 0.50),
        frameon=False,
        fontsize=9.6,
        title_fontsize=10.4,
        handletextpad=0.52,
        labelspacing=0.78,
        borderaxespad=0.0,
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(
        top=0.96,
        bottom=0.09,
        left=0.07,
        right=0.68,
    )

    salvar_figura(
        fig,
        "Fig_Annotated_Microglial_UMAP_English",
    )


# =============================================================================
# 7. DOTPLOT COMPACTO PARA JUSTIFICAR A ANOTAÇÃO
# =============================================================================

def obter_painel_presente(adata: sc.AnnData):
    genes_presentes = set(map(str, adata.var_names))

    painel = {}
    ausentes = []

    for grupo, genes in PAINEL_DOTPLOT.items():
        presentes = [g for g in genes if g in genes_presentes]
        faltantes = [g for g in genes if g not in genes_presentes]
        painel[grupo] = presentes
        ausentes.extend((grupo, g) for g in faltantes)

    painel = {k: v for k, v in painel.items() if v}

    if not painel:
        raise RuntimeError("Nenhum gene do painel compacto está presente no objeto final.")

    if ausentes:
        print("Genes do painel ausentes no objeto final:")
        for grupo, gene in ausentes:
            print(f"  - {grupo}: {gene}")

    pd.DataFrame(
        [
            {
                "Group": grupo,
                "Gene": gene,
                "Present": gene in genes_presentes,
            }
            for grupo, genes in PAINEL_DOTPLOT.items()
            for gene in genes
        ]
    ).to_csv(
        PASTA_SAIDA_RESULTADOS / "Dotplot_gene_panel_audit.csv",
        index=False,
    )

    return painel


def calcular_metricas_dotplot(
    adata: sc.AnnData,
    painel: dict[str, list[str]],
):
    genes = [g for grupo in painel.values() for g in grupo]
    clusters = ordenar_clusters(adata.obs[COL_LEIDEN].astype(str).unique())

    # Subset pequeno: ~20 genes, portanto seguro em memória.
    sub = adata[:, genes]
    X = sub.X

    linhas = []

    cluster_series = adata.obs[COL_LEIDEN].astype(str).to_numpy()

    for cluster in clusters:
        mask = cluster_series == cluster
        n = int(mask.sum())
        if n == 0:
            continue

        Xc = X[mask]

        if sp.issparse(Xc):
            Xc = sp.csr_matrix(Xc)
            mean = np.asarray(Xc.mean(axis=0)).ravel()
            pct = np.asarray(Xc.getnnz(axis=0)).ravel() / n * 100.0
        else:
            arr = np.asarray(Xc)
            mean = arr.mean(axis=0)
            pct = (arr > 0).mean(axis=0) * 100.0

        for gene, m, p in zip(genes, mean, pct):
            linhas.append(
                {
                    "Cluster": cluster,
                    "Gene": gene,
                    "Mean_log1p_CP10k": float(m),
                    "Percent_expressing": float(p),
                }
            )

    df = pd.DataFrame(linhas)

    # Mesmo princípio de standard_scale='var': cada gene é escalado ENTRE clusters.
    scaled = []
    for gene, grupo in df.groupby("Gene", sort=False):
        valores = grupo["Mean_log1p_CP10k"].to_numpy(dtype=float)
        minimo = float(np.min(valores))
        maximo = float(np.max(valores))

        if maximo <= minimo:
            escala = np.zeros_like(valores)
        else:
            escala = (valores - minimo) / (maximo - minimo)

        temp = grupo.copy()
        temp["Relative_mean_expression_0_1_per_gene"] = escala
        scaled.append(temp)

    df = pd.concat(scaled, ignore_index=True)

    # Restaura ordem desejada.
    ordem_gene = {g: i for i, g in enumerate(genes)}
    ordem_cluster = {c: i for i, c in enumerate(clusters)}
    df["gene_order"] = df["Gene"].map(ordem_gene)
    df["cluster_order"] = df["Cluster"].map(ordem_cluster)
    df = df.sort_values(["cluster_order", "gene_order"]).reset_index(drop=True)

    df.to_csv(
        PASTA_SAIDA_RESULTADOS / "Compact_dotplot_values.csv",
        index=False,
    )

    return df, genes, clusters


def plotar_dotplot_compacto(
    df: pd.DataFrame,
    painel: dict[str, list[str]],
    genes: list[str],
    clusters: list[str],
) -> None:
    # Figura maior e com área lateral dedicada às legendas, para ficar legível
    # mesmo quando o painel ocupar uma área relativamente pequena no pôster.
    fig, ax = plt.subplots(figsize=(15.8, 9.8))

    x = df["Gene"].map({g: i for i, g in enumerate(genes)}).to_numpy()
    y = df["Cluster"].map({c: i for i, c in enumerate(clusters)}).to_numpy()

    pct = df["Percent_expressing"].to_numpy(dtype=float)
    expr = df["Relative_mean_expression_0_1_per_gene"].to_numpy(dtype=float)

    # Área dos pontos. Mantém genes raros visíveis sem sugerir alta frequência.
    sizes = 12.0 + 2.20 * pct

    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=expr,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=1.0),
        edgecolors="0.35",
        linewidths=0.28,
    )

    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=55, ha="right", fontsize=11.8)

    ax.set_yticks(np.arange(len(clusters)))
    ax.set_yticklabels(
        [ROTULOS_CLUSTER_CURTOS.get(c, c) for c in clusters],
        fontsize=11.6,
    )
    ax.invert_yaxis()

    # Separadores + nomes dos blocos de genes.
    # Para evitar sobreposição no topo do dotplot, usamos rótulos com quebra
    # de linha nos grupos longos e um espaço vertical um pouco maior.
    rotulos_grupos = {
        "Homeostatic": "Homeostatic",
        "Activated / phagocytic / PVM": "Activated / phagocytic\n/ PVM",
        "Antigen / interferon": "Antigen /\ninterferon",
        "Inflammatory / metabolic / stress": "Inflammatory / metabolic\n/ stress",
        "QC": "QC",
    }

    inicio = 0
    for grupo, genes_grupo in painel.items():
        fim = inicio + len(genes_grupo)

        if inicio > 0:
            ax.axvline(inicio - 0.5, color="0.80", linewidth=1.0)

        centro = (inicio + fim - 1) / 2
        ax.text(
            centro,
            -1.45,
            rotulos_grupos.get(grupo, grupo),
            ha="center",
            va="bottom",
            fontsize=11.2,
            fontweight="bold",
            linespacing=1.05,
            clip_on=False,
        )
        inicio = fim

    ax.set_xlim(-0.65, len(genes) - 0.35)
    ax.set_ylim(len(clusters) - 0.45, -1.90)

    ax.grid(axis="y", color="0.90", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Reserva espaço à direita para colorbar + legenda de tamanho, evitando sobreposição.
    fig.subplots_adjust(
        left=0.215,
        right=0.835,
        top=0.95,
        bottom=0.13,
    )

    # Colorbar em eixo próprio, para não colidir com a legenda de tamanho.
    cax = fig.add_axes([0.855, 0.235, 0.022, 0.62])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(
        "Relative mean expression\nwithin each gene (0–1)",
        fontsize=11.2,
    )
    cbar.ax.tick_params(labelsize=10.2)

    # Legenda do tamanho em posição separada, abaixo da colorbar.
    valores_legenda = [25, 50, 75]
    handles = [
        plt.scatter(
            [],
            [],
            s=12.0 + 2.20 * v,
            facecolor="0.65",
            edgecolor="0.35",
            linewidth=0.45,
            label=f"{v}%",
        )
        for v in valores_legenda
    ]
    legend = fig.legend(
        handles=handles,
        title="Dot size\n(% cells expressing)",
        loc="upper left",
        bbox_to_anchor=(0.868, 0.225),
        frameon=False,
        fontsize=10.0,
        title_fontsize=10.8,
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.55,
        labelspacing=0.55,
    )
    try:
        legend._legend_box.align = "left"
    except Exception:
        pass

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    salvar_figura(fig, "Fig_Annotation_Compact_Dotplot")


# =============================================================================
# 8. LEGENDAS PRONTAS PARA O PÔSTER
# =============================================================================

def salvar_legendas(resumo_coorte: dict) -> None:
    texto = f"""POSTER FIGURE CAPTIONS — BIN1 / MICROGLIA / MILO

METHODS — Cohort definition
SEA-AD microglial dataset: {resumo_coorte['raw_cells']:,} cells from {resumo_coorte['raw_donors']} donors before clinical filtering. Donors classified as ADNC Low, Intermediate, or High were retained, while Not AD was excluded, yielding {resumo_coorte['post_filter_cells']:,} cells from {resumo_coorte['post_filter_donors']} donors (11 homozygous risk-allele carriers and 64 non-recessive individuals).

RESULTS — Annotated microglial UMAP
Final Leiden 0.50 UMAP of 32,657 cells from 75 donors, showing the 13 transcriptional states used for biological interpretation. Colors are identical to those used for the background states in the G2 overlay; the UMAP coordinates and cluster assignments are reused from the frozen final object and are not recalculated for this figure.

RESULTS — G2 overlay
G2 differential-abundance neighborhoods mapped onto the final annotated microglial UMAP. Background cells represent the 13 Leiden 0.50 states using the original frozen UMAP coordinates. Overlaid circles denote neighborhoods with SpatialFDR < 0.10 in G2 (genotype + age + sex + Braak); red indicates enrichment and blue indicates depletion in homozygous risk-allele carriers. Circle size reflects neighborhood size.

RESULTS — Compact annotation dotplot
Selected marker genes and transcriptional programs supporting the 13-state annotation. Dot size indicates the fraction of cells expressing each gene, while color represents mean log1p(CP10k) expression rescaled from 0 to 1 separately for each gene across clusters. The compact panel summarizes pre-existing annotation evidence; the final annotation also incorporated broader differential markers, microglial programs, SEA-AD reference labels, donor representation, and QC evidence.
"""

    caminho = PASTA_SAIDA_RESULTADOS / "Poster_figure_captions.txt"
    caminho.write_text(texto, encoding="utf-8")
    print(f"[OK] Legendas: {caminho}")


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("FIGURAS FINAIS PARA O PÔSTER — BIN1 / MICROGLIA / MILO")
    print("=" * 78)
    print("Este script é de VISUALIZAÇÃO/AUDITORIA e não refaz a inferência edgeR.")

    # 1) Coorte.
    resumo_coorte = auditar_coorte()

    # 2) Objeto final completo para UMAP/dotplot/Milo geométrico.
    print("\nCarregando objeto final em memória...")
    adata = sc.read_h5ad(ARQUIVO_FINAL)
    validar_objeto_final(adata)
    print("[OK] Objeto final validado.")

    # 3) UMAP final anotado em inglês.
    print("\n" + "=" * 78)
    print("[3/5] UMAP FINAL ANOTADO EM INGLÊS")
    print("=" * 78)
    plotar_umap_anotado_ingles(adata)

    # 4) Reconstrói SOMENTE a geometria Milo e exige o hash final.
    mdata = reconstruir_milo_para_visualizacao(adata)

    # 5) Lê G2 final, anexa coordenadas e gera overlay.
    print("\n" + "=" * 78)
    print("[4/5] G2 SOBREPOSTO AO UMAP ANOTADO")
    print("=" * 78)
    g2 = carregar_validar_g2(mdata)

    plotar_g2_sobre_umap(
        adata=mdata["rna"],
        g2=g2,
        incluir_legenda_estados=False,
        nome_base="Fig_G2_Overlay_Annotated_UMAP_COMPACT",
    )

    # Versão de apoio/estudo, com legenda completa dos estados.
    plotar_g2_sobre_umap(
        adata=mdata["rna"],
        g2=g2,
        incluir_legenda_estados=True,
        nome_base="Fig_G2_Overlay_Annotated_UMAP_FULL",
    )

    # 5) Dotplot compacto.
    print("\n" + "=" * 78)
    print("[5/5] DOTPLOT COMPACTO DA ANOTAÇÃO")
    print("=" * 78)

    painel = obter_painel_presente(mdata["rna"])
    df_dot, genes, clusters = calcular_metricas_dotplot(
        mdata["rna"],
        painel,
    )
    plotar_dotplot_compacto(
        df=df_dot,
        painel=painel,
        genes=genes,
        clusters=clusters,
    )

    # Legendas prontas.
    salvar_legendas(resumo_coorte)

    print("\n" + "=" * 78)
    print("CONCLUÍDO")
    print("=" * 78)
    print(f"Figuras : {PASTA_SAIDA_FIGURAS}")
    print(f"Auditoria: {PASTA_SAIDA_RESULTADOS}")
    print("\nArquivos principais para o banner:")
    print("  1) Fig_Methods_Cohort_Flow.pdf/png")
    print("  2) Fig_Annotated_Microglial_UMAP_English.pdf/png")
    print("  3) Fig_G2_Overlay_Annotated_UMAP_COMPACT.pdf/png")
    print("  4) Fig_Annotation_Compact_Dotplot.pdf/png")
    print("\nA versão FULL do overlay fica como material de apoio/estudo.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "!" * 78, file=sys.stderr)
        print("ERRO — execução interrompida para proteger a consistência do projeto.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("!" * 78, file=sys.stderr)
        raise