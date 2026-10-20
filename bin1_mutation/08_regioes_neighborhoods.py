#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_regioes_neighborhoods.py

Consolidação DESCRITIVA de neighborhoods Milo sobrepostas em regiões do grafo.

EXECUTAR NO AMBIENTE:
    conda activate milo_bin1
    python -u bin1_mutation/08_regioes_neighborhoods.py

IMPORTANTE:
- este script NÃO cria novos p-valores;
- NÃO substitui edgeR/SpatialFDR;
- NÃO recalcula Milo, PCA, KNN, Leiden ou UMAP;
- agrupa apenas neighborhoods já selecionadas pelos resultados finais,
  usando a sobreposição de suas memberships celulares.

Critério primário de conexão:
    overlap coefficient >= 0,50
onde:
    overlap coefficient = |A ∩ B| / min(|A|, |B|)

Interpretação: pelo menos metade da menor das duas neighborhoods é
compartilhada com a outra.

Sensibilidade:
    thresholds 0,30 e 0,70 também são calculados e exportados.

Conjuntos analisados:
1) 28 neighborhoods significativas simultaneamente em G0/G1/G2;
2) 13 neighborhoods significativas simultaneamente nas 6 rotas;
3) depleções homeostáticas significativas de G2, QC OK e não mistas.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "results" / "milo_final_edger410_reuniao"
EXPORT = BASE / "export"
FINAL = BASE / "final"
OUT = BASE / "regioes"
FIG = SCRIPT_DIR / "figures" / "milo_final_edger410_reuniao" / "regioes"

for p in [OUT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

ROTAS_PRINCIPAIS = [
    "RotaG0_BIN1",
    "RotaG1_BIN1_Idade_Sexo",
    "RotaG2_BIN1_Idade_Sexo_Braak",
]

ROTAS_6 = [
    "RotaG0_BIN1",
    "RotaG1_BIN1_Idade_Sexo",
    "RotaG2_BIN1_Idade_Sexo_Braak",
    "Rota0_BIN1_Idade",
    "Rota1_BIN1_Idade_ADNC",
    "Rota2_BIN1_Idade_Braak_Thal",
]

SFDR_LIMIAR = 0.10
THRESHOLDS = [0.30, 0.50, 0.70]
THRESHOLD_PRIMARIO = 0.50
HASH_ESPERADO = (
    "c06ce07f1ded7c9d96e932901a84e864"
    "f1487ec62ca952ee9152ec341aa08033"
)


# ============================================================
# 2. FUNÇÕES
# ============================================================


def carregar_rota(rota: str) -> pd.DataFrame:
    p = FINAL / f"{rota}_todos_neighborhoods.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    d["Neighborhood"] = d["Neighborhood"].astype(str)
    return d.set_index("Neighborhood", drop=False)


def sig_set(rota: str) -> set[str]:
    d = carregar_rota(rota)
    return set(
        d.loc[
            pd.to_numeric(d["SpatialFDR"], errors="coerce") < SFDR_LIMIAR,
            "Neighborhood",
        ].astype(str)
    )


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.upper().eq("TRUE")


def modo_texto(valores):
    vals = [str(x) for x in valores if pd.notna(x)]
    if not vals:
        return ""
    c = Counter(vals)
    return c.most_common(1)[0][0]


def salvar_figura(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def calcular_overlap_coef(membership_sub: sp.csc_matrix):
    # membership_sub: células x neighborhoods selecionadas, binária.
    m = membership_sub.astype(bool).astype(np.int32).tocsc()
    inter = (m.T @ m).toarray().astype(float)
    sizes = np.asarray(m.sum(axis=0)).ravel().astype(float)
    den = np.minimum.outer(sizes, sizes)
    coef = np.divide(
        inter,
        den,
        out=np.zeros_like(inter, dtype=float),
        where=den > 0,
    )
    np.fill_diagonal(coef, 1.0)
    return coef, inter, sizes


def construir_regioes(nome_set: str, neighborhoods: list[str], direcao_ref: dict[str, int]):
    idx_global = [nhood_to_pos[n] for n in neighborhoods]
    msub = membership[:, idx_global].tocsc()
    coef, inter, sizes = calcular_overlap_coef(msub)

    resultados_threshold = []

    for threshold in THRESHOLDS:
        adj = (coef >= threshold).astype(np.int8)
        np.fill_diagonal(adj, 0)

        # Não conecta neighborhoods com direções opostas.
        dirs = np.array([direcao_ref.get(n, 0) for n in neighborhoods], dtype=int)
        same_dir = dirs[:, None] == dirs[None, :]
        nonzero_dir = (dirs[:, None] != 0) & (dirs[None, :] != 0)
        adj = adj * same_dir * nonzero_dir

        graph = sp.csr_matrix(adj)
        n_comp, labels = connected_components(
            graph,
            directed=False,
            return_labels=True,
        )

        # Ordena componentes por tamanho para IDs mais intuitivos.
        comp_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
        comp_order = {old: i + 1 for i, old in enumerate(comp_sizes.index)}
        region_ids = np.array([comp_order[x] for x in labels], dtype=int)

        mapa = pd.DataFrame(
            {
                "Conjunto": nome_set,
                "Threshold_overlap": threshold,
                "Neighborhood": neighborhoods,
                "Regiao": [f"R{r:02d}" for r in region_ids],
                "Direcao": [
                    "Enriquecimento_Recessivo" if direcao_ref.get(n, 0) > 0
                    else "Deplecao_Recessivo" if direcao_ref.get(n, 0) < 0
                    else "Indefinida"
                    for n in neighborhoods
                ],
                "Nhood_size_membership": sizes.astype(int),
            }
        )

        mapa.to_csv(
            OUT / f"{nome_set}_threshold_{threshold:.2f}_mapeamento_neighborhood_regiao.csv",
            index=False,
        )

        # Arestas observadas acima do threshold.
        arestas = []
        for i in range(len(neighborhoods)):
            for j in range(i + 1, len(neighborhoods)):
                if adj[i, j] != 0:
                    arestas.append(
                        {
                            "Conjunto": nome_set,
                            "Threshold_overlap": threshold,
                            "Neighborhood_A": neighborhoods[i],
                            "Neighborhood_B": neighborhoods[j],
                            "Intersecao_celulas": int(inter[i, j]),
                            "Overlap_coefficient": float(coef[i, j]),
                        }
                    )
        pd.DataFrame(arestas).to_csv(
            OUT / f"{nome_set}_threshold_{threshold:.2f}_arestas.csv",
            index=False,
        )

        # Resumo das regiões.
        resumo_rows = []
        for regiao, g in mapa.groupby("Regiao", sort=False):
            ns = g["Neighborhood"].astype(str).tolist()
            pos = [neighborhoods.index(n) for n in ns]
            union_cells = np.asarray(msub[:, pos].sum(axis=1)).ravel() > 0

            # Coeficientes internos (sem diagonal).
            if len(pos) > 1:
                subcoef = coef[np.ix_(pos, pos)]
                tri = subcoef[np.triu_indices(len(pos), k=1)]
                med_overlap = float(np.median(tri)) if len(tri) else np.nan
                max_overlap = float(np.max(tri)) if len(tri) else np.nan
            else:
                med_overlap = np.nan
                max_overlap = np.nan

            row = {
                "Conjunto": nome_set,
                "Threshold_overlap": threshold,
                "Regiao": regiao,
                "N_neighborhoods": len(ns),
                "Neighborhoods": ";".join(ns),
                "Direcao": modo_texto(g["Direcao"]),
                "N_celulas_unicas_cobertas": int(union_cells.sum()),
                "Overlap_interno_mediano": med_overlap,
                "Overlap_interno_maximo": max_overlap,
            }

            # Metadados biológicos dominantes.
            meta = nhood_meta.reindex(ns)
            if "nhood_Leiden_0_5" in meta.columns:
                row["Leiden_dominante"] = modo_texto(meta["nhood_Leiden_0_5"])
            if "nhood_bio_por_Leiden_dominante" in meta.columns:
                row["Estado_biologico_dominante"] = modo_texto(
                    meta["nhood_bio_por_Leiden_dominante"]
                )
            if "nhood_QC_cluster" in meta.columns:
                row["Fracao_QC_OK"] = float(
                    (meta["nhood_QC_cluster"].astype(str) == "OK").mean()
                )
            if "nhood_mista" in meta.columns:
                row["Fracao_nhoods_mistas"] = float(bool_series(meta["nhood_mista"]).mean())

            # Estatísticas das rotas completas já finalizadas.
            for rota in ROTAS_6:
                d = route_cache[rota].reindex(ns)
                lfc = pd.to_numeric(d["logFC"], errors="coerce")
                sfdr = pd.to_numeric(d["SpatialFDR"], errors="coerce")
                row[f"{rota}_logFC_mediana"] = float(lfc.median())
                row[f"{rota}_logFC_min"] = float(lfc.min())
                row[f"{rota}_logFC_max"] = float(lfc.max())
                row[f"{rota}_menor_SpatialFDR"] = float(sfdr.min())
                row[f"{rota}_N_SpatialFDR_lt_0_1"] = int((sfdr < SFDR_LIMIAR).sum())

            # Se o LODO completo já estiver pronto, incorpora robustez regional.
            lodo_final = BASE / "lodo" / "complete" / "final"
            for rota in ROTAS_PRINCIPAIS:
                stab_file = lodo_final / f"Estabilidade_LODO_{rota}_todos_neighborhoods.csv"
                if stab_file.exists():
                    stab = pd.read_csv(stab_file)
                    stab["Neighborhood"] = stab["Neighborhood"].astype(str)
                    stab = stab.set_index("Neighborhood").reindex(ns)
                    row[f"{rota}_LODO_prop_mesma_direcao_mediana"] = float(
                        pd.to_numeric(stab["Prop_mesma_direcao"], errors="coerce").median()
                    )
                    row[f"{rota}_LODO_prop_SFDR_lt_0_1_mediana"] = float(
                        pd.to_numeric(stab["Prop_SpatialFDR_lt_0_1"], errors="coerce").median()
                    )

            resumo_rows.append(row)

        resumo = pd.DataFrame(resumo_rows).sort_values(
            ["N_neighborhoods", "N_celulas_unicas_cobertas"],
            ascending=[False, False],
        )
        resumo.to_csv(
            OUT / f"{nome_set}_threshold_{threshold:.2f}_resumo_regioes.csv",
            index=False,
        )

        resultados_threshold.append(
            {
                "Conjunto": nome_set,
                "Threshold_overlap": threshold,
                "N_neighborhoods": len(neighborhoods),
                "N_regioes": int(n_comp),
                "Maior_regiao_N_neighborhoods": int(comp_sizes.iloc[0]),
                "N_regioes_com_mais_de_1_nhood": int((comp_sizes > 1).sum()),
            }
        )

        if threshold == THRESHOLD_PRIMARIO:
            # Heatmap de sobreposição.
            fig, ax = plt.subplots(figsize=(max(6, len(neighborhoods) * 0.28), max(5, len(neighborhoods) * 0.25)))
            im = ax.imshow(coef, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(np.arange(len(neighborhoods)))
            ax.set_xticklabels(neighborhoods, rotation=90, fontsize=7)
            ax.set_yticks(np.arange(len(neighborhoods)))
            ax.set_yticklabels(neighborhoods, fontsize=7)
            ax.set_title(f"{nome_set}: overlap coefficient entre neighborhoods")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Overlap coefficient")
            salvar_figura(fig, FIG / f"{nome_set}_overlap_heatmap_threshold_0.50.png")

            # Tamanho das regiões.
            counts = mapa["Regiao"].value_counts().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, max(4, len(counts) * 0.28)))
            ax.barh(np.arange(len(counts)), counts.to_numpy())
            ax.set_yticks(np.arange(len(counts)))
            ax.set_yticklabels(counts.index)
            ax.invert_yaxis()
            ax.set_xlabel("Número de neighborhoods")
            ax.set_title(f"{nome_set}: regiões por sobreposição (threshold 0,50)")
            salvar_figura(fig, FIG / f"{nome_set}_tamanho_regioes_threshold_0.50.png")

    return resultados_threshold


# ============================================================
# 3. AUDITORIA DA ENTRADA
# ============================================================

status_file = BASE / "auditoria" / "Execucao_final_status.json"
if not status_file.exists():
    raise FileNotFoundError(status_file)

status = json.loads(status_file.read_text(encoding="utf-8"))
if status.get("SHA256_neighborhoods") != HASH_ESPERADO:
    raise RuntimeError("SHA das neighborhoods não corresponde ao congelado.")
if int(status.get("N_neighborhoods", -1)) != 2201:
    raise RuntimeError("Esperava 2201 neighborhoods.")

membership_file = EXPORT / "milo_membership_cell_x_neighborhood.npz"
ids_file = EXPORT / "milo_neighborhood_ids.txt"
meta_file = EXPORT / "milo_neighborhood_metadata.tsv"

for p in [membership_file, ids_file, meta_file]:
    if not p.exists():
        raise FileNotFoundError(p)

membership = sp.load_npz(membership_file).tocsr()
nhood_ids = pd.read_csv(ids_file, header=None, dtype=str)[0].astype(str).tolist()

if membership.shape[1] != len(nhood_ids):
    raise RuntimeError("Membership e lista de IDs têm números de neighborhoods diferentes.")
if len(nhood_ids) != 2201:
    raise RuntimeError(f"Esperava 2201 IDs; encontrei {len(nhood_ids)}.")

nhood_to_pos = {n: i for i, n in enumerate(nhood_ids)}

nhood_meta = pd.read_csv(meta_file, sep="\t", index_col=0)
nhood_meta.index = nhood_meta.index.astype(str)

route_cache = {r: carregar_rota(r) for r in ROTAS_6}

# ============================================================
# 4. CONJUNTOS DE INTERESSE
# ============================================================

rec28 = set.intersection(*(sig_set(r) for r in ROTAS_PRINCIPAIS))
rec13 = set.intersection(*(sig_set(r) for r in ROTAS_6))

if len(rec28) != 28:
    raise RuntimeError(f"Esperava 28 recorrentes principais; encontrei {len(rec28)}.")
if len(rec13) != 13:
    raise RuntimeError(f"Esperava 13 no núcleo das 6 rotas; encontrei {len(rec13)}.")

# Direção de referência: mediana do logFC entre as rotas relevantes.
def direcao_mediana(ns: set[str], rotas: list[str]) -> dict[str, int]:
    out = {}
    for n in ns:
        vals = []
        for r in rotas:
            v = pd.to_numeric(
                pd.Series([route_cache[r].loc[n, "logFC"]]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(v):
                vals.append(v)
        med = np.median(vals) if vals else 0.0
        out[n] = int(np.sign(med))
    return out

# Depleções homeostáticas de G2 com QC OK e não mistas.
g2 = route_cache["RotaG2_BIN1_Idade_Sexo_Braak"].copy()
g2_mista = bool_series(g2["nhood_mista"]) if "nhood_mista" in g2.columns else pd.Series(False, index=g2.index)
g2_estado = g2["nhood_bio_por_Leiden_dominante"].astype(str)
g2_qc = g2["nhood_QC_cluster"].astype(str)

dep_g2 = set(
    g2.loc[
        (pd.to_numeric(g2["SpatialFDR"], errors="coerce") < SFDR_LIMIAR)
        & (pd.to_numeric(g2["logFC"], errors="coerce") < 0)
        & (~g2_mista)
        & (g2_qc == "OK")
        & g2_estado.str.contains("homeost", case=False, na=False),
        "Neighborhood",
    ].astype(str)
)

conjuntos = {
    "Recorrentes_28_G0_G1_G2": (
        sorted(rec28, key=lambda x: int(x) if x.isdigit() else x),
        direcao_mediana(rec28, ROTAS_PRINCIPAIS),
    ),
    "Nucleo_13_6_rotas": (
        sorted(rec13, key=lambda x: int(x) if x.isdigit() else x),
        direcao_mediana(rec13, ROTAS_6),
    ),
    "Deplecoes_homeostaticas_G2": (
        sorted(dep_g2, key=lambda x: int(x) if x.isdigit() else x),
        {n: -1 for n in dep_g2},
    ),
}

print("=" * 78)
print("AGRUPAMENTO DESCRITIVO DE NEIGHBORHOODS EM REGIÕES")
print("=" * 78)
print(f"Membership: {membership.shape[0]} células x {membership.shape[1]} neighborhoods")
print(f"Recorrentes G0/G1/G2: {len(rec28)}")
print(f"Núcleo 6 rotas: {len(rec13)}")
print(f"Depleções homeostáticas G2 selecionadas: {len(dep_g2)}")

# ============================================================
# 5. AGRUPAMENTO
# ============================================================

sumarios = []
for nome_set, (ns, dirs) in conjuntos.items():
    if not ns:
        print(f"[AVISO] {nome_set}: conjunto vazio; pulando.")
        continue

    faltantes = [n for n in ns if n not in nhood_to_pos]
    if faltantes:
        raise RuntimeError(f"Neighborhoods ausentes na membership ({nome_set}): {faltantes}")

    print(f"\n[{nome_set}] N={len(ns)}")
    sumarios.extend(construir_regioes(nome_set, ns, dirs))

pd.DataFrame(sumarios).to_csv(
    OUT / "Resumo_sensibilidade_thresholds_regioes.csv",
    index=False,
)

status_out = {
    "Concluido": True,
    "SHA256_neighborhoods": HASH_ESPERADO,
    "N_neighborhoods_total": 2201,
    "Membership_shape": list(membership.shape),
    "Overlap_metric": "intersection / min(size_A, size_B)",
    "Threshold_primario": THRESHOLD_PRIMARIO,
    "Thresholds_sensibilidade": THRESHOLDS,
    "N_recorrentes_G0_G1_G2": len(rec28),
    "N_nucleo_6_rotas": len(rec13),
    "N_deplecoes_homeostaticas_G2": len(dep_g2),
    "Inferencia_nova": False,
    "Observacao": (
        "Agrupamento descritivo. edgeR/SpatialFDR permanecem como inferência; "
        "regiões não recebem novos p-valores."
    ),
}

(OUT / "Execucao_regioes_status.json").write_text(
    json.dumps(status_out, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\n" + "=" * 78)
print("AGRUPAMENTO DE REGIÕES CONCLUÍDO")
print(f"Resultados: {OUT}")
print(f"Figuras: {FIG}")
print("=" * 78)