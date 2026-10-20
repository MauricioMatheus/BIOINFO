#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_robustez_lodo_finalize.py

Finalização da análise leave-one-donor-out (LODO) do projeto BIN1.

EXECUTAR NO AMBIENTE:
    conda activate milo_bin1

MODOS:
    python -u bin1_mutation/07_robustez_lodo_finalize.py critical
    python -u bin1_mutation/07_robustez_lodo_finalize.py complete

O script:
- lê os resultados edgeR gerados pelo 06_robustez_lodo_edger410.R;
- recalcula SpatialFDR usando exatamente a regra do Pertpy 0.10.0;
- mantém a geometria das 2.201 neighborhoods congelada;
- compara cada cenário LODO ao modelo completo já finalizado;
- quantifica retenção de significância, concordância de direção e
  influência de cada doador;
- produz tabelas específicas para as 28 neighborhoods recorrentes
  nas três rotas principais e para o núcleo de 13 recorrente nas seis rotas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "results" / "milo_final_edger410_reuniao"
EXPORT = BASE / "export"
BASELINE_FINAL = BASE / "final"
FIG_BASE = SCRIPT_DIR / "figures" / "milo_final_edger410_reuniao"

ROTAS = [
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
HASH_ESPERADO = (
    "c06ce07f1ded7c9d96e932901a84e864"
    "f1487ec62ca952ee9152ec341aa08033"
)


# ============================================================
# 2. FUNÇÕES
# ============================================================


def spatial_fdr_pertpy_0100(pvalues, kth_distance):
    """Reprodução da função Milo._graph_spatial_fdr do Pertpy 0.10.0."""
    p = pd.Series(pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float))
    d = pd.Series(pd.to_numeric(kth_distance, errors="coerce").to_numpy(dtype=float))

    w = 1.0 / d
    w[np.isinf(w)] = 0.0

    keep = (~p.isna()) & (~w.isna())
    p_keep = p.loc[keep]
    w_keep = w.loc[keep]

    order = np.argsort(p_keep.to_numpy())
    p_sorted = p_keep.iloc[order].to_numpy(dtype=float)
    w_sorted = w_keep.iloc[order].to_numpy(dtype=float)

    if len(w_sorted) == 0 or w_sorted.sum() <= 0:
        return np.full(len(p), np.nan, dtype=float)

    cumul_w = np.cumsum(w_sorted)
    if np.any(cumul_w <= 0):
        raise RuntimeError("cumsum dos pesos contém zero antes do fim.")

    bruto = w_sorted.sum() * p_sorted / cumul_w
    monotono = np.minimum.accumulate(bruto[::-1])[::-1]
    monotono = np.minimum(monotono, 1.0)

    ajustado_keep = np.empty(len(order), dtype=float)
    ajustado_keep[order] = monotono

    saida = np.full(len(p), np.nan, dtype=float)
    idx_keep = np.flatnonzero(keep.to_numpy())
    saida[idx_keep] = ajustado_keep
    return saida


def normalizar_neighborhood(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Neighborhood"] = out["Neighborhood"].astype(str)
    return out


def carregar_baseline(rota: str) -> pd.DataFrame:
    p = BASELINE_FINAL / f"{rota}_todos_neighborhoods.csv"
    if not p.exists():
        raise FileNotFoundError(f"Baseline ausente: {p}")
    d = normalizar_neighborhood(pd.read_csv(p))
    d = d.set_index("Neighborhood", drop=False)
    return d


def conjunto_recorrente(rotas: list[str]) -> set[str]:
    conjuntos = []
    for rota in rotas:
        d = carregar_baseline(rota)
        conjuntos.append(
            set(
                d.loc[
                    pd.to_numeric(d["SpatialFDR"], errors="coerce") < SFDR_LIMIAR,
                    "Neighborhood",
                ].astype(str)
            )
        )
    return set.intersection(*conjuntos)


def salvar_figura(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def str_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.upper().eq("TRUE")


# ============================================================
# 3. ARGUMENTO
# ============================================================

if len(sys.argv) < 2:
    raise SystemExit("Informe o modo: critical ou complete.")

MODO = sys.argv[1].lower()
if MODO not in {"critical", "complete"}:
    raise SystemExit("Modo inválido. Use: critical ou complete.")

LODO = BASE / "lodo" / MODO
EDGER = LODO / "edger"
FINAL = LODO / "final"
AUDIT = LODO / "auditoria"
FIG = FIG_BASE / "lodo" / MODO

for p in [FINAL, AUDIT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

print("=" * 78)
print(f"FINALIZAÇÃO LODO — {MODO.upper()}")
print("=" * 78)


# ============================================================
# 4. AUDITORIA DO OBJETO CONGELADO
# ============================================================

status_export_file = BASE / "auditoria" / "Etapa1_export_status.json"
if not status_export_file.exists():
    raise FileNotFoundError(status_export_file)

status_export = json.loads(status_export_file.read_text(encoding="utf-8"))

if status_export.get("SHA256_neighborhoods") != HASH_ESPERADO:
    raise RuntimeError("SHA das neighborhoods congeladas não corresponde ao esperado.")

if int(status_export.get("N_nhoods", -1)) != 2201:
    raise RuntimeError("Número de neighborhoods congeladas inesperado.")

nhood_file = EXPORT / "milo_neighborhood_metadata.tsv"
if not nhood_file.exists():
    raise FileNotFoundError(nhood_file)

nhood = pd.read_csv(nhood_file, sep="\t", index_col=0)
nhood.index = nhood.index.astype(str)
nhood.index.name = "Neighborhood"

if len(nhood) != 2201:
    raise RuntimeError(f"Metadata de neighborhoods com {len(nhood)} linhas, esperado 2201.")

if "kth_distance" not in nhood.columns:
    raise KeyError("kth_distance ausente no metadata de neighborhoods.")

# ============================================================
# 5. STATUS edgeR E CONJUNTOS DE REFERÊNCIA
# ============================================================

status_file = LODO / "status_edgeR.csv"
if not status_file.exists():
    raise FileNotFoundError(
        f"{status_file}\nRode primeiro o script 06 no modo {MODO}."
    )

status = pd.read_csv(status_file)
ok = status["Status"].astype(str).str.startswith("OK")

if not ok.all():
    falhas = status.loc[~ok]
    falhas.to_csv(AUDIT / "Falhas_edgeR_LODO.csv", index=False)
    print("\n[ATENÇÃO] Há ajustes LODO que não terminaram com status OK:")
    print(falhas.to_string(index=False))

status_ok = status.loc[ok].copy()

rec28 = conjunto_recorrente(ROTAS)
rec13 = conjunto_recorrente(ROTAS_6)

print(f"Neighborhoods recorrentes G0/G1/G2 no baseline: {len(rec28)}")
print(f"Neighborhoods recorrentes nas 6 rotas no baseline: {len(rec13)}")

if len(rec28) != 28:
    raise RuntimeError(f"Esperava 28 recorrentes G0/G1/G2, encontrei {len(rec28)}.")
if len(rec13) != 13:
    raise RuntimeError(f"Esperava 13 recorrentes nas 6 rotas, encontrei {len(rec13)}.")

# ============================================================
# 6. FINALIZA CADA RESULTADO LODO E COMPARA AO BASELINE
# ============================================================

resumo_cenarios = []
retencao_rec28 = []
retencao_rec13 = []
resultados_cache: dict[tuple[str, str], pd.DataFrame] = {}

for _, srow in status_ok.iterrows():
    cenario = str(srow["Cenario"])
    rota = str(srow["Rota"])

    if rota not in ROTAS:
        continue

    raw_file = EDGER / cenario / f"{rota}_edgeR_raw.csv"
    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    raw = normalizar_neighborhood(pd.read_csv(raw_file))
    if len(raw) != 2201:
        raise RuntimeError(f"{raw_file}: esperado 2201 linhas, encontrado {len(raw)}.")

    raw = raw.set_index("Neighborhood", drop=False).reindex(nhood.index)
    raw["Neighborhood"] = raw.index.astype(str)
    raw["Testavel"] = str_bool(raw["Testavel"])

    raw["SpatialFDR"] = spatial_fdr_pertpy_0100(
        raw["PValue"],
        nhood["kth_distance"],
    )
    raw["Significativa_SpatialFDR_0_1"] = raw["SpatialFDR"] < SFDR_LIMIAR
    raw["abs_logFC"] = pd.to_numeric(raw["logFC"], errors="coerce").abs()

    # Junta metadados biológicos sem sobrescrever colunas do edgeR.
    for col in nhood.columns:
        if col not in raw.columns:
            raw[col] = nhood[col]

    scenario_final_dir = FINAL / cenario
    scenario_final_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(
        scenario_final_dir / f"{rota}_LODO_final.csv",
        index=False,
    )

    resultados_cache[(cenario, rota)] = raw.copy()

    baseline = carregar_baseline(rota).reindex(nhood.index)
    baseline_sig = pd.to_numeric(baseline["SpatialFDR"], errors="coerce") < SFDR_LIMIAR
    lodo_sig = raw["Significativa_SpatialFDR_0_1"].fillna(False)
    testavel = raw["Testavel"].fillna(False)

    base_logfc = pd.to_numeric(baseline["logFC"], errors="coerce")
    lodo_logfc = pd.to_numeric(raw["logFC"], errors="coerce")
    direcao_concordante = np.sign(base_logfc) == np.sign(lodo_logfc)
    direcao_concordante = pd.Series(direcao_concordante, index=nhood.index) & testavel

    n_baseline_sig = int(baseline_sig.sum())
    n_lodo_sig = int(lodo_sig.sum())
    n_retidas = int((baseline_sig & lodo_sig).sum())
    n_perdidas = int((baseline_sig & ~lodo_sig).sum())
    n_ganhas = int((~baseline_sig & lodo_sig).sum())

    resumo_cenarios.append(
        {
            "Cenario": cenario,
            "Removidos": str(srow["Removidos"]),
            "Rota": rota,
            "N_doadores": int(srow["N_doadores"]),
            "N_testaveis": int(testavel.sum()),
            "N_baseline_sig": n_baseline_sig,
            "N_LODO_sig": n_lodo_sig,
            "N_baseline_sig_retidas": n_retidas,
            "N_baseline_sig_perdidas": n_perdidas,
            "N_novas_sig": n_ganhas,
            "Fracao_baseline_sig_retida": n_retidas / n_baseline_sig if n_baseline_sig else np.nan,
            "Fracao_direcao_concordante_baseline_sig": float(
                direcao_concordante.loc[baseline_sig].mean()
            ) if n_baseline_sig else np.nan,
            "Mediana_abs_delta_logFC_baseline_sig": float(
                (lodo_logfc.loc[baseline_sig] - base_logfc.loc[baseline_sig]).abs().median()
            ) if n_baseline_sig else np.nan,
        }
    )

    for nome_set, conjunto, coletor in [
        ("Recorrentes_G0_G1_G2", rec28, retencao_rec28),
        ("Nucleo_6_rotas", rec13, retencao_rec13),
    ]:
        idx = pd.Index(sorted(conjunto)).intersection(nhood.index)
        coletor.append(
            {
                "Cenario": cenario,
                "Removidos": str(srow["Removidos"]),
                "Rota": rota,
                "Conjunto": nome_set,
                "N_total_conjunto": len(idx),
                "N_testaveis": int(testavel.loc[idx].sum()),
                "N_SpatialFDR_lt_0_1": int(lodo_sig.loc[idx].sum()),
                "Fracao_SpatialFDR_lt_0_1": float(lodo_sig.loc[idx].mean()),
                "N_mesma_direcao": int(direcao_concordante.loc[idx].sum()),
                "Fracao_mesma_direcao": float(direcao_concordante.loc[idx].mean()),
                "N_sig_e_mesma_direcao": int(
                    (lodo_sig.loc[idx] & direcao_concordante.loc[idx]).sum()
                ),
                "Fracao_sig_e_mesma_direcao": float(
                    (lodo_sig.loc[idx] & direcao_concordante.loc[idx]).mean()
                ),
            }
        )

resumo = pd.DataFrame(resumo_cenarios)
ret28 = pd.DataFrame(retencao_rec28)
ret13 = pd.DataFrame(retencao_rec13)

resumo.to_csv(FINAL / "Resumo_LODO_cenarios_rotas.csv", index=False)
ret28.to_csv(FINAL / "Retencao_28_recorrentes_G0_G1_G2.csv", index=False)
ret13.to_csv(FINAL / "Retencao_13_nucleo_6_rotas.csv", index=False)

# ============================================================
# 7. ESTABILIDADE POR NEIGHBORHOOD
# ============================================================

for rota in ROTAS:
    baseline = carregar_baseline(rota).reindex(nhood.index)
    base_logfc = pd.to_numeric(baseline["logFC"], errors="coerce")
    base_sfdr = pd.to_numeric(baseline["SpatialFDR"], errors="coerce")
    base_sig = base_sfdr < SFDR_LIMIAR

    rows = []
    cenarios_rota = sorted(
        c for (c, r) in resultados_cache.keys() if r == rota
    )

    for n in nhood.index:
        vals_logfc = []
        vals_sfdr = []
        vals_test = []
        vals_sig = []
        vals_dir = []
        vals_sig_dir = []
        donor_delta = []

        for c in cenarios_rota:
            d = resultados_cache[(c, rota)].loc[n]
            test = bool(d["Testavel"])
            lf = pd.to_numeric(pd.Series([d["logFC"]]), errors="coerce").iloc[0]
            sf = pd.to_numeric(pd.Series([d["SpatialFDR"]]), errors="coerce").iloc[0]

            sig = bool(np.isfinite(sf) and sf < SFDR_LIMIAR)
            same_dir = bool(
                test
                and np.isfinite(lf)
                and np.isfinite(base_logfc.loc[n])
                and np.sign(lf) == np.sign(base_logfc.loc[n])
            )

            vals_test.append(test)
            vals_logfc.append(lf)
            vals_sfdr.append(sf)
            vals_sig.append(sig)
            vals_dir.append(same_dir)
            vals_sig_dir.append(sig and same_dir)

            if test and np.isfinite(lf) and np.isfinite(base_logfc.loc[n]):
                donor_delta.append((c, abs(lf - base_logfc.loc[n])))

        test_arr = np.array(vals_test, dtype=bool)
        n_test = int(test_arr.sum())

        log_arr = np.array(vals_logfc, dtype=float)
        sfdr_arr = np.array(vals_sfdr, dtype=float)

        influente = ""
        maior_delta = np.nan
        if donor_delta:
            influente, maior_delta = max(donor_delta, key=lambda x: x[1])

        rows.append(
            {
                "Neighborhood": n,
                "Rota": rota,
                "Baseline_logFC": base_logfc.loc[n],
                "Baseline_SpatialFDR": base_sfdr.loc[n],
                "Baseline_significativa": bool(base_sig.loc[n]),
                "N_LODO_total": len(cenarios_rota),
                "N_LODO_testavel": n_test,
                "Prop_testavel": n_test / len(cenarios_rota) if cenarios_rota else np.nan,
                "N_mesma_direcao": int(np.sum(vals_dir)),
                "Prop_mesma_direcao": np.sum(vals_dir) / n_test if n_test else np.nan,
                "N_SpatialFDR_lt_0_1": int(np.sum(vals_sig)),
                "Prop_SpatialFDR_lt_0_1": np.sum(vals_sig) / n_test if n_test else np.nan,
                "N_sig_e_mesma_direcao": int(np.sum(vals_sig_dir)),
                "Prop_sig_e_mesma_direcao": np.sum(vals_sig_dir) / n_test if n_test else np.nan,
                "LODO_logFC_min": np.nanmin(log_arr) if np.isfinite(log_arr).any() else np.nan,
                "LODO_logFC_mediana": np.nanmedian(log_arr) if np.isfinite(log_arr).any() else np.nan,
                "LODO_logFC_max": np.nanmax(log_arr) if np.isfinite(log_arr).any() else np.nan,
                "LODO_SpatialFDR_min": np.nanmin(sfdr_arr) if np.isfinite(sfdr_arr).any() else np.nan,
                "LODO_SpatialFDR_mediana": np.nanmedian(sfdr_arr) if np.isfinite(sfdr_arr).any() else np.nan,
                "LODO_SpatialFDR_max": np.nanmax(sfdr_arr) if np.isfinite(sfdr_arr).any() else np.nan,
                "Cenario_maior_delta_logFC": influente,
                "Maior_abs_delta_logFC": maior_delta,
            }
        )

    stab = pd.DataFrame(rows)

    for col in [
        "nhood_Leiden_0_5",
        "nhood_bio_por_Leiden_dominante",
        "nhood_QC_cluster",
        "nhood_mista",
        "Nhood_size",
    ]:
        if col in nhood.columns:
            stab[col] = stab["Neighborhood"].map(nhood[col])

    stab.to_csv(FINAL / f"Estabilidade_LODO_{rota}_todos_neighborhoods.csv", index=False)

    stab[stab["Neighborhood"].isin(rec28)].to_csv(
        FINAL / f"Estabilidade_LODO_{rota}_28_recorrentes_G0_G1_G2.csv",
        index=False,
    )
    stab[stab["Neighborhood"].isin(rec13)].to_csv(
        FINAL / f"Estabilidade_LODO_{rota}_13_nucleo_6_rotas.csv",
        index=False,
    )

# ============================================================
# 8. FIGURAS
# ============================================================

# 8.1 Número de significativas por cenário e rota
if not resumo.empty:
    pivot = resumo.pivot(index="Cenario", columns="Rota", values="N_LODO_sig")
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.22), 5.5))
    x = np.arange(len(pivot))
    width = 0.25
    for i, rota in enumerate(ROTAS):
        vals = pivot[rota].to_numpy() if rota in pivot.columns else np.zeros(len(pivot))
        ax.bar(x + (i - 1) * width, vals, width=width, label=rota)
    ax.set_ylabel("Neighborhoods com SpatialFDR < 0,10")
    ax.set_title(f"LODO {MODO}: número de neighborhoods significativas")
    ax.set_xticks(x)
    if MODO == "critical":
        ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    else:
        # 75 labels seriam ilegíveis; o detalhe completo permanece no CSV.
        ax.set_xticklabels([str(i + 1) if i % 5 == 0 else "" for i in range(len(pivot))])
        ax.set_xlabel("Cenários LODO (ordem alfabética; rótulo a cada 5)")
    ax.legend(fontsize=8)
    salvar_figura(fig, FIG / "LODO_numero_significativas_por_cenario.png")

# 8.2 Retenção das 28 recorrentes: matriz cenário x rota
if not ret28.empty:
    mat = ret28.pivot(
        index="Cenario",
        columns="Rota",
        values="Fracao_sig_e_mesma_direcao",
    ).reindex(columns=ROTAS)

    fig, ax = plt.subplots(figsize=(8, max(3.8, len(mat) * 0.22)))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    if MODO == "critical":
        ax.set_yticklabels(mat.index)
    else:
        ax.set_yticklabels([x if i % 5 == 0 else "" for i, x in enumerate(mat.index)])
    ax.set_title("Retenção das 28 recorrentes: significância + mesma direção")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fração das 28 neighborhoods")
    salvar_figura(fig, FIG / "LODO_retencao_28_recorrentes_heatmap.png")

# 8.3 Complete: ranking dos doadores mais influentes por perda de baseline significativo
if MODO == "complete" and not resumo.empty:
    influencia = resumo.copy()
    influencia["N_perdidas"] = influencia["N_baseline_sig_perdidas"]
    influencia.to_csv(FINAL / "Influencia_doadores_por_rota.csv", index=False)

    for rota in ROTAS:
        d = influencia[influencia["Rota"] == rota].copy()
        d = d.sort_values(
            ["N_perdidas", "Mediana_abs_delta_logFC_baseline_sig"],
            ascending=[False, False],
        ).head(15)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(np.arange(len(d)), d["N_perdidas"].to_numpy())
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(d["Removidos"].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel("Baseline significativas que deixam SpatialFDR < 0,10")
        ax.set_title(f"Top 15 exclusões mais influentes — {rota}")
        salvar_figura(fig, FIG / f"{rota}_Top15_doadores_influentes.png")

# ============================================================
# 9. STATUS FINAL
# ============================================================

status_final = {
    "Concluido": True,
    "Modo": MODO,
    "N_cenarios_edgeR_OK": int(status_ok["Cenario"].nunique()),
    "N_ajustes_edgeR_OK": int(len(status_ok)),
    "N_rotas": 3,
    "N_neighborhoods": 2201,
    "SHA256_neighborhoods": HASH_ESPERADO,
    "SpatialFDR_limiar": SFDR_LIMIAR,
    "SpatialFDR_implementacao": "Reprodução de Pertpy 0.10.0 Milo._graph_spatial_fdr",
    "Recorrentes_G0_G1_G2_baseline": len(rec28),
    "Nucleo_6_rotas_baseline": len(rec13),
}

(AUDIT / "Execucao_LODO_status.json").write_text(
    json.dumps(status_final, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nResumo dos cenários:")
print(resumo.to_string(index=False))
print("\n" + "=" * 78)
print("FINALIZAÇÃO LODO CONCLUÍDA")
print(f"Resultados: {FINAL}")
print(f"Figuras: {FIG}")
print("=" * 78)