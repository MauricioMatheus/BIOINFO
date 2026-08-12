import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# 04_milo_annotation.py
#
# ANOTAÇÃO BIOLÓGICA DOS CLUSTERS LEIDEN 0.45
#
# Estratégia:
#   1) identificar marcadores dos 10 clusters;
#   2) avaliar programas gênicos microgliais;
#   3) comparar com o Supertype original do SEA-AD;
#   4) checar sinais de possível contaminação;
#   5) verificar representatividade entre doadores;
#   6) gerar uma tabela integrada para revisão manual;
#   7) somente depois, preencher ANOTACAO_FINAL.
#
# IMPORTANTE:
# - O script NÃO usa BIN1/Milo para decidir a identidade celular.
# - Na primeira execução, deixe ANOTACAO_FINAL = {}.
# - O arquivo original .h5ad nunca é sobrescrito.
# =========================================================


# =========================================================
# CONFIGURAÇÃO
# =========================================================

RESOLUCAO = "0.45"
CLUSTER_ORIGINAL = f"leiden_{RESOLUCAO}"
CLUSTER_ANALISE = "Leiden_045_Anotacao"

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
sc.settings.n_jobs = n_cpus

print(f"Usando {n_cpus} CPUs.")
print(f"Resolução alvo: Leiden {RESOLUCAO}")


# =========================================================
# ANOTAÇÃO FINAL
# =========================================================
#
# PRIMEIRA EXECUÇÃO:
#     ANOTACAO_FINAL = {}
#
# Depois de revisar as evidências, preencher TODOS os clusters:
#
# ANOTACAO_FINAL = {
#     "0": "...",
#     "1": "...",
#     ...
#     "9": "...",
# }
# =========================================================

ANOTACAO_FINAL = {}


# =========================================================
# PAINÉIS DE GENES
# =========================================================
#
# Estes conjuntos são usados como evidência de programas celulares.
# Um único gene NÃO define a identidade do cluster.
# =========================================================

PROGRAMAS_MICROGLIAIS = {
    "Homeostase": [
        "P2RY12", "P2RY13", "TMEM119", "CX3CR1", "GPR34",
        "SALL1", "HEXB", "CSF1R", "SIGLEC10",
    ],
    "Doenca_DAM_like": [
        "APOE", "TREM2", "LPL", "CST7", "GPNMB", "SPP1",
        "CD9", "ITGAX", "CTSB", "CTSD", "APOC1", "LGALS3",
    ],
    "Fagocitose_Lipidios": [
        "APOC1", "LPL", "FABP5", "LIPA", "CTSB", "CTSD",
        "FCER1G", "TYROBP", "AXL", "MSR1",
    ],
    "Apresentacao_Antigeno": [
        "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
        "CD74", "B2M", "CIITA",
    ],
    "Interferon": [
        "ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1",
        "OAS1", "OAS2", "STAT1", "IRF7",
    ],
    "Inflamacao": [
        "IL1B", "TNF", "NFKBIA", "PTGS2",
        "CCL2", "CXCL8", "CCL3", "CCL4",
    ],
    "Proliferacao": [
        "MKI67", "TOP2A", "STMN1", "TYMS", "UBE2C", "CENPF",
    ],
    "Resposta_Imediata_Estresse": [
        "FOS", "JUN", "JUNB", "DUSP1", "ATF3", "HSPA1A", "HSPA1B",
    ],
}

PAINEIS_CONTAMINACAO = {
    "Neuronal": [
        "SNAP25", "SYT1", "RBFOX3", "SLC17A7",
        "GAD1", "GAD2", "CAMK2A",
    ],
    "Oligodendroglial": [
        "MBP", "PLP1", "MOG", "MAG", "MOBP", "OLIG1", "OLIG2",
    ],
    "Astrocitario": [
        "GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1", "SOX9",
    ],
    "Endotelial": [
        "CLDN5", "FLT1", "KDR", "PECAM1", "VWF", "RAMP2",
    ],
    "Linfocitario": [
        "CD3D", "CD3E", "TRAC", "CD247", "MS4A1", "CD79A", "NKG7",
    ],
}


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def localizar_h5ad():
    candidatos = [
        Path("adata_limpo_anotado.h5ad"),
        Path("bin1_mutation/adata_limpo_anotado.h5ad"),
    ]

    for caminho in candidatos:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        "Não encontrei 'adata_limpo_anotado.h5ad'. "
        "Execute o script no diretório do projeto ou ajuste o caminho."
    )


def remover_genes_ruido(adata):
    genes_ruido = [
        gene
        for gene in adata.var_names
        if (
            gene == "MALAT1"
            or gene.startswith("MT-")
            or gene.startswith("RPS")
            or gene.startswith("RPL")
        )
    ]

    if genes_ruido:
        print(
            f"Removendo {len(genes_ruido)} genes de ruído "
            "da matriz de anotação."
        )
        return adata[:, ~adata.var_names.isin(genes_ruido)].copy()

    return adata.copy()


def genes_presentes(adata, genes):
    universo = set(adata.var_names)
    return [gene for gene in genes if gene in universo]


def garantir_cluster_045(adata):
    if CLUSTER_ORIGINAL in adata.obs.columns:
        origem = CLUSTER_ORIGINAL

    elif "Estado_Microglial" in adata.obs.columns:
        origem = "Estado_Microglial"

        print(
            "AVISO: 'leiden_0.45' não foi encontrado. "
            "Usando 'Estado_Microglial' como os IDs do Leiden 0.45."
        )

    else:
        raise ValueError(
            "Nem 'leiden_0.45' nem 'Estado_Microglial' estão em adata.obs."
        )

    adata.obs[CLUSTER_ANALISE] = (
        adata.obs[origem]
        .astype(str)
        .astype("category")
    )

    categorias = list(
        adata.obs[CLUSTER_ANALISE].cat.categories
    )

    try:
        categorias = sorted(categorias, key=int)
    except ValueError:
        categorias = sorted(categorias)

    adata.obs[CLUSTER_ANALISE] = (
        adata.obs[CLUSTER_ANALISE]
        .cat.reorder_categories(categorias)
    )

    return categorias


def diagnosticar_matriz_raw(adata_expr):
    """
    Verifica se a matriz oriunda de .raw se parece com contagens.

    Retorna um dicionário com estatísticas simples sem densificar
    a matriz inteira. Isso evita decidir de forma automática e
    insegura se devemos aplicar normalize_total + log1p novamente.
    """
    X = adata_expr.X

    if hasattr(X, "data"):
        valores = np.asarray(X.data)
    else:
        valores = np.asarray(X).ravel()

    if valores.size == 0:
        raise ValueError("A matriz de expressão está vazia.")

    if valores.size > 200_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(
            valores.size,
            size=200_000,
            replace=False,
        )
        valores = valores[idx]

    if not np.isfinite(valores).all():
        raise ValueError(
            "A matriz de expressão contém NaN ou valores infinitos."
        )

    minimo = float(np.min(valores))
    maximo = float(np.max(valores))
    mediana = float(np.median(valores))

    if minimo < 0:
        raise ValueError(
            "A matriz em adata.raw contém valores negativos. "
            "Ela não deve ser tratada automaticamente como contagens."
        )

    fracao_inteiros = float(
        np.mean(
            np.isclose(
                valores,
                np.rint(valores),
                atol=1e-6,
            )
        )
    )

    parece_contagem = fracao_inteiros >= 0.95

    print(
        "Diagnóstico de adata.raw:"
    )
    print(
        f"  fração de valores compatíveis com inteiros: "
        f"{100 * fracao_inteiros:.2f}%"
    )
    print(
        f"  mínimo={minimo:.4f} | "
        f"mediana={mediana:.4f} | "
        f"máximo={maximo:.4f}"
    )
    print(
        f"  interpretação automática: "
        f"{'compatível com contagens' if parece_contagem else 'não compatível com contagens inteiras'}"
    )

    return {
        "fracao_inteiros": fracao_inteiros,
        "minimo": minimo,
        "mediana": mediana,
        "maximo": maximo,
        "parece_contagem": parece_contagem,
    }


def preparar_expressao(adata_original, pasta_resultados):
    """
    Prepara a matriz usada na anotação sem aplicar transformação dupla.

    Regra de segurança:
    1) Se .raw parece conter contagens inteiras:
       normalize_total -> log1p.
    2) Se .raw NÃO parece conter contagens, mas existe metadado log1p:
       assume-se que a matriz já foi transformada e ela é usada como está.
    3) Se .raw NÃO parece conter contagens e não há evidência de log1p:
       a execução é interrompida para não transformar os dados às cegas.
    """
    if adata_original.raw is not None:
        print("Usando adata.raw como ponto de partida.")

        adata_expr = adata_original.raw.to_adata()
        adata_expr = remover_genes_ruido(adata_expr)

        diagnostico = diagnosticar_matriz_raw(
            adata_expr
        )

        log1p_no_original = (
            "log1p" in adata_original.uns
        )

        log1p_na_copia_raw = (
            "log1p" in adata_expr.uns
        )

        diagnostico_saida = {
            **diagnostico,
            "log1p_em_adata_original_uns": log1p_no_original,
            "log1p_em_raw_to_adata_uns": log1p_na_copia_raw,
        }

        pd.DataFrame(
            [diagnostico_saida]
        ).to_csv(
            pasta_resultados / "Diagnostico_matriz_expressao.csv",
            index=False,
        )

        if diagnostico["parece_contagem"]:
            # O aviso do Scanpy pode ser causado por metadado herdado.
            # Neste ramo X é compatível com contagens, então removemos
            # SOMENTE a marca de log1p antes da transformação correta.
            adata_expr.uns.pop("log1p", None)

            print(
                "adata.raw é compatível com contagens. "
                "Aplicando normalize_total + log1p uma única vez."
            )

            sc.pp.normalize_total(
                adata_expr,
                target_sum=1e4,
            )

            sc.pp.log1p(
                adata_expr
            )

        elif log1p_no_original or log1p_na_copia_raw:
            print(
                "adata.raw não parece conter contagens inteiras e há "
                "metadado de log1p no objeto. Para evitar dupla "
                "transformação, a matriz será usada COMO ESTÁ."
            )

        else:
            raise ValueError(
                "adata.raw não parece conter contagens inteiras e não há "
                "metadado suficiente para confirmar que a matriz já está "
                "log-transformada. A execução foi interrompida por "
                "segurança. Envie o arquivo "
                "'results/anotacao_045/Diagnostico_matriz_expressao.csv' "
                "e a saída do job antes de prosseguir."
            )

    else:
        warnings.warn(
            "adata.raw não existe. Usando uma cópia de adata.X. "
            "Nenhuma transformação adicional será aplicada "
            "automaticamente."
        )

        adata_expr = remover_genes_ruido(
            adata_original
        )

    adata_expr.obs[CLUSTER_ANALISE] = (
        adata_original.obs[CLUSTER_ANALISE]
        .copy()
    )

    if "X_umap" in adata_original.obsm:
        adata_expr.obsm["X_umap"] = (
            adata_original.obsm["X_umap"].copy()
        )

    return adata_expr


def exportar_status_paineis(
    adata_expr,
    pasta_resultados,
):
    registros = []

    for categoria, painel in [
        ("Programa_microglial", PROGRAMAS_MICROGLIAIS),
        ("Possivel_contaminacao", PAINEIS_CONTAMINACAO),
    ]:
        for nome, genes in painel.items():
            for gene in genes:
                registros.append(
                    {
                        "Categoria": categoria,
                        "Painel": nome,
                        "Gene": gene,
                        "Presente_no_dataset": (
                            gene in adata_expr.var_names
                        ),
                    }
                )

    pd.DataFrame(
        registros
    ).to_csv(
        pasta_resultados / "Status_genes_paineis.csv",
        index=False,
    )


def exportar_marcadores(
    adata_expr,
    clusters,
    pasta_resultados,
):
    print(
        "\nCalculando marcadores Wilcoxon "
        "de cada cluster contra o restante..."
    )

    sc.tl.rank_genes_groups(
        adata_expr,
        groupby=CLUSTER_ANALISE,
        method="wilcoxon",
        key_added="marcadores_leiden045",
        pts=True,
        use_raw=False,
    )

    tabelas = []

    for cluster in clusters:
        df = sc.get.rank_genes_groups_df(
            adata_expr,
            group=cluster,
            key="marcadores_leiden045",
        ).copy()

        df.insert(
            0,
            "Cluster_Leiden045",
            str(cluster),
        )

        tabelas.append(df)

        df.head(50).to_csv(
            pasta_resultados
            / f"Cluster_{cluster}_Top50_marcadores.csv",
            index=False,
        )

    todos = pd.concat(
        tabelas,
        ignore_index=True,
    )

    todos.to_csv(
        pasta_resultados
        / "Marcadores_todos_clusters_Leiden045.csv",
        index=False,
    )

    top50 = (
        todos
        .groupby(
            "Cluster_Leiden045",
            group_keys=False,
        )
        .head(50)
    )

    top50.to_csv(
        pasta_resultados
        / "Top50_marcadores_por_cluster_Leiden045.csv",
        index=False,
    )

    return todos


def calcular_scores(
    adata_expr,
    painel,
    prefixo,
):
    colunas = []
    genes_usados = {}

    for nome, genes in painel.items():
        presentes = genes_presentes(
            adata_expr,
            genes,
        )

        genes_usados[nome] = presentes

        if len(presentes) < 2:
            print(
                f"AVISO: '{nome}' possui somente "
                f"{len(presentes)} gene(s) presente(s); "
                "score não será calculado."
            )
            continue

        score_name = f"{prefixo}_{nome}"

        sc.tl.score_genes(
            adata_expr,
            gene_list=presentes,
            score_name=score_name,
            random_state=0,
            use_raw=False,
        )

        colunas.append(score_name)

    return colunas, genes_usados


def resumir_scores(
    adata_expr,
    colunas,
    prefixo,
    nome,
    pasta_resultados,
):
    if not colunas:
        return pd.DataFrame()

    obs = adata_expr.obs[
        [CLUSTER_ANALISE] + colunas
    ].copy()

    media = (
        obs
        .groupby(
            CLUSTER_ANALISE,
            observed=True,
        )[colunas]
        .mean()
    )

    mediana = (
        obs
        .groupby(
            CLUSTER_ANALISE,
            observed=True,
        )[colunas]
        .median()
    )

    media.columns = [
        c.replace(prefixo, "")
        for c in media.columns
    ]

    mediana.columns = [
        c.replace(prefixo, "")
        for c in mediana.columns
    ]

    media.index = media.index.astype(str)
    mediana.index = mediana.index.astype(str)

    media.index.name = "Cluster_Leiden045"
    mediana.index.name = "Cluster_Leiden045"

    media.to_csv(
        pasta_resultados / f"{nome}_media.csv"
    )

    mediana.to_csv(
        pasta_resultados / f"{nome}_mediana.csv"
    )

    return media


def zscore_colunas(df):
    if df.empty:
        return df.copy()

    z = df.copy()

    for col in z.columns:
        desvio = z[col].std(ddof=0)

        if desvio == 0 or np.isnan(desvio):
            z[col] = 0.0
        else:
            z[col] = (
                z[col] - z[col].mean()
            ) / desvio

    return z


def heatmap(
    df,
    caminho,
    titulo,
    xlabel,
    ylabel,
    cmap="RdBu_r",
    annot=True,
    formato=".2f",
):
    if df.empty:
        return

    matriz = df.to_numpy(dtype=float)

    fig, ax = plt.subplots(
        figsize=(
            max(8, 0.9 * len(df.columns) + 3),
            max(6, 0.55 * len(df.index) + 3),
        )
    )

    im = ax.imshow(
        matriz,
        aspect="auto",
        cmap=cmap,
    )

    ax.set_xticks(
        np.arange(len(df.columns))
    )

    ax.set_xticklabels(
        df.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(df.index))
    )

    ax.set_yticklabels(
        df.index
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)

    if annot:
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                if np.isfinite(matriz[i, j]):
                    ax.text(
                        j,
                        i,
                        format(
                            matriz[i, j],
                            formato,
                        ),
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.8,
    )

    plt.tight_layout()

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def dotplot_painel(
    adata_expr,
    painel,
    caminho,
    titulo,
):
    painel_filtrado = {
        nome: genes_presentes(
            adata_expr,
            genes,
        )
        for nome, genes in painel.items()
    }

    painel_filtrado = {
        nome: genes
        for nome, genes in painel_filtrado.items()
        if genes
    }

    if not painel_filtrado:
        return

    sc.pl.dotplot(
        adata_expr,
        var_names=painel_filtrado,
        groupby=CLUSTER_ANALISE,
        standard_scale="var",
        show=False,
        title=titulo,
    )

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close("all")


def analisar_supertype(
    adata_expr,
    pasta_resultados,
    pasta_figuras,
):
    """
    Compara Leiden 0.45 com Supertype do SEA-AD.

    CORREÇÃO IMPORTANTE:
    Supertype pode ter dtype pandas.Categorical. Nesse caso,
    fillna("Desconhecido") falha se "Desconhecido" não for uma
    categoria registrada. Por isso convertemos primeiro a série
    para object e somente depois substituímos os NAs.
    """
    if "Supertype" not in adata_expr.obs.columns:
        print(
            "AVISO: coluna 'Supertype' não encontrada; "
            "etapa SEA-AD ignorada."
        )
        return pd.DataFrame()

    dados = adata_expr.obs[
        [
            CLUSTER_ANALISE,
            "Supertype",
        ]
    ].copy()

    # Corrige o erro:
    # TypeError: Cannot setitem on a Categorical with a new category
    dados["Supertype"] = (
        dados["Supertype"]
        .astype("object")
    )

    dados["Supertype"] = (
        dados["Supertype"]
        .where(
            dados["Supertype"].notna(),
            "Desconhecido",
        )
        .astype(str)
    )

    tabela = pd.crosstab(
        dados["Supertype"],
        dados[CLUSTER_ANALISE],
        normalize="columns",
    ) * 100.0

    tabela.to_csv(
        pasta_resultados
        / "Composicao_SEAAD_Supertype_por_Leiden045.csv"
    )

    heatmap(
        tabela,
        pasta_figuras
        / "Heatmap_Leiden045_vs_SEAAD_Supertype.png",
        "Correspondência entre Leiden 0.45 e SEA-AD Supertype (%)",
        "Cluster Leiden 0.45",
        "SEA-AD Supertype",
        cmap="Blues",
        annot=False,
    )

    registros = []

    for cluster in tabela.columns:
        coluna = tabela[cluster]

        registros.append(
            {
                "Cluster_Leiden045": str(cluster),
                "SEAAD_Supertype_dominante": coluna.idxmax(),
                "SEAAD_Supertype_percentual": float(
                    coluna.max()
                ),
            }
        )

    resumo = (
        pd.DataFrame(registros)
        .set_index("Cluster_Leiden045")
    )

    resumo.to_csv(
        pasta_resultados
        / "Resumo_SEAAD_Supertype_dominante.csv"
    )

    return resumo


def analisar_doadores(
    adata_expr,
    pasta_resultados,
):
    if "donor_id" not in adata_expr.obs.columns:
        print(
            "AVISO: donor_id não encontrado; "
            "etapa donor-level ignorada."
        )
        return pd.DataFrame()

    obs = adata_expr.obs[
        [
            "donor_id",
            CLUSTER_ANALISE,
        ]
    ].copy()

    obs["donor_id"] = (
        obs["donor_id"]
        .astype(str)
    )

    contagem = pd.crosstab(
        obs["donor_id"],
        obs[CLUSTER_ANALISE],
    )

    contagem.to_csv(
        pasta_resultados
        / "Contagem_celulas_doador_por_Leiden045.csv"
    )

    total_celulas = len(obs)
    total_doadores = obs["donor_id"].nunique()

    registros = []

    for cluster in contagem.columns:
        valores = contagem[cluster]

        n_celulas = int(
            valores.sum()
        )

        n_doadores = int(
            (valores > 0).sum()
        )

        max_doador = int(
            valores.max()
        )

        registros.append(
            {
                "Cluster_Leiden045": str(cluster),
                "N_celulas": n_celulas,
                "Percentual_total_celulas": (
                    100.0
                    * n_celulas
                    / total_celulas
                ),
                "N_doadores_com_cluster": n_doadores,
                "Percentual_doadores_representados": (
                    100.0
                    * n_doadores
                    / total_doadores
                ),
                "Mediana_celulas_por_doador": float(
                    valores.median()
                ),
                "Max_celulas_um_doador": max_doador,
                "Percentual_cluster_maior_doador": (
                    100.0
                    * max_doador
                    / n_celulas
                    if n_celulas
                    else np.nan
                ),
            }
        )

    resumo = (
        pd.DataFrame(registros)
        .set_index("Cluster_Leiden045")
    )

    resumo.to_csv(
        pasta_resultados
        / "Resumo_representatividade_por_doador.csv"
    )

    return resumo


def top_genes_string(
    marcadores,
    n=10,
):
    resultado = {}

    for cluster, df in marcadores.groupby(
        "Cluster_Leiden045",
        sort=False,
    ):
        resultado[str(cluster)] = "; ".join(
            df["names"]
            .dropna()
            .astype(str)
            .head(n)
            .tolist()
        )

    return resultado


def linha_por_indice_str(
    df,
    cluster,
):
    if df.empty:
        return None

    mascara = (
        df.index.astype(str)
        == str(cluster)
    )

    if not mascara.any():
        return None

    return df.loc[mascara].iloc[0]


def criar_resumo_evidencias(
    clusters,
    marcadores,
    z_programas,
    z_contaminacao,
    resumo_supertype,
    resumo_doadores,
    pasta_resultados,
):
    top10 = top_genes_string(
        marcadores,
        n=10,
    )

    registros = []

    for cluster in clusters:
        cluster = str(cluster)

        registro = {
            "Cluster_Leiden045": cluster,
            "Top10_marcadores": top10.get(
                cluster,
                "",
            ),
        }

        linha = linha_por_indice_str(
            z_programas,
            cluster,
        )

        if linha is not None:
            registro[
                "Programa_relativamente_mais_alto"
            ] = linha.idxmax()

            registro[
                "Z_programa_max"
            ] = float(
                linha.max()
            )

        linha = linha_por_indice_str(
            z_contaminacao,
            cluster,
        )

        if linha is not None:
            registro[
                "Painel_contaminacao_relativamente_mais_alto"
            ] = linha.idxmax()

            registro[
                "Z_contaminacao_max"
            ] = float(
                linha.max()
            )

        linha = linha_por_indice_str(
            resumo_supertype,
            cluster,
        )

        if linha is not None:
            registro.update(
                linha.to_dict()
            )

        linha = linha_por_indice_str(
            resumo_doadores,
            cluster,
        )

        if linha is not None:
            registro.update(
                linha.to_dict()
            )

        registros.append(registro)

    resumo = pd.DataFrame(
        registros
    )

    resumo.to_csv(
        pasta_resultados
        / "Resumo_evidencias_por_cluster.csv",
        index=False,
    )

    template = resumo.copy()

    template[
        "Nome_biologico_final"
    ] = ""

    template[
        "Confianca"
    ] = ""

    template[
        "Justificativa"
    ] = ""

    template.to_csv(
        pasta_resultados
        / "Template_anotacao_manual_Leiden045.csv",
        index=False,
    )

    return resumo


def aplicar_anotacao_final(
    adata_original,
    clusters,
    caminho_h5ad,
    pasta_figuras,
):
    if not ANOTACAO_FINAL:
        print(
            "\nANOTACAO_FINAL está vazio: "
            "nenhum nome biológico será atribuído "
            "nesta primeira execução."
        )
        return

    esperados = {
        str(c)
        for c in clusters
    }

    fornecidos = {
        str(c)
        for c in ANOTACAO_FINAL
    }

    if esperados != fornecidos:
        raise ValueError(
            "ANOTACAO_FINAL precisa conter exatamente todos os clusters. "
            f"Faltantes: {sorted(esperados - fornecidos)}; "
            f"extras: {sorted(fornecidos - esperados)}."
        )

    adata_original.obs[
        "Estado_Microglial_Biologico"
    ] = (
        adata_original.obs[
            CLUSTER_ANALISE
        ]
        .astype(str)
        .map(ANOTACAO_FINAL)
        .astype("category")
    )

    saida_h5ad = (
        caminho_h5ad.parent
        / "adata_limpo_anotado_biologicamente.h5ad"
    )

    adata_original.write_h5ad(
        saida_h5ad,
        compression="gzip",
    )

    print(
        f"Objeto biologicamente anotado salvo em: "
        f"{saida_h5ad}"
    )

    if "X_umap" in adata_original.obsm:
        sc.pl.umap(
            adata_original,
            color="Estado_Microglial_Biologico",
            title=(
                "Estados microgliais biologicamente anotados "
                "- Leiden 0.45"
            ),
            show=False,
        )

        plt.savefig(
            pasta_figuras
            / "UMAP_Estado_Microglial_Biologico.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close("all")


# =========================================================
# ETAPA 1: CARREGAMENTO
# =========================================================

caminho_h5ad = localizar_h5ad()
pasta_projeto = caminho_h5ad.parent

pasta_resultados = (
    pasta_projeto
    / "results"
    / "anotacao_045"
)

pasta_figuras = (
    pasta_projeto
    / "figures"
    / "anotacao_045"
)

pasta_resultados.mkdir(
    parents=True,
    exist_ok=True,
)

pasta_figuras.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    f"\nCarregando: {caminho_h5ad}"
)

adata_original = sc.read_h5ad(
    caminho_h5ad
)

print(
    f"Células: {adata_original.n_obs}"
)

print(
    f"Genes: {adata_original.n_vars}"
)


# =========================================================
# ETAPA 2: CLUSTERS LEIDEN 0.45
# =========================================================

clusters = garantir_cluster_045(
    adata_original
)

print(
    "\nClusters Leiden 0.45 encontrados:"
)

print(
    clusters
)

print(
    "\nNúmero de células por cluster:"
)

print(
    adata_original.obs[
        CLUSTER_ANALISE
    ]
    .value_counts()
    .sort_index()
)


# =========================================================
# ETAPA 3: MATRIZ DE EXPRESSÃO
# =========================================================

adata_expr = preparar_expressao(
    adata_original,
    pasta_resultados,
)

exportar_status_paineis(
    adata_expr,
    pasta_resultados,
)


# =========================================================
# ETAPA 4: MARCADORES DOS CLUSTERS
# =========================================================

marcadores = exportar_marcadores(
    adata_expr,
    clusters,
    pasta_resultados,
)


# Dendrogram explícito para evitar o warning automático do dotplot.
print(
    "Calculando dendrograma dos clusters "
    "para organização dos dotplots..."
)

sc.tl.dendrogram(
    adata_expr,
    groupby=CLUSTER_ANALISE,
)


print(
    "Gerando dotplot dos Top 10 marcadores por cluster..."
)

sc.pl.rank_genes_groups_dotplot(
    adata_expr,
    n_genes=10,
    key="marcadores_leiden045",
    groupby=CLUSTER_ANALISE,
    dendrogram=True,
    title=(
        "Top 10 genes marcadores por cluster "
        "- Leiden 0.45"
    ),
    show=False,
)

plt.savefig(
    pasta_figuras
    / "DotPlot_Top10_marcadores_Leiden045.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close("all")


# =========================================================
# ETAPA 5: PROGRAMAS MICROGLIAIS
# =========================================================

print(
    "\nCalculando programas gênicos microgliais..."
)

scores_programas, _ = calcular_scores(
    adata_expr,
    PROGRAMAS_MICROGLIAIS,
    "ScorePrograma",
)

media_programas = resumir_scores(
    adata_expr,
    scores_programas,
    "ScorePrograma_",
    "Scores_programas_microgliais",
    pasta_resultados,
)

z_programas = zscore_colunas(
    media_programas
)

z_programas.to_csv(
    pasta_resultados
    / "Scores_programas_microgliais_Z_por_cluster.csv"
)

heatmap(
    z_programas,
    pasta_figuras
    / "Heatmap_programas_microgliais_Z.png",
    "Programas microgliais por cluster - z-score entre clusters",
    "Programa gênico",
    "Cluster Leiden 0.45",
)

dotplot_painel(
    adata_expr,
    PROGRAMAS_MICROGLIAIS,
    pasta_figuras
    / "DotPlot_paineis_microgliais.png",
    "Painéis de genes microgliais - Leiden 0.45",
)


# =========================================================
# ETAPA 6: POSSÍVEIS OUTRAS LINHAGENS / CONTAMINAÇÃO
# =========================================================

print(
    "\nCalculando sinais exploratórios "
    "de possível contaminação..."
)

scores_contaminacao, _ = calcular_scores(
    adata_expr,
    PAINEIS_CONTAMINACAO,
    "ScoreContaminacao",
)

media_contaminacao = resumir_scores(
    adata_expr,
    scores_contaminacao,
    "ScoreContaminacao_",
    "Scores_contaminacao",
    pasta_resultados,
)

z_contaminacao = zscore_colunas(
    media_contaminacao
)

z_contaminacao.to_csv(
    pasta_resultados
    / "Scores_contaminacao_Z_por_cluster.csv"
)

heatmap(
    z_contaminacao,
    pasta_figuras
    / "Heatmap_contaminacao_Z.png",
    "Sinais relativos de outras linhagens - z-score entre clusters",
    "Painel de genes",
    "Cluster Leiden 0.45",
)

dotplot_painel(
    adata_expr,
    PAINEIS_CONTAMINACAO,
    pasta_figuras
    / "DotPlot_paineis_contaminacao.png",
    "Genes de possíveis outras linhagens - Leiden 0.45",
)


# =========================================================
# ETAPA 7: REFERÊNCIA SEA-AD
# =========================================================

print(
    "\nComparando Leiden 0.45 "
    "com SEA-AD Supertype..."
)

resumo_supertype = analisar_supertype(
    adata_expr,
    pasta_resultados,
    pasta_figuras,
)


# =========================================================
# ETAPA 8: REPRESENTATIVIDADE ENTRE DOADORES
# =========================================================

print(
    "\nAvaliando representatividade "
    "dos clusters entre doadores..."
)

resumo_doadores = analisar_doadores(
    adata_expr,
    pasta_resultados,
)


# =========================================================
# ETAPA 9: UMAP DO LEIDEN 0.45
# =========================================================

if "X_umap" in adata_expr.obsm:
    sc.pl.umap(
        adata_expr,
        color=CLUSTER_ANALISE,
        title=(
            "Clusters microgliais "
            "- Leiden 0.45"
        ),
        legend_loc="on data",
        show=False,
    )

    plt.savefig(
        pasta_figuras
        / "UMAP_Leiden045.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close("all")


# =========================================================
# ETAPA 10: RESUMO INTEGRADO
# =========================================================

print(
    "\nConstruindo resumo integrado das evidências..."
)

resumo = criar_resumo_evidencias(
    clusters,
    marcadores,
    z_programas,
    z_contaminacao,
    resumo_supertype,
    resumo_doadores,
    pasta_resultados,
)


# =========================================================
# ETAPA 11: ANOTAÇÃO FINAL, SE O MAPA ESTIVER PREENCHIDO
# =========================================================

aplicar_anotacao_final(
    adata_original,
    clusters,
    caminho_h5ad,
    pasta_figuras,
)


# =========================================================
# FINALIZAÇÃO
# =========================================================

print(
    "\n============================================"
)

print(
    "ANÁLISE PARA ANOTAÇÃO LEIDEN 0.45 CONCLUÍDA"
)

print(
    "============================================"
)

print(
    f"Resultados: {pasta_resultados}"
)

print(
    f"Figuras: {pasta_figuras}"
)

print(
    "\nArquivos prioritários para revisão:"
)

print(
    "1. Resumo_evidencias_por_cluster.csv"
)

print(
    "2. Template_anotacao_manual_Leiden045.csv"
)

print(
    "3. Top50_marcadores_por_cluster_Leiden045.csv"
)

print(
    "4. Composicao_SEAAD_Supertype_por_Leiden045.csv"
)

print(
    "5. Scores_programas_microgliais_Z_por_cluster.csv"
)

print(
    "6. Scores_contaminacao_Z_por_cluster.csv"
)

print(
    "\nNa primeira execução, mantenha ANOTACAO_FINAL = {}. "
    "Os nomes biológicos serão definidos somente após revisar "
    "as evidências dos 10 clusters."
)