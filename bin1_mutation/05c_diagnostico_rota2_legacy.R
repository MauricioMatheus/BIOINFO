# ============================================================
# 05c_diagnostico_rota2_legacy.R
#
# Diagnóstico da Rota 2 usando o caminho LEGACY do glmQLFit.
#
# Objetivo:
#   descobrir se o SIGBUS está especificamente no novo caminho
#   edgeR v4 de glmQLFit(legacy=FALSE), que usa ajuste de
#   deviances/df para contagens pequenas.
#
# IMPORTANTE:
#   ESTE É UM TESTE DIAGNÓSTICO.
#   NÃO substitui automaticamente a análise final.
#
# Reutiliza:
#   results/diagnostico_rota2_edger/
#       rota2_counts_neighborhood_x_donor.tsv
#       rota2_metadata_donor.tsv
#
# Executar:
#
#   Rscript bin1_mutation/05c_diagnostico_rota2_legacy.R
#
# ============================================================

options(warn = 1)

checkpoint <- function(msg) {
    cat(sprintf("\n[CHECKPOINT OK] %s\n", msg))
    flush.console()
}

inicio <- function(msg) {
    cat(sprintf("\n[INICIANDO] %s\n", msg))
    flush.console()
}

fail <- function(msg) {
    stop(msg, call. = FALSE)
}


# ------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------

args_all <- commandArgs(
    trailingOnly = FALSE
)

file_arg <- grep(
    "^--file=",
    args_all,
    value = TRUE
)

if (length(file_arg) == 0) {
    script_dir <- normalizePath(
        getwd()
    )
} else {
    script_path <- sub(
        "^--file=",
        "",
        file_arg[1]
    )

    script_dir <- dirname(
        normalizePath(
            script_path
        )
    )
}

base_dir <- file.path(
    script_dir,
    "results",
    "diagnostico_rota2_edger"
)

counts_file <- file.path(
    base_dir,
    "rota2_counts_neighborhood_x_donor.tsv"
)

metadata_file <- file.path(
    base_dir,
    "rota2_metadata_donor.tsv"
)

result_file <- file.path(
    base_dir,
    "rota2_edgeR_legacy_TRUE_resultado.csv"
)

session_file <- file.path(
    base_dir,
    "rota2_edgeR_legacy_TRUE_sessionInfo.txt"
)

environment_file <- file.path(
    base_dir,
    "rota2_edgeR_legacy_TRUE_ambiente.txt"
)


cat("============================================================\n")
cat("DIAGNÓSTICO edgeR — ROTA 2 — legacy=TRUE\n")
cat("============================================================\n")
cat("Counts: ", counts_file, "\n", sep = "")
cat("Metadata: ", metadata_file, "\n", sep = "")
flush.console()


# ------------------------------------------------------------
# Pacotes / ambiente
# ------------------------------------------------------------

suppressPackageStartupMessages(
    library(edgeR)
)

suppressPackageStartupMessages(
    library(limma)
)

edgeR_desc <- packageDescription(
    "edgeR"
)

limma_desc <- packageDescription(
    "limma"
)

edgeR_lib <- system.file(
    "libs",
    package = "edgeR"
)

cat(
    "\nedgeR version: ",
    as.character(
        packageVersion("edgeR")
    ),
    "\n",
    sep = ""
)

cat(
    "limma version: ",
    as.character(
        packageVersion("limma")
    ),
    "\n",
    sep = ""
)

cat(
    "edgeR Built: ",
    edgeR_desc$Built,
    "\n",
    sep = ""
)

cat(
    "edgeR libs: ",
    edgeR_lib,
    "\n",
    sep = ""
)

cat(
    "R: ",
    R.version.string,
    "\n",
    sep = ""
)

flush.console()

writeLines(
    c(
        paste0(
            "R: ",
            R.version.string
        ),
        paste0(
            "edgeR: ",
            as.character(
                packageVersion("edgeR")
            )
        ),
        paste0(
            "edgeR Built: ",
            edgeR_desc$Built
        ),
        paste0(
            "edgeR libs: ",
            edgeR_lib
        ),
        paste0(
            "limma: ",
            as.character(
                packageVersion("limma")
            )
        ),
        paste0(
            "OMP_NUM_THREADS=",
            Sys.getenv("OMP_NUM_THREADS")
        ),
        paste0(
            "OPENBLAS_NUM_THREADS=",
            Sys.getenv("OPENBLAS_NUM_THREADS")
        ),
        paste0(
            "MKL_NUM_THREADS=",
            Sys.getenv("MKL_NUM_THREADS")
        )
    ),
    environment_file
)

checkpoint(
    "pacotes e ambiente registrados"
)


# ------------------------------------------------------------
# Dados
# ------------------------------------------------------------

if (!file.exists(counts_file)) {
    fail(
        paste(
            "Counts não encontrados:",
            counts_file
        )
    )
}

if (!file.exists(metadata_file)) {
    fail(
        paste(
            "Metadados não encontrados:",
            metadata_file
        )
    )
}

counts_df <- read.delim(
    counts_file,
    row.names = 1,
    check.names = FALSE
)

metadata <- read.delim(
    metadata_file,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
)

counts <- as.matrix(
    counts_df
)

storage.mode(
    counts
) <- "integer"

if (anyNA(counts)) {
    fail(
        "Counts contêm NA."
    )
}

if (any(counts < 0)) {
    fail(
        "Counts contêm valores negativos."
    )
}

metadata <- metadata[
    colnames(counts),
    ,
    drop = FALSE
]

if (!identical(
    colnames(counts),
    rownames(metadata)
)) {
    fail(
        "Ordem counts/metadata incompatível."
    )
}

checkpoint(
    paste0(
        "dados lidos | counts=",
        nrow(counts),
        "x",
        ncol(counts),
        " | metadata=",
        nrow(metadata),
        "x",
        ncol(metadata)
    )
)


# ------------------------------------------------------------
# Fatores
# ------------------------------------------------------------

metadata$Modelo_Recessivo <- factor(
    metadata$Modelo_Recessivo,
    levels = c(
        "Nao_Recessivo",
        "Recessivo"
    )
)

metadata$Age_at_death <- factor(
    metadata$Age_at_death,
    levels = c(
        "65_77",
        "78_89",
        "90_plus"
    )
)

metadata$Braak_stage <- factor(
    metadata$Braak_stage,
    levels = c(
        "Braak_0",
        "Braak_II",
        "Braak_III",
        "Braak_IV",
        "Braak_V",
        "Braak_VI"
    )
)

metadata$Thal_phase <- factor(
    metadata$Thal_phase,
    levels = c(
        "Thal_1",
        "Thal_2",
        "Thal_3",
        "Thal_4",
        "Thal_5"
    )
)

if (anyNA(metadata)) {
    print(
        colSums(
            is.na(metadata)
        )
    )

    fail(
        "NAs após fatoração."
    )
}

checkpoint(
    "fatores definidos"
)


# ------------------------------------------------------------
# Design idêntico à Rota 2
# ------------------------------------------------------------

design <- model.matrix(
    ~ Modelo_Recessivo
    + Age_at_death
    + Braak_stage
    + Thal_phase
    + 0,
    data = metadata
)

rank_design <- qr(
    design
)$rank

cat(
    "\nDesign:\n"
)

print(
    dim(
        design
    )
)

cat(
    "rank = ",
    rank_design,
    "\n",
    sep = ""
)

print(
    colnames(
        design
    )
)

flush.console()

if (
    rank_design
    != ncol(
        design
    )
) {
    fail(
        "Design não possui rank completo."
    )
}

checkpoint(
    "model.matrix"
)


# ------------------------------------------------------------
# DGEList / TMM / estimateDisp
# ------------------------------------------------------------

lib_size <- colSums(
    counts
)

keep_smp <- (
    lib_size > 0
)

counts_use <- counts[
    ,
    keep_smp,
    drop = FALSE
]

design_use <- design[
    keep_smp,
    ,
    drop = FALSE
]

lib_size_use <- lib_size[
    keep_smp
]

keep_nhoods <- (
    rowSums(
        counts_use
    )
    > 0
)

counts_use <- counts_use[
    keep_nhoods,
    ,
    drop = FALSE
]

dge <- DGEList(
    counts = counts_use,
    lib.size = lib_size_use
)

checkpoint(
    "DGEList"
)

dge <- calcNormFactors(
    dge,
    method = "TMM"
)

checkpoint(
    "calcNormFactors(TMM)"
)

inicio(
    "estimateDisp"
)

dge <- estimateDisp(
    dge,
    design_use
)

checkpoint(
    "estimateDisp"
)

cat(
    "\nDispersões disponíveis:\n"
)

cat(
    "  common.dispersion: ",
    dge$common.dispersion,
    "\n",
    sep = ""
)

cat(
    "  trended.dispersion length: ",
    length(
        dge$trended.dispersion
    ),
    "\n",
    sep = ""
)

cat(
    "  tagwise.dispersion length: ",
    length(
        dge$tagwise.dispersion
    ),
    "\n",
    sep = ""
)

flush.console()


# ------------------------------------------------------------
# TESTE PRINCIPAL: legacy=TRUE
# ------------------------------------------------------------

inicio(
    "glmQLFit(robust=TRUE, legacy=TRUE)"
)

fit_legacy <- glmQLFit(
    dge,
    design_use,
    robust = TRUE,
    legacy = TRUE
)

checkpoint(
    "glmQLFit(robust=TRUE, legacy=TRUE)"
)

cat(
    "\nComponentes principais do fit legacy:\n"
)

cat(
    "  coefficients: ",
    paste(
        dim(
            fit_legacy$coefficients
        ),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "  df.residual.zeros length: ",
    length(
        fit_legacy$df.residual.zeros
    ),
    "\n",
    sep = ""
)

cat(
    "  df.prior length: ",
    length(
        fit_legacy$df.prior
    ),
    "\n",
    sep = ""
)

cat(
    "  s2.post length: ",
    length(
        fit_legacy$s2.post
    ),
    "\n",
    sep = ""
)

flush.console()


# ------------------------------------------------------------
# Contraste
# ------------------------------------------------------------

contrast <- makeContrasts(
    Modelo_RecessivoRecessivo
    - Modelo_RecessivoNao_Recessivo,
    levels = design_use
)

checkpoint(
    "makeContrasts"
)


# ------------------------------------------------------------
# QLF test
# ------------------------------------------------------------

inicio(
    "glmQLFTest(legacy fit)"
)

qlf <- glmQLFTest(
    fit_legacy,
    contrast = contrast
)

checkpoint(
    "glmQLFTest(legacy fit)"
)


# ------------------------------------------------------------
# Resultados
# ------------------------------------------------------------

resultado <- topTags(
    qlf,
    sort.by = "none",
    n = Inf
)$table

checkpoint(
    "topTags"
)

resultado$Neighborhood <- rownames(
    resultado
)

resultado <- resultado[
    ,
    c(
        "Neighborhood",
        setdiff(
            colnames(
                resultado
            ),
            "Neighborhood"
        )
    )
]

write.csv(
    resultado,
    result_file,
    row.names = FALSE
)

writeLines(
    capture.output(
        sessionInfo()
    ),
    session_file
)

checkpoint(
    paste(
        "resultado salvo em",
        result_file
    )
)

cat(
    "\nResumo PValue/FDR do caminho legacy:\n"
)

print(
    summary(
        resultado[
            ,
            intersect(
                c(
                    "logFC",
                    "F",
                    "PValue",
                    "FDR"
                ),
                colnames(
                    resultado
                )
            ),
            drop = FALSE
        ]
    )
)

cat(
    "\n============================================================\n"
)
cat(
    "LEGACY=TRUE CONCLUIU SEM SIGBUS\n"
)
cat(
    "============================================================\n"
)