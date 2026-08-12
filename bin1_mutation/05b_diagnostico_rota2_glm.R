# ============================================================
# 05b_diagnostico_rota2_glm.R
#
# Diagnóstico fino do crash da Rota 2.
#
# Usa os TSVs já exportados por 05_diagnostico_rota2_export.py.
#
# Modos:
#
#   EDGER_STEP=glmFit
#       Testa glmFit() diretamente.
#
#   EDGER_STEP=mglmLevenberg
#       Testa diretamente o ajuste Levenberg de baixo nível usado
#       por glmFit() quando o design não é equivalente a one-way.
#
# Exemplos:
#
#   EDGER_STEP=glmFit Rscript bin1_mutation/05b_diagnostico_rota2_glm.R
#
#   EDGER_STEP=mglmLevenberg Rscript bin1_mutation/05b_diagnostico_rota2_glm.R
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
# Localiza a pasta do script
# ------------------------------------------------------------

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)

if (length(file_arg) == 0) {
    script_dir <- normalizePath(getwd())
} else {
    script_path <- sub("^--file=", "", file_arg[1])
    script_dir <- dirname(normalizePath(script_path))
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

mode <- Sys.getenv(
    "EDGER_STEP",
    unset = "glmFit"
)

if (!mode %in% c("glmFit", "mglmLevenberg")) {
    fail(
        paste0(
            "EDGER_STEP inválido: ", mode,
            ". Use glmFit ou mglmLevenberg."
        )
    )
}

cat("============================================================\n")
cat("DIAGNÓSTICO FINO edgeR — ROTA 2\n")
cat("============================================================\n")
cat("Modo: ", mode, "\n", sep = "")
cat("Counts: ", counts_file, "\n", sep = "")
cat("Metadata: ", metadata_file, "\n", sep = "")
flush.console()

# ------------------------------------------------------------
# Pacotes
# ------------------------------------------------------------

suppressPackageStartupMessages(library(edgeR))
suppressPackageStartupMessages(library(limma))

checkpoint(
    paste0(
        "pacotes carregados | edgeR=",
        as.character(packageVersion("edgeR")),
        " | limma=",
        as.character(packageVersion("limma"))
    )
)

# ------------------------------------------------------------
# Leitura
# ------------------------------------------------------------

if (!file.exists(counts_file)) {
    fail(paste("Counts não encontrados:", counts_file))
}

if (!file.exists(metadata_file)) {
    fail(paste("Metadados não encontrados:", metadata_file))
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

counts <- as.matrix(counts_df)
storage.mode(counts) <- "integer"

metadata <- metadata[
    colnames(counts),
    ,
    drop = FALSE
]

if (!identical(colnames(counts), rownames(metadata))) {
    fail("Ordem donor counts/metadata incompatível.")
}

checkpoint(
    paste0(
        "dados lidos | counts=",
        nrow(counts), "x", ncol(counts)
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
    print(colSums(is.na(metadata)))
    fail("NAs após fatoração.")
}

checkpoint("fatores definidos")

# ------------------------------------------------------------
# Design idêntico ao diagnóstico anterior
# ------------------------------------------------------------

design <- model.matrix(
    ~ Modelo_Recessivo
    + Age_at_death
    + Braak_stage
    + Thal_phase
    + 0,
    data = metadata
)

rank_design <- qr(design)$rank

cat("\nDesign:\n")
print(dim(design))
cat("rank = ", rank_design, "\n", sep = "")
print(colnames(design))
flush.console()

if (rank_design != ncol(design)) {
    fail("Design sem rank completo.")
}

checkpoint("model.matrix")

# ------------------------------------------------------------
# DGEList + TMM + estimateDisp
# ------------------------------------------------------------

lib_size <- colSums(counts)
keep_smp <- lib_size > 0

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

keep_nhoods <- rowSums(counts_use) > 0

counts_use <- counts_use[
    keep_nhoods,
    ,
    drop = FALSE
]

dge <- DGEList(
    counts = counts_use,
    lib.size = lib_size_use
)

checkpoint("DGEList")

dge <- calcNormFactors(
    dge,
    method = "TMM"
)

checkpoint("calcNormFactors")

inicio("estimateDisp")

dge <- estimateDisp(
    dge,
    design_use
)

checkpoint("estimateDisp")

# ------------------------------------------------------------
# Informa qual caminho glmFit tenderia a usar
# ------------------------------------------------------------

group_from_design <- designAsFactor(
    design_use
)

cat("\nDiagnóstico do designAsFactor:\n")
cat(
    "  nlevels(group) = ",
    nlevels(group_from_design),
    "\n",
    sep = ""
)
cat(
    "  ncol(design)   = ",
    ncol(design_use),
    "\n",
    sep = ""
)
flush.console()

if (nlevels(group_from_design) == ncol(design_use)) {
    cat(
        "  Caminho esperado de glmFit: mglmOneWay\n"
    )
} else {
    cat(
        "  Caminho esperado de glmFit: mglmLevenberg\n"
    )
}
flush.console()

# ------------------------------------------------------------
# TESTE 1 — glmFit diretamente
# ------------------------------------------------------------

if (mode == "glmFit") {
    inicio("glmFit(dge, design_use)")

    fit <- glmFit(
        dge,
        design_use
    )

    checkpoint("glmFit")

    cat("\nMétodo reportado por glmFit: ", fit$method, "\n", sep = "")
    cat(
        "Dimensão coefficients: ",
        paste(dim(fit$coefficients), collapse = " x "),
        "\n",
        sep = ""
    )
    cat(
        "Deviance finita: ",
        all(is.finite(fit$deviance)),
        "\n",
        sep = ""
    )

    checkpoint(
        "glmFit terminou; o crash do glmQLFit ocorre DEPOIS de glmFit"
    )
}

# ------------------------------------------------------------
# TESTE 2 — mglmLevenberg diretamente
# ------------------------------------------------------------

if (mode == "mglmLevenberg") {
    dispersion <- getDispersion(
        dge
    )

    offset <- getOffset(
        dge
    )

    cat("\nEstruturas para mglmLevenberg:\n")
    cat(
        "  counts      = ",
        paste(dim(dge$counts), collapse = " x "),
        "\n",
        sep = ""
    )
    cat(
        "  design      = ",
        paste(dim(design_use), collapse = " x "),
        "\n",
        sep = ""
    )
    cat(
        "  dispersion  = length ",
        length(dispersion),
        "\n",
        sep = ""
    )
    cat(
        "  offset      = length ",
        length(offset),
        "\n",
        sep = ""
    )
    flush.console()

    inicio("mglmLevenberg")

    fit_low <- mglmLevenberg(
        y = dge$counts,
        design = design_use,
        dispersion = dispersion,
        offset = offset,
        weights = dge$weights,
        maxit = 250
    )

    checkpoint("mglmLevenberg")

    cat(
        "\nFalhas reportadas pelo ajuste: ",
        sum(fit_low$failed),
        "\n",
        sep = ""
    )

    cat(
        "Iterações máximas: ",
        max(fit_low$iter),
        "\n",
        sep = ""
    )

    checkpoint(
        "mglmLevenberg terminou sem SIGBUS"
    )
}

cat("\n============================================================\n")
cat("DIAGNÓSTICO FINO CONCLUÍDO SEM CRASH\n")
cat("============================================================\n")