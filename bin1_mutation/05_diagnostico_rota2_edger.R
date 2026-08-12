# ============================================================
# 05_diagnostico_rota2_edger.R
#
# Diagnóstico da Rota 2 executando edgeR DIRETAMENTE em Rscript,
# sem Python e sem rpy2.
#
# Por padrão:
#   glmQLFit(..., robust=TRUE)
#
# Somente se o teste robust=TRUE cair especificamente em glmQLFit,
# você poderá repetir de forma DIAGNÓSTICA com:
#
#   EDGER_ROBUST=false Rscript bin1_mutation/05_diagnostico_rota2_edger.R
#
# robust=FALSE NÃO deve ser adotado como análise final só para
# "fazer funcionar"; serve para localizar a origem do problema.
# ============================================================

options(
    warn = 1
)

checkpoint <- function(msg) {
    cat(
        sprintf(
            "\n[CHECKPOINT OK] %s\n",
            msg
        )
    )
    flush.console()
}

fail <- function(msg) {
    stop(
        msg,
        call. = FALSE
    )
}

args <- commandArgs(
    trailingOnly = FALSE
)

file_arg <- grep(
    "^--file=",
    args,
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

output_file <- file.path(
    base_dir,
    "rota2_edgeR_resultado_Rscript.csv"
)

design_file <- file.path(
    base_dir,
    "rota2_design_matrix_Rscript.csv"
)

session_file <- file.path(
    base_dir,
    "rota2_R_sessionInfo.txt"
)

robust_env <- tolower(
    Sys.getenv(
        "EDGER_ROBUST",
        unset = "true"
    )
)

robust_flag <- robust_env %in% c(
    "true",
    "1",
    "yes",
    "y"
)

cat(
    "============================================================\n"
)
cat(
    "DIAGNÓSTICO edgeR PURO — ROTA 2\n"
)
cat(
    "============================================================\n"
)
cat(
    "robust = ",
    robust_flag,
    "\n",
    sep = ""
)
cat(
    "counts = ",
    counts_file,
    "\n",
    sep = ""
)
cat(
    "metadata = ",
    metadata_file,
    "\n",
    sep = ""
)
flush.console()

# ------------------------------------------------------------
# 1. Pacotes
# ------------------------------------------------------------

suppressPackageStartupMessages(
    library(
        edgeR
    )
)

suppressPackageStartupMessages(
    library(
        limma
    )
)

checkpoint(
    paste0(
        "pacotes carregados | edgeR=",
        as.character(
            packageVersion(
                "edgeR"
            )
        ),
        " | limma=",
        as.character(
            packageVersion(
                "limma"
            )
        )
    )
)

# ------------------------------------------------------------
# 2. Leitura dos dados
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
        "A matriz de counts contém NA."
    )
}

if (any(counts < 0)) {
    fail(
        "A matriz de counts contém valores negativos."
    )
}

if (!all(
    colnames(counts)
    %in%
    rownames(metadata)
)) {
    fail(
        "Há doadores dos counts ausentes nos metadados."
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
        "A ordem donor counts/metadata não pôde ser alinhada."
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
# 3. Fatores EXPLÍCITOS
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
            is.na(
                metadata
            )
        )
    )

    fail(
        "NAs apareceram após definir os níveis dos fatores."
    )
}

checkpoint(
    "fatores definidos"
)

# ------------------------------------------------------------
# 4. model.matrix
#
# Pertpy 0.10.0 adiciona '+ 0' quando model_contrasts é usado.
# ------------------------------------------------------------

design <- model.matrix(
    ~ Modelo_Recessivo
    + Age_at_death
    + Braak_stage
    + Thal_phase
    + 0,
    data = metadata
)

write.csv(
    design,
    design_file
)

rank_design <- qr(
    design
)$rank

cat(
    "\nDesign matrix:\n"
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
cat(
    "colunas:\n"
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
        "Design matrix não possui rank completo."
    )
}

checkpoint(
    "model.matrix"
)

# ------------------------------------------------------------
# 5. Lib sizes / filtros equivalentes ao Pertpy
# ------------------------------------------------------------

lib_size <- colSums(
    counts
)

keep_smp <- (
    lib_size > 0
)

if (!all(keep_smp)) {
    cat(
        "Amostras com library size zero serão removidas.\n"
    )
}

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

checkpoint(
    paste0(
        "filtros | nhoods=",
        nrow(counts_use),
        " | samples=",
        ncol(counts_use)
    )
)

# ------------------------------------------------------------
# 6. DGEList
# ------------------------------------------------------------

dge <- DGEList(
    counts = counts_use,
    lib.size = lib_size_use
)

checkpoint(
    "DGEList"
)

# ------------------------------------------------------------
# 7. calcNormFactors
# ------------------------------------------------------------

dge <- calcNormFactors(
    dge,
    method = "TMM"
)

checkpoint(
    "calcNormFactors(TMM)"
)

# ------------------------------------------------------------
# 8. estimateDisp
# ------------------------------------------------------------

cat(
    "\n[INICIANDO] estimateDisp\n"
)
flush.console()

dge <- estimateDisp(
    dge,
    design_use
)

checkpoint(
    "estimateDisp"
)

# ------------------------------------------------------------
# 9. glmQLFit
# ------------------------------------------------------------

cat(
    "\n[INICIANDO] glmQLFit(robust=",
    robust_flag,
    ")\n",
    sep = ""
)
flush.console()

fit <- glmQLFit(
    dge,
    design_use,
    robust = robust_flag
)

checkpoint(
    paste0(
        "glmQLFit(robust=",
        robust_flag,
        ")"
    )
)

# ------------------------------------------------------------
# 10. Contraste
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
# 11. glmQLFTest
# ------------------------------------------------------------

cat(
    "\n[INICIANDO] glmQLFTest\n"
)
flush.console()

qlf <- glmQLFTest(
    fit,
    contrast = contrast
)

checkpoint(
    "glmQLFTest"
)

# ------------------------------------------------------------
# 12. topTags
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
    output_file,
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
        output_file
    )
)

cat(
    "\n============================================================\n"
)
cat(
    "DIAGNÓSTICO ROTA 2 CONCLUÍDO SEM CRASH\n"
)
cat(
    "============================================================\n"
)