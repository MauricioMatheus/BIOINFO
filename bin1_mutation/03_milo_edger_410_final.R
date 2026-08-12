# ============================================================
# 03_milo_edger_410_final.R
#
# MILO FINAL — ETAPA 2
# edgeR 4.10.x para as três rotas
#
# EXECUTAR NO AMBIENTE:
#   conda activate edger_410
#
# O script:
#   - exige edgeR >= 4.10.0;
#   - usa exatamente a matriz de 2201 neighborhoods exportada;
#   - executa as 3 rotas com:
#       robust = TRUE
#       legacy = FALSE
#   - usa TMM, estimateDisp, glmQLFit e glmQLFTest;
#   - salva imediatamente cada rota;
#   - NÃO calcula SpatialFDR (isso será feito na Etapa 3 em Python
#     reproduzindo a função do Pertpy 0.10.0).
# ============================================================

options(warn = 1)

checkpoint <- function(msg) {
    cat(
        sprintf(
            "\n[CHECKPOINT OK] %s\n",
            msg
        )
    )
    flush.console()
}

inicio <- function(msg) {
    cat(
        sprintf(
            "\n[INICIANDO] %s\n",
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


# ============================================================
# 1. CAMINHOS
# ============================================================

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
    "milo_final_edger410"
)

export_dir <- file.path(
    base_dir,
    "export"
)

edger_dir <- file.path(
    base_dir,
    "edger"
)

audit_dir <- file.path(
    base_dir,
    "auditoria"
)

dir.create(
    edger_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    audit_dir,
    recursive = TRUE,
    showWarnings = FALSE
)

counts_file <- file.path(
    export_dir,
    "milo_counts_neighborhood_x_donor.tsv"
)

metadata_file <- file.path(
    export_dir,
    "milo_metadata_donor.tsv"
)


# ============================================================
# 2. PACOTES
# ============================================================

cat("============================================================\n")
cat("MILO FINAL — ETAPA 2 — edgeR 4.10.x\n")
cat("============================================================\n")

suppressPackageStartupMessages(
    library(edgeR)
)

suppressPackageStartupMessages(
    library(limma)
)

edgeR_version <- packageVersion(
    "edgeR"
)

limma_version <- packageVersion(
    "limma"
)

cat(
    "R: ",
    R.version.string,
    "\n",
    sep = ""
)

cat(
    "edgeR: ",
    as.character(
        edgeR_version
    ),
    "\n",
    sep = ""
)

cat(
    "limma: ",
    as.character(
        limma_version
    ),
    "\n",
    sep = ""
)

if (
    edgeR_version
    < package_version(
        "4.10.0"
    )
) {
    fail(
        paste0(
            "edgeR >= 4.10.0 é obrigatório. Encontrado: ",
            edgeR_version
        )
    )
}

checkpoint(
    "versão do edgeR validada"
)


# ============================================================
# 3. DADOS
# ============================================================

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
            "Metadata não encontrados:",
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

if (
    nrow(counts) != 2201
    || ncol(counts) != 75
) {
    fail(
        paste0(
            "Dimensão de counts inesperada: ",
            nrow(counts),
            "x",
            ncol(counts)
        )
    )
}

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

if (!all(
    colnames(counts)
    %in%
    rownames(metadata)
)) {
    fail(
        "Há donors de counts ausentes em metadata."
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
        "Ordem donor counts/metadata incompatível."
    )
}

checkpoint(
    "counts 2201x75 e metadata alinhados"
)


# ============================================================
# 4. FATORES
# ============================================================

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

metadata$ADNC <- factor(
    metadata$ADNC,
    levels = c(
        "Low",
        "Intermediate",
        "High"
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
    cat("\nNA por coluna:\n")
    print(
        colSums(
            is.na(metadata)
        )
    )

    fail(
        "NAs após definição dos fatores."
    )
}

checkpoint(
    "fatores definidos"
)


# ============================================================
# 5. ROTAS
# ============================================================

rotas <- list(
    Rota0_BIN1_Idade = list(
        nome = "ROTA 0: BIN1 + IDADE",
        formula = (
            ~ Modelo_Recessivo
            + Age_at_death
            + 0
        )
    ),
    Rota1_BIN1_Idade_ADNC = list(
        nome = "ROTA 1: BIN1 + IDADE + ADNC",
        formula = (
            ~ Modelo_Recessivo
            + Age_at_death
            + ADNC
            + 0
        )
    ),
    Rota2_BIN1_Idade_Braak_Thal = list(
        nome = "ROTA 2: BIN1 + IDADE + BRAAK + THAL",
        formula = (
            ~ Modelo_Recessivo
            + Age_at_death
            + Braak_stage
            + Thal_phase
            + 0
        )
    )
)

# ============================================================
# 6. FUNÇÃO DE EXECUÇÃO
# ============================================================

executar_rota <- function(
    prefixo,
    info
) {
    cat("\n")
    cat("============================================================\n")
    cat(info$nome, "\n")
    cat("============================================================\n")

    design <- model.matrix(
        info$formula,
        data = metadata
    )

    rank_design <- qr(
        design
    )$rank

    cat(
        "Design: ",
        nrow(design),
        " x ",
        ncol(design),
        " | rank=",
        rank_design,
        "\n",
        sep = ""
    )

    cat("Colunas:\n")
    print(
        colnames(design)
    )
    flush.console()

    if (
        rank_design
        != ncol(design)
    ) {
        fail(
            paste(
                prefixo,
                "possui design rank-deficient."
            )
        )
    }

    write.csv(
        design,
        file.path(
            audit_dir,
            paste0(
                prefixo,
                "_design_matrix.csv"
            )
        )
    )

    lib_size <- colSums(
        counts
    )

    keep_smp <- (
        lib_size > 0
    )

    if (!all(keep_smp)) {
        fail(
            paste(
                prefixo,
                "possui donor com library size zero."
            )
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

    if (!all(keep_nhoods)) {
        fail(
            paste(
                prefixo,
                "possui neighborhood com soma zero."
            )
        )
    }

    checkpoint(
        paste(
            prefixo,
            "model.matrix e filtros"
        )
    )

    dge <- DGEList(
        counts = counts_use,
        lib.size = lib_size_use
    )

    checkpoint(
        paste(
            prefixo,
            "DGEList"
        )
    )

    # normLibSizes é o nome atual da antiga calcNormFactors.
    dge <- normLibSizes(
        dge,
        method = "TMM"
    )

    checkpoint(
        paste(
            prefixo,
            "TMM / normLibSizes"
        )
    )

    inicio(
        paste(
            prefixo,
            "estimateDisp"
        )
    )

    dge <- estimateDisp(
        dge,
        design_use
    )

    checkpoint(
        paste(
            prefixo,
            "estimateDisp"
        )
    )

    disp_summary <- data.frame(
        Rota = prefixo,
        CommonDispersion = dge$common.dispersion,
        Trended_min = min(
            dge$trended.dispersion,
            na.rm = TRUE
        ),
        Trended_median = median(
            dge$trended.dispersion,
            na.rm = TRUE
        ),
        Trended_max = max(
            dge$trended.dispersion,
            na.rm = TRUE
        ),
        Tagwise_min = min(
            dge$tagwise.dispersion,
            na.rm = TRUE
        ),
        Tagwise_median = median(
            dge$tagwise.dispersion,
            na.rm = TRUE
        ),
        Tagwise_max = max(
            dge$tagwise.dispersion,
            na.rm = TRUE
        )
    )

    write.csv(
        disp_summary,
        file.path(
            audit_dir,
            paste0(
                prefixo,
                "_dispersion_summary.csv"
            )
        ),
        row.names = FALSE
    )

    inicio(
        paste(
            prefixo,
            "glmQLFit robust=TRUE legacy=FALSE"
        )
    )

    fit <- glmQLFit(
        dge,
        design_use,
        robust = TRUE,
        legacy = FALSE
    )

    checkpoint(
        paste(
            prefixo,
            "glmQLFit"
        )
    )

    contrast <- makeContrasts(
        Modelo_RecessivoRecessivo
        - Modelo_RecessivoNao_Recessivo,
        levels = design_use
    )

    checkpoint(
        paste(
            prefixo,
            "makeContrasts"
        )
    )

    inicio(
        paste(
            prefixo,
            "glmQLFTest"
        )
    )

    qlf <- glmQLFTest(
        fit,
        contrast = contrast
    )

    checkpoint(
        paste(
            prefixo,
            "glmQLFTest"
        )
    )

    resultado <- topTags(
        qlf,
        sort.by = "none",
        n = Inf
    )$table

    if (
        nrow(resultado)
        != 2201
    ) {
        fail(
            paste(
                prefixo,
                "retornou número inesperado de neighborhoods."
            )
        )
    }

    if (!identical(
        rownames(resultado),
        rownames(counts_use)
    )) {
        fail(
            paste(
                prefixo,
                "alterou a ordem das neighborhoods."
            )
        )
    }

    resultado$Neighborhood <- rownames(
        resultado
    )

    resultado <- resultado[
        ,
        c(
            "Neighborhood",
            setdiff(
                colnames(resultado),
                "Neighborhood"
            )
        )
    ]

    resultado$EdgeR_version <- as.character(
        edgeR_version
    )

    resultado$Limma_version <- as.character(
        limma_version
    )

    resultado$Robust <- TRUE
    resultado$Legacy <- FALSE

    output_file <- file.path(
        edger_dir,
        paste0(
            prefixo,
            "_edgeR_raw.csv"
        )
    )

    write.csv(
        resultado,
        output_file,
        row.names = FALSE
    )

    checkpoint(
        paste(
            prefixo,
            "resultado salvo"
        )
    )

    summary_row <- data.frame(
        Rota = prefixo,
        Nome = info$nome,
        N_neighborhoods = nrow(resultado),
        Min_PValue = min(
            resultado$PValue,
            na.rm = TRUE
        ),
        Min_FDR = min(
            resultado$FDR,
            na.rm = TRUE
        ),
        LogFC_min = min(
            resultado$logFC,
            na.rm = TRUE
        ),
        LogFC_max = max(
            resultado$logFC,
            na.rm = TRUE
        ),
        N_PValue_lt_0_05 = sum(
            resultado$PValue < 0.05,
            na.rm = TRUE
        ),
        N_FDR_lt_0_10 = sum(
            resultado$FDR < 0.10,
            na.rm = TRUE
        ),
        EdgeR = as.character(
            edgeR_version
        ),
        Limma = as.character(
            limma_version
        ),
        Robust = TRUE,
        Legacy = FALSE
    )

    return(
        summary_row
    )
}


# ============================================================
# 7. EXECUÇÃO DAS 3 ROTAS
# ============================================================

resumos <- list()

for (
    prefixo
    in names(rotas)
) {
    resumos[[prefixo]] <- executar_rota(
        prefixo,
        rotas[[prefixo]]
    )

    gc(
        verbose = FALSE
    )
}

resumo_final <- do.call(
    rbind,
    resumos
)

write.csv(
    resumo_final,
    file.path(
        edger_dir,
        "Resumo_edgeR_pre_SpatialFDR.csv"
    ),
    row.names = FALSE
)

writeLines(
    capture.output(
        sessionInfo()
    ),
    file.path(
        audit_dir,
        "Etapa2_edgeR_sessionInfo.txt"
    )
)

cat("\n")
print(
    resumo_final
)

cat("\n============================================================\n")
cat("ETAPA 2 CONCLUÍDA — AS 3 ROTAS edgeR FORAM SALVAS\n")
cat("============================================================\n")