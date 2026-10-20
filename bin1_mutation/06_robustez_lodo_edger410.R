# ============================================================
# 06_robustez_lodo_edger410.R
#
# ROBUSTEZ MILO — LEAVE-ONE-DONOR-OUT (LODO)
# Projeto BIN1 / micróglia
#
# EXECUTAR NO AMBIENTE:
#   conda activate edger_410
#
# MODOS:
#   Rscript bin1_mutation/06_robustez_lodo_edger410.R critical
#   Rscript bin1_mutation/06_robustez_lodo_edger410.R complete
#
# O script NÃO reconstrói neighborhoods, PCA, KNN, Leiden ou UMAP.
# Ele reutiliza a matriz congelada neighborhood x donor já exportada.
#
# critical:
#   - remove H21.33.008
#   - remove H21.33.042
#   - remove ambos
#   - executa G0, G1 e G2 em cada cenário (9 ajustes)
#
# complete:
#   - remove um dos 75 doadores por vez
#   - executa G0, G1 e G2 (225 ajustes)
#
# Cada cenário recalcula TMM, dispersões, glmQLFit e glmQLFTest.
# Neighborhoods que ficarem com soma zero após exclusão são marcadas
# como não testáveis naquela iteração, em vez de provocar falha.
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

sanitizar <- function(x) {
    gsub("[^A-Za-z0-9]+", "_", x)
}

# ============================================================
# 1. ARGUMENTOS E CAMINHOS
# ============================================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
    fail("Informe o modo: critical ou complete.")
}

modo <- tolower(args[1])

if (!(modo %in% c("critical", "complete"))) {
    fail("Modo inválido. Use: critical ou complete.")
}

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
    "milo_final_edger410_reuniao"
)

export_dir <- file.path(base_dir, "export")
lodo_dir <- file.path(base_dir, "lodo", modo)
edger_dir <- file.path(lodo_dir, "edger")
audit_dir <- file.path(lodo_dir, "auditoria")

for (d in c(lodo_dir, edger_dir, audit_dir)) {
    dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

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
cat("ROBUSTEZ MILO — LODO — edgeR 4.10.x\n")
cat("Modo: ", modo, "\n", sep = "")
cat("============================================================\n")

suppressPackageStartupMessages(library(edgeR))
suppressPackageStartupMessages(library(limma))

edgeR_version <- packageVersion("edgeR")
limma_version <- packageVersion("limma")

cat("R: ", R.version.string, "\n", sep = "")
cat("edgeR: ", as.character(edgeR_version), "\n", sep = "")
cat("limma: ", as.character(limma_version), "\n", sep = "")

if (edgeR_version < package_version("4.10.0")) {
    fail(
        paste0(
            "edgeR >= 4.10.0 é obrigatório. Encontrado: ",
            edgeR_version
        )
    )
}

checkpoint("versão do edgeR validada")

# ============================================================
# 3. CARREGAMENTO DA MATRIZ CONGELADA
# ============================================================

if (!file.exists(counts_file)) {
    fail(paste("Counts não encontrados:", counts_file))
}

if (!file.exists(metadata_file)) {
    fail(paste("Metadata não encontrados:", metadata_file))
}

counts_df <- read.delim(
    counts_file,
    row.names = 1,
    check.names = FALSE
)

metadata_raw <- read.delim(
    metadata_file,
    row.names = 1,
    check.names = FALSE,
    stringsAsFactors = FALSE
)

counts <- as.matrix(counts_df)
storage.mode(counts) <- "integer"

if (nrow(counts) != 2201 || ncol(counts) != 75) {
    fail(
        paste0(
            "Dimensão de counts inesperada: ",
            nrow(counts), "x", ncol(counts)
        )
    )
}

if (anyNA(counts) || any(counts < 0)) {
    fail("Counts contêm NA ou valores negativos.")
}

if (!all(colnames(counts) %in% rownames(metadata_raw))) {
    fail("Há donors de counts ausentes em metadata.")
}

metadata_raw <- metadata_raw[
    colnames(counts),
    ,
    drop = FALSE
]

if (!identical(colnames(counts), rownames(metadata_raw))) {
    fail("Ordem donor counts/metadata incompatível.")
}

checkpoint("counts 2201x75 e metadata alinhados")

# ============================================================
# 4. FUNÇÃO DE PREPARAÇÃO DE FATORES
# ============================================================

preparar_fatores <- function(df) {
    df$Modelo_Recessivo <- factor(
        df$Modelo_Recessivo,
        levels = c("Nao_Recessivo", "Recessivo")
    )

    df$Age_at_death <- factor(
        df$Age_at_death,
        levels = c("65_77", "78_89", "90_plus")
    )

    df$Sex <- factor(
        df$Sex,
        levels = c("Female", "Male")
    )

    df$ADNC <- factor(
        df$ADNC,
        levels = c("Low", "Intermediate", "High")
    )

    df$Braak_stage <- factor(
        df$Braak_stage,
        levels = c(
            "Braak_0",
            "Braak_II",
            "Braak_III",
            "Braak_IV",
            "Braak_V",
            "Braak_VI"
        )
    )

    df$Thal_phase <- factor(
        df$Thal_phase,
        levels = c(
            "Thal_1",
            "Thal_2",
            "Thal_3",
            "Thal_4",
            "Thal_5"
        )
    )

    if (anyNA(df)) {
        cat("\nNA por coluna após fatoração:\n")
        print(colSums(is.na(df)))
        fail("NAs após definição dos fatores.")
    }

    # Crucial para LODO: remove níveis que ficaram vazios depois
    # da exclusão de um doador (ex.: Braak_0 possui apenas 1 donor).
    df <- droplevels(df)

    return(df)
}

# ============================================================
# 5. ROTAS PRINCIPAIS DA REUNIÃO
# ============================================================

rotas <- list(
    RotaG0_BIN1 = list(
        nome = "ROTA G0: BIN1",
        formula = ~ Modelo_Recessivo + 0
    ),
    RotaG1_BIN1_Idade_Sexo = list(
        nome = "ROTA G1: BIN1 + IDADE + SEXO",
        formula = ~ Modelo_Recessivo + Age_at_death + Sex + 0
    ),
    RotaG2_BIN1_Idade_Sexo_Braak = list(
        nome = "ROTA G2: BIN1 + IDADE + SEXO + BRAAK",
        formula = ~ Modelo_Recessivo + Age_at_death + Sex + Braak_stage + 0
    )
)

# ============================================================
# 6. CENÁRIOS DE EXCLUSÃO
# ============================================================

if (modo == "critical") {
    cenarios <- list(
        sem_H21_33_008 = c("H21.33.008"),
        sem_H21_33_042 = c("H21.33.042"),
        sem_H21_33_008_e_H21_33_042 = c(
            "H21.33.008",
            "H21.33.042"
        )
    )
} else {
    donors <- rownames(metadata_raw)
    cenarios <- lapply(donors, function(d) c(d))
    names(cenarios) <- paste0("sem_", sanitizar(donors))
}

# Confirma que os donors críticos existem.
for (rem in unique(unlist(cenarios))) {
    if (!(rem %in% rownames(metadata_raw))) {
        fail(paste("Doador ausente no metadata:", rem))
    }
}

cat("Número de cenários: ", length(cenarios), "\n", sep = "")
cat("Número de rotas por cenário: ", length(rotas), "\n", sep = "")
cat(
    "Total máximo de ajustes: ",
    length(cenarios) * length(rotas),
    "\n",
    sep = ""
)

# ============================================================
# 7. EXECUÇÃO DE UM CENÁRIO / ROTA
# ============================================================

executar_ajuste <- function(cenario, removidos, prefixo, info) {
    scenario_dir <- file.path(edger_dir, cenario)
    scenario_audit_dir <- file.path(audit_dir, cenario)

    dir.create(
        scenario_dir,
        recursive = TRUE,
        showWarnings = FALSE
    )
    dir.create(
        scenario_audit_dir,
        recursive = TRUE,
        showWarnings = FALSE
    )

    output_file <- file.path(
        scenario_dir,
        paste0(prefixo, "_edgeR_raw.csv")
    )

    # Permite retomar uma execução longa sem recalcular arquivos já prontos.
    if (file.exists(output_file)) {
        existente <- read.csv(
            output_file,
            stringsAsFactors = FALSE,
            check.names = FALSE
        )

        if (nrow(existente) == 2201 && "PValue" %in% colnames(existente)) {
            cat(
                "\n[PULANDO] ", cenario, " / ", prefixo,
                " — arquivo já existe e possui 2201 neighborhoods.\n",
                sep = ""
            )

            return(
                data.frame(
                    Cenario = cenario,
                    Removidos = paste(removidos, collapse = ";"),
                    Rota = prefixo,
                    Nome = info$nome,
                    Status = "OK_EXISTENTE",
                    N_doadores = 75 - length(removidos),
                    N_testaveis = sum(existente$Testavel %in% c(TRUE, "TRUE"), na.rm = TRUE),
                    N_nao_testaveis = sum(!(existente$Testavel %in% c(TRUE, "TRUE")), na.rm = TRUE),
                    Design_ncol = NA_integer_,
                    Design_rank = NA_integer_,
                    EdgeR = as.character(edgeR_version),
                    Limma = as.character(limma_version),
                    Mensagem = "Arquivo pré-existente reutilizado.",
                    stringsAsFactors = FALSE
                )
            )
        }
    }

    cat("\n============================================================\n")
    cat("CENÁRIO: ", cenario, "\n", sep = "")
    cat("REMOVIDO(S): ", paste(removidos, collapse = ", "), "\n", sep = "")
    cat(info$nome, "\n")
    cat("============================================================\n")

    keep_donors <- !(colnames(counts) %in% removidos)

    counts_use <- counts[
        ,
        keep_donors,
        drop = FALSE
    ]

    metadata_use <- metadata_raw[
        colnames(counts_use),
        ,
        drop = FALSE
    ]

    metadata_use <- preparar_fatores(metadata_use)

    # Ambos os grupos genéticos precisam continuar representados.
    if (
        nlevels(metadata_use$Modelo_Recessivo) != 2 ||
        any(table(metadata_use$Modelo_Recessivo) == 0)
    ) {
        return(
            data.frame(
                Cenario = cenario,
                Removidos = paste(removidos, collapse = ";"),
                Rota = prefixo,
                Nome = info$nome,
                Status = "FALHA_GRUPO_GENETICO",
                N_doadores = ncol(counts_use),
                N_testaveis = NA_integer_,
                N_nao_testaveis = NA_integer_,
                Design_ncol = NA_integer_,
                Design_rank = NA_integer_,
                EdgeR = as.character(edgeR_version),
                Limma = as.character(limma_version),
                Mensagem = "Um dos grupos genéticos ficou vazio.",
                stringsAsFactors = FALSE
            )
        )
    }

    design <- tryCatch(
        model.matrix(info$formula, data = metadata_use),
        error = function(e) e
    )

    if (inherits(design, "error")) {
        return(
            data.frame(
                Cenario = cenario,
                Removidos = paste(removidos, collapse = ";"),
                Rota = prefixo,
                Nome = info$nome,
                Status = "FALHA_MODEL_MATRIX",
                N_doadores = ncol(counts_use),
                N_testaveis = NA_integer_,
                N_nao_testaveis = NA_integer_,
                Design_ncol = NA_integer_,
                Design_rank = NA_integer_,
                EdgeR = as.character(edgeR_version),
                Limma = as.character(limma_version),
                Mensagem = conditionMessage(design),
                stringsAsFactors = FALSE
            )
        )
    }

    rank_design <- qr(design)$rank

    cat(
        "Design: ", nrow(design), " x ", ncol(design),
        " | rank=", rank_design, "\n",
        sep = ""
    )

    if (rank_design != ncol(design)) {
        return(
            data.frame(
                Cenario = cenario,
                Removidos = paste(removidos, collapse = ";"),
                Rota = prefixo,
                Nome = info$nome,
                Status = "FALHA_RANK",
                N_doadores = ncol(counts_use),
                N_testaveis = NA_integer_,
                N_nao_testaveis = NA_integer_,
                Design_ncol = ncol(design),
                Design_rank = rank_design,
                EdgeR = as.character(edgeR_version),
                Limma = as.character(limma_version),
                Mensagem = "Design rank-deficient após exclusão.",
                stringsAsFactors = FALSE
            )
        )
    }

    write.csv(
        design,
        file.path(
            scenario_audit_dir,
            paste0(prefixo, "_design_matrix.csv")
        )
    )

    lib_size <- colSums(counts_use)

    if (any(lib_size <= 0)) {
        return(
            data.frame(
                Cenario = cenario,
                Removidos = paste(removidos, collapse = ";"),
                Rota = prefixo,
                Nome = info$nome,
                Status = "FALHA_LIBRARY_SIZE",
                N_doadores = ncol(counts_use),
                N_testaveis = NA_integer_,
                N_nao_testaveis = NA_integer_,
                Design_ncol = ncol(design),
                Design_rank = rank_design,
                EdgeR = as.character(edgeR_version),
                Limma = as.character(limma_version),
                Mensagem = "Há donor com library size zero.",
                stringsAsFactors = FALSE
            )
        )
    }

    keep_nhoods <- rowSums(counts_use) > 0

    counts_fit <- counts_use[
        keep_nhoods,
        ,
        drop = FALSE
    ]

    if (nrow(counts_fit) == 0) {
        return(
            data.frame(
                Cenario = cenario,
                Removidos = paste(removidos, collapse = ";"),
                Rota = prefixo,
                Nome = info$nome,
                Status = "FALHA_SEM_NHOODS",
                N_doadores = ncol(counts_use),
                N_testaveis = 0L,
                N_nao_testaveis = 2201L,
                Design_ncol = ncol(design),
                Design_rank = rank_design,
                EdgeR = as.character(edgeR_version),
                Limma = as.character(limma_version),
                Mensagem = "Nenhuma neighborhood testável.",
                stringsAsFactors = FALSE
            )
        )
    }

    inicio(paste(cenario, prefixo, "DGEList/TMM/estimateDisp"))

    fit_result <- tryCatch({
        dge <- DGEList(
            counts = counts_fit,
            lib.size = lib_size
        )

        dge <- normLibSizes(
            dge,
            method = "TMM"
        )

        dge <- estimateDisp(
            dge,
            design
        )

        fit <- glmQLFit(
            dge,
            design,
            robust = TRUE,
            legacy = FALSE
        )

        contrast <- makeContrasts(
            Modelo_RecessivoRecessivo - Modelo_RecessivoNao_Recessivo,
            levels = design
        )

        qlf <- glmQLFTest(
            fit,
            contrast = contrast
        )

        tabela <- topTags(
            qlf,
            sort.by = "none",
            n = Inf
        )$table

        list(
            tabela = tabela,
            common_dispersion = dge$common.dispersion
        )
    }, error = function(e) e)

    if (inherits(fit_result, "error")) {
        return(
            data.frame(
                Cenario = cenario,
                Removidos = paste(removidos, collapse = ";"),
                Rota = prefixo,
                Nome = info$nome,
                Status = "FALHA_EDGER",
                N_doadores = ncol(counts_use),
                N_testaveis = sum(keep_nhoods),
                N_nao_testaveis = sum(!keep_nhoods),
                Design_ncol = ncol(design),
                Design_rank = rank_design,
                EdgeR = as.character(edgeR_version),
                Limma = as.character(limma_version),
                Mensagem = conditionMessage(fit_result),
                stringsAsFactors = FALSE
            )
        )
    }

    tabela <- fit_result$tabela

    if (!identical(rownames(tabela), rownames(counts_fit))) {
        fail(
            paste(
                cenario,
                prefixo,
                "alterou a ordem das neighborhoods testáveis."
            )
        )
    }

    # Reconstrói uma tabela com TODAS as 2201 neighborhoods.
    # As que ficaram zeradas após a exclusão recebem NA e Testavel=FALSE.
    resultado <- data.frame(
        Neighborhood = rownames(counts),
        logFC = NA_real_,
        logCPM = NA_real_,
        F = NA_real_,
        PValue = NA_real_,
        FDR = NA_real_,
        Testavel = keep_nhoods,
        Motivo_nao_testavel = ifelse(
            keep_nhoods,
            "",
            "Soma_zero_apos_exclusao"
        ),
        stringsAsFactors = FALSE
    )

    idx <- match(
        rownames(tabela),
        resultado$Neighborhood
    )

    resultado$logFC[idx] <- tabela$logFC
    resultado$logCPM[idx] <- tabela$logCPM
    resultado$F[idx] <- tabela$F
    resultado$PValue[idx] <- tabela$PValue
    resultado$FDR[idx] <- tabela$FDR

    resultado$Cenario <- cenario
    resultado$Doadores_removidos <- paste(removidos, collapse = ";")
    resultado$N_doadores_modelo <- ncol(counts_use)
    resultado$Rota <- prefixo
    resultado$EdgeR_version <- as.character(edgeR_version)
    resultado$Limma_version <- as.character(limma_version)
    resultado$Robust <- TRUE
    resultado$Legacy <- FALSE

    write.csv(
        resultado,
        output_file,
        row.names = FALSE
    )

    write.csv(
        data.frame(
            Cenario = cenario,
            Rota = prefixo,
            CommonDispersion = fit_result$common_dispersion,
            N_testaveis = sum(keep_nhoods),
            N_nao_testaveis = sum(!keep_nhoods)
        ),
        file.path(
            scenario_audit_dir,
            paste0(prefixo, "_dispersion_summary.csv")
        ),
        row.names = FALSE
    )

    checkpoint(
        paste(
            cenario,
            prefixo,
            "resultado salvo"
        )
    )

    return(
        data.frame(
            Cenario = cenario,
            Removidos = paste(removidos, collapse = ";"),
            Rota = prefixo,
            Nome = info$nome,
            Status = "OK",
            N_doadores = ncol(counts_use),
            N_testaveis = sum(keep_nhoods),
            N_nao_testaveis = sum(!keep_nhoods),
            Design_ncol = ncol(design),
            Design_rank = rank_design,
            EdgeR = as.character(edgeR_version),
            Limma = as.character(limma_version),
            Mensagem = "",
            stringsAsFactors = FALSE
        )
    )
}

# ============================================================
# 8. LOOP PRINCIPAL
# ============================================================

status_list <- list()
contador <- 0L
total <- length(cenarios) * length(rotas)

for (cenario in names(cenarios)) {
    removidos <- cenarios[[cenario]]

    for (prefixo in names(rotas)) {
        contador <- contador + 1L

        cat(
            "\n[PROGRESSO] ", contador, "/", total,
            " — ", cenario, " / ", prefixo, "\n",
            sep = ""
        )

        status_list[[length(status_list) + 1L]] <- executar_ajuste(
            cenario = cenario,
            removidos = removidos,
            prefixo = prefixo,
            info = rotas[[prefixo]]
        )

        gc(verbose = FALSE)
    }
}

status_df <- do.call(rbind, status_list)

write.csv(
    status_df,
    file.path(lodo_dir, "status_edgeR.csv"),
    row.names = FALSE
)

writeLines(
    capture.output(sessionInfo()),
    file.path(audit_dir, "LODO_edgeR_sessionInfo.txt")
)

cat("\n============================================================\n")
cat("LODO edgeR CONCLUÍDO\n")
cat("Modo: ", modo, "\n", sep = "")
cat("Resultados: ", edger_dir, "\n", sep = "")
cat("Status: ", file.path(lodo_dir, "status_edgeR.csv"), "\n", sep = "")
cat("============================================================\n")

print(status_df)