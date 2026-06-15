library(Seurat)
library(Matrix)

# 1. Carrega o arquivo do seu professor
obj <- readRDS("Microglial.rds")
assay_name <- DefaultAssay(obj)

# 2. Puxa os counts brutos (com a trava de versão)
counts <- tryCatch({
    GetAssayData(obj, assay = assay_name, layer = "counts")
}, error = function(e) {
    GetAssayData(obj, assay = assay_name, slot = "counts")
})

# 3. Imprime a Impressão Digital Matemática
cat("=== AUDITORIA DO R ===\n")
cat("Total de Células (Colunas):", ncol(counts), "\n")
cat("Total de Genes (Linhas):", nrow(counts), "\n")
cat("Soma Absoluta de todo o RNA:", sum(counts), "\n")
cat("Maior pico de RNA em uma única célula:", max(counts), "\n")