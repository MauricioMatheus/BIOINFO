# Pega os argumentos passados no terminal
args <- commandArgs(trailingOnly = TRUE)
arquivo_entrada <- args[1] # Ex: "Microglial.rds"

library(Seurat)
library(Matrix)

cat("-> Carregando objeto:", arquivo_entrada, "\n")
obj <- readRDS(arquivo_entrada)
assay_name <- DefaultAssay(obj)

cat("-> Extraindo dados do assay:", assay_name, "\n")
# O tryCatch garante que ele tente a versão nova (v5) e se falhar, tenta a velha (v4)
matriz <- tryCatch({
    GetAssayData(obj, assay = assay_name, layer = "data")
}, error = function(e) {
    GetAssayData(obj, assay = assay_name, slot = "data")
})

cat("-> Exportando matriz e metadados...\n")
writeMM(matriz, file = "temp_matriz.mtx")
write.table(colnames(matriz), file = "temp_celulas.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(rownames(matriz), file = "temp_genes.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.csv(obj@meta.data, file = "temp_metadados.csv")

cat("-> Extração finalizada com sucesso!\n")