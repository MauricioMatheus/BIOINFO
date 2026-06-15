# Pega os argumentos passados no terminal
args <- commandArgs(trailingOnly = TRUE)
arquivo_entrada <- args[1]

cat("-> Carregando objeto:", arquivo_entrada, "\n")
obj <- readRDS(arquivo_entrada)

# TRAVA DE SEGURANÇA 1: O "Inspetor de Impostores"
# Checa a identidade estrutural do objeto antes de tentar extrair
if (inherits(obj, "Seurat")) {
    library(Seurat)
    library(Matrix)
    assay_name <- DefaultAssay(obj)
    
    # Costura camadas fragmentadas do Seurat v5 (comum após integração)
    try({ obj <- JoinLayers(obj) }, silent = TRUE)
    
    cat("-> Objeto Seurat detectado. Extraindo RAW COUNTS do assay:", assay_name, "\n")
    matriz <- tryCatch({
        GetAssayData(obj, assay = assay_name, layer = "counts")
    }, error = function(e) {
        GetAssayData(obj, assay = assay_name, slot = "counts")
    })
    
    metadados <- obj@meta.data

} else if (inherits(obj, "SingleCellExperiment")) {
    library(SingleCellExperiment)
    library(Matrix)
    
    cat("-> Objeto SingleCellExperiment detectado. Extraindo RAW COUNTS...\n")
    matriz <- counts(obj)
    metadados <- as.data.frame(colData(obj))

} else {
    stop("ERRO FATAL: Formato alienígena detectado! O arquivo não é Seurat nem SingleCellExperiment.")
}

# TRAVA DE SEGURANÇA 2: A "Prensa Compressora"
# Força qualquer matriz (mesmo que o pesquisador tenha salvo errado) a voltar para o formato Esparso
cat("-> Garantindo compressão matemática (CsparseMatrix)...\n")
matriz <- as(matriz, "CsparseMatrix")

cat("-> Exportando matriz e metadados...\n")
writeMM(matriz, file = "temp_matriz.mtx")
write.table(colnames(matriz), file = "temp_celulas.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(rownames(matriz), file = "temp_genes.csv", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.csv(metadados, file = "temp_metadados.csv")

cat("-> Extração finalizada com sucesso!\n")