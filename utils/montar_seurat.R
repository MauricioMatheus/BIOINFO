args <- commandArgs(trailingOnly = TRUE)
arquivo_saida <- args[1]

library(Seurat)
library(Matrix)

cat("-> Lendo arquivos brutos temporários...\n")
matriz_bruta <- readMM("temp_matriz.mtx")
celulas <- read.csv("temp_celulas.csv", header = FALSE, stringsAsFactors = FALSE)$V1
genes <- read.csv("temp_genes.csv", header = FALSE, stringsAsFactors = FALSE)$V1

# TRAVA DE SEGURANÇA 4: O Escudo Anti-Mutação do R (check.names = FALSE)
# Impede que o R altere secretamente traços (-) para pontos (.) nos nomes das células
metadados <- read.csv("temp_metadados.csv", row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)

# TRAVA DE SEGURANÇA 5: Inversão e Otimização de Memória Direta
# Transpõe a matriz (Python -> R) e força imediatamente para a classe dgCMatrix exigida pelo Seurat
cat("-> Transpondo e otimizando a matriz para CsparseMatrix...\n")
matriz <- as(t(matriz_bruta), "CsparseMatrix")

# TRAVA DE SEGURANÇA 6: Dupla Checagem de Unicidade
# Garante que, mesmo que algo passe, o R force a unicidade (ex: GeneA.1, GeneA.2)
colnames(matriz) <- make.unique(as.character(celulas))
rownames(matriz) <- make.unique(as.character(genes))
rownames(metadados) <- colnames(matriz) # Força o alinhamento exato

# TRAVA DE SEGURANÇA 7: Sincronização Matricial
# Garante que os metadados contenham APENAS as células que estão na matriz, na exata ordem
metadados <- metadados[colnames(matriz), , drop = FALSE]

cat(sprintf("-> Montando o objeto Seurat final em: %s\n", arquivo_saida))
# O parâmetro min.cells = 0 e min.features = 0 evita que o Seurat filtre os dados na entrada
obj <- CreateSeuratObject(counts = matriz, meta.data = metadados, min.cells = 0, min.features = 0)

saveRDS(obj, file = arquivo_saida)

cat("-> Limpando arquivos temporários...\n")
file.remove("temp_matriz.mtx")
file.remove("temp_celulas.csv")
file.remove("temp_genes.csv")
file.remove("temp_metadados.csv")

cat("-> Processo concluído com sucesso! Objeto perfeitamente recriado.\n")