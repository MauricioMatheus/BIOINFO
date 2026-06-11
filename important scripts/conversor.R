library(Seurat)
library(SeuratDisk)

objeto_microglia <- readRDS("Microglial.rds")
SaveH5Seurat(objeto_microglia, filename = "Microglial.h5Seurat")
Convert("Microglial.h5Seurat", dest = "h5ad")
