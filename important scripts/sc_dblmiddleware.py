import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import logging
import rpy2.rinterface_lib.callbacks as rcb

# Suprime os avisos do R no terminal do Python
rcb.logger.setLevel(logging.ERROR)

def _run_scdblfinder_single(adata, random_seed):
    """Função interna que roda o R numa única matriz (um único lote)."""
    data_mat = adata.X.T.tocsc()
    x = data_mat.data.astype(np.float64)
    i = data_mat.indices.astype(np.int32)
    p = data_mat.indptr.astype(np.int32)
    dims = np.array(data_mat.shape, dtype=np.int32)

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["x"] = x
        ro.globalenv["i"] = i
        ro.globalenv["p"] = p
        ro.globalenv["dims"] = dims
        ro.globalenv["random_seed"] = random_seed

    ro.r('''
        suppressPackageStartupMessages(library(Matrix))
        suppressPackageStartupMessages(library(scDblFinder))
        suppressPackageStartupMessages(library(SingleCellExperiment))

        x <- as.numeric(x)
        i <- as.integer(i)
        p <- as.integer(p)
        dims <- as.integer(dims)

        data_mat <- new("dgCMatrix", Dim = dims, x = x, i = i, p = p)
        set.seed(random_seed)
        
        sce <- scDblFinder(SingleCellExperiment(list(counts = data_mat)))
        doublet_score <- sce$scDblFinder.score
        doublet_class <- sce$scDblFinder.class
    ''')

    adata.obs["scDblFinder_score"] = np.array(ro.globalenv['doublet_score'])
    adata.obs["scDblFinder_class"] = np.array(ro.globalenv['doublet_class'])
    return adata


def run_scdblfinder(adata, batch_key=None, random_seed=123):
    """
    Executa o scDblFinder de forma inteligente, iterando sobre os lotes 
    para evitar OOM (Out of Memory) e salvaguardar a biologia real.
    """
    if batch_key is None or batch_key not in adata.obs.columns:
        print("A executar scDblFinder em matriz única (Sem divisão de lotes)...")
        return _run_scdblfinder_single(adata, random_seed)
    
    print(f"Lotes detetados na coluna '{batch_key}'. A dividir o processamento...")
    
    # Prepara as colunas originais com valores nulos para receber os resultados
    adata.obs["scDblFinder_score"] = np.nan
    adata.obs["scDblFinder_class"] = "unknown"
    
    lotes = adata.obs[batch_key].unique()
    
    for lote in lotes:
        print(f"  -> A processar lote: {lote}")
        # Recorta apenas as células deste lote
        adata_subset = adata[adata.obs[batch_key] == lote].copy()
        
        # Roda o R apenas neste pedaço (salvaguarda a RAM)
        adata_subset = _run_scdblfinder_single(adata_subset, random_seed)
        
        # Mapeamento Direto: Injeta os resultados no adata original usando os nomes das células (índices)
        # Isto é infinitamente superior ao ad.concat() pois não apaga metadados!
        adata.obs.loc[adata_subset.obs_names, "scDblFinder_score"] = adata_subset.obs["scDblFinder_score"]
        adata.obs.loc[adata_subset.obs_names, "scDblFinder_class"] = adata_subset.obs["scDblFinder_class"]
        
    print("Processamento concluído. Dataset original atualizado com sucesso!")
    return adata


'''# Exemplo de aplicação com datasets diversos:

import scanpy as sc
# Importa o arquivo onde você salvou a função acima
import sc_middleware 

# ... (Seu código anterior carregando o adata e calculando MT/Ribo) ...

# Filtra genes não informativos
sc.pp.filter_genes(adata, min_cells=20)

# Chama a função middleware para dataset simples
adata = sc_middleware.run_scdblfinder(adata, random_seed=42)

#Chama a função middleware para dataset com vários lotes
adata = sc_middleware.run_scdblfinder(adata, batch_key="patient", random_seed=42)

# Verifica os resultados
print(adata.obs["scDblFinder_class"].value_counts())

# Remove os doublets mantendo apenas os "singlets"
adata = adata[adata.obs["scDblFinder_class"] == "singlet"].copy()

print(f"Número de células reais após a filtragem: {adata.n_obs}")'''