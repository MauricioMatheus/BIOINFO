import lamindb as ln
import numpy as np
import scanpy as sc
import seaborn as sns
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from scipy.sparse import csc_matrix
from scipy.stats import median_abs_deviation

#suppressing verbose logging from Scanpy
sc.settings.verbosity=0

sc.settings.set_figure_params(dpi=80, facecolor="white", frameon=False)

#assert ln.setup.settings.instance.slug == "theislab/sc-best-practices"
#ln.track()

#using LaminDB to connect to the tutorial server
af = ln.Artifact.connect("theislab/sc-best-practices").get(key="preprocessing_visualization/quality_control_adata.h5ad", is_latest=True)
adata = af.load()
#print(adata)
adata.var_names_make_unique()
print(adata)

# mitochondrial genes
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains((r"^HB[ABDEGMQZ]\d*(?!\w)"))

#Calculating the respective QC metrics with Scanpy

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
)
print(adata)

#plots
p1 = sns.displot(adata.obs["total_counts"], bins=100, kde=False)
p1.savefig("figures/distribution_total_counts.png", dpi=300)
p2 = sc.pl.violin(adata, "pct_counts_mt", save="_violin_pct_counts_mt_QCTutorial.png", show=False)
p3 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color = "pct_counts_mt", save="_scatter_QC.png", show=False)

#MAD function:
def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
)
adata.obs.outlier.value_counts()

#In this tutorial, pct_counts_Mt is filtered with 3 MADs.
#Additionally, cells with a percentage of mitochondrial counts
#exceeding 8 % are filtered out.


adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
    adata.obs["pct_counts_mt"] > 8
)
adata.obs.mt_outlier.value_counts()

print(f"Total number of cells: {adata.n_obs}")
adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

print(f"Number os cells after filtering of low quality cells: {adata.n_obs}")

#plot mt
p1 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", color="pct_counts_mt", save="_scatter_QC_pos_filtro.png", show=False)

#interoperability
import logging
import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
rcb.logger.setLevel(logging.ERROR)

#importanto a biblioteca SoupX do R para QC de Ambient RNA

ro.r('''
    library(SoupX)''')

adata_pp = adata.copy()
sc.pp.normalize_total(adata_pp, target_sum=1e4)
sc.pp.log1p(adata_pp) #Normalização

sc.pp.pca(adata_pp) #Compressão de dados
sc.pp.neighbors(adata_pp) #Rede social das células
sc.tl.leiden( #Agrupamento/Clustering
    adata_pp,
    key_added="soupx_groups",
    flavor="igraph",
    n_iterations=2,
    directed=False
)

# Preprocess variables for SoupX
adata.obs["soupx_groups"] = adata_pp.obs["soupx_groups"]

del adata_pp


#Transpondo a Matriz para o R

cells = adata.obs_names
genes = adata.var_names
data = adata.X.T

adata_raw = af.load()
adata_raw.var_names_make_unique()

genes_raw = adata_raw.var_names
cells_raw = adata_raw.obs_names

data_tod = adata_raw.X.T

del adata_raw

#Middleware (Ambient RNA)

# Converting the data into the right structure to be used in R.
# ETL (Extração, Transformação e Carga) da infraestrutura

data_csc = data.tocsc()
data_tod_csc = data_tod.tocsc()

# Extract sparse components and cast to correct types
x = data_csc.data.astype(np.float64)
i = data_csc.indices.astype(np.int32)
p = data_csc.indptr.astype(np.int32)
dims = np.array(data_csc.shape, dtype=np.int32)

x_tod = data_tod_csc.data.astype(np.float64)
i_tod = data_tod_csc.indices.astype(np.int32)
p_tod = data_tod_csc.indptr.astype(np.int32)
dims_tod = np.array(data_tod_csc.shape, dtype=np.int32)

with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
    ro.globalenv["x"] = x
    ro.globalenv["i"] = i
    ro.globalenv["p"] = p
    ro.globalenv["dims"] = dims

    ro.globalenv["x_tod"] = x_tod
    ro.globalenv["i_tod"] = i_tod
    ro.globalenv["p_tod"] = p_tod
    ro.globalenv["dims_tod"] = dims_tod

    ro.globalenv["genes"] = np.array(genes)
    ro.globalenv["genes_raw"] = np.array(genes_raw)
    ro.globalenv["cells"] = np.array(cells)
    ro.globalenv["cells_raw"] = np.array(cells_raw)
    ro.globalenv["soupx_groups"] = adata.obs["soupx_groups"].to_numpy()
    
#Execução do Algorítmo/Modelo

ro.r('''
    library(Matrix)
    
    # Manually coerce types to avoid "array" class errors
    x <- as.numeric(x)
    i <- as.integer(i)
    p <- as.integer(p)
    dims <- as.integer(dims)

    x_tod <- as.numeric(x_tod)
    i_tod <- as.integer(i_tod)
    p_tod <- as.integer(p_tod)
    dims_tod <- as.integer(dims_tod)

    # Reconstruct sparse matrices
    data <- new("dgCMatrix",
                Dim = dims,
                x = x,
                i = i,
                p = p)

    data_tod <- new("dgCMatrix",
                    Dim = dims_tod,
                    x = x_tod,
                    i = i_tod,
                    p = p_tod)

    # Assign row and column names
    rownames(data) <- genes
    colnames(data) <- cells
    rownames(data_tod) <- genes_raw
    colnames(data_tod) <- cells_raw

    # SoupX pipeline
    sc = SoupChannel(data_tod, data, calcSoupProfile = TRUE)
    sc = setClusters(sc, soupx_groups)
    sc = autoEstCont(sc, doPlot = FALSE)
    
    # Matriz final corrigida salva na memória do R
    out = adjustCounts(sc, roundToInt = TRUE)
''')

# 2. Resgatando a matriz corrigida ('out') de volta para o Python
# Isso substitui o "%%R -o out"

# caminho de volta (o Download)
with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
    out_py = ro.conversion.rpy2py(ro.globalenv["out"])

x = np.array(out_py.slots["x"])
i = np.array(out_py.slots["i"])
p = np.array(out_py.slots["p"])
shape = tuple(out_py.slots["Dim"])

out_matrix = csc_matrix((x, i, p), shape=shape)

# Trazendo de volta para python/Rotação final e substituição
adata.layers["counts"] = adata.X.copy()
adata.layers["soupX_counts"] = out_matrix.T
adata.X = adata.layers["soupX_counts"]

'''we additionally filter out genes that are not detected in at least 20 cells as these are not informative. This exclusion is necessary because genes detected in very few cells are often the result of technical noise, ambient RNA contamination, or stochastic low-level transcription, rather than true biological signal.'''

print(f"Total number of genes: {adata.n_vars}")

# Min 20 cells - filters out 0 count genes
sc.pp.filter_genes(adata, min_cells=20)
print(f"Number of genes after cell filter: {adata.n_vars}")

ro.r('''
     library(Seurat)
     library(scater)
     library(scDblFinder)
     library(SingleCellExperiment)
     library(BiocParallel)''')

data_mat = adata.X.T

#converting the matrix from python to R (Basically, a middleware - Doublet detection)

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
    
ro.r('''
    x <- as.numeric(x)
    i <- as.integer(i)
    p <- as.integer(p)
    dims <- as.integer(dims)

    data_mat <- new("dgCMatrix", Dim = dims, x = x, i = i, p = p)

    set.seed(123)
    sce <- scDblFinder(SingleCellExperiment(list(counts = data_mat)))

    # As variáveis são criadas e ficam na memória global do R
    doublet_score <- sce$scDblFinder.score
    doublet_class <- sce$scDblFinder.class '''
)

# 2. Resgatando as variáveis de volta para o Python
# Isso substitui completamente o "%%R -o doublet_score -o doublet_class"
doublet_score = ro.globalenv['doublet_score']
doublet_class = ro.globalenv['doublet_class']

print("Scores e classes recuperados com sucesso!")

# 3. Agora você acopla essas predições no seu dataset em Python
adata.obs["scDblFinder_score"] = doublet_score
adata.obs["scDblFinder_class"] = doublet_class

# Apenas para checar se deu tudo certo
print(adata.obs["scDblFinder_class"].value_counts())
adata.write_h5ad("Dataset after QC.h5ad")




