
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

counts = csr_matrix(
    np.random.default_rng().poisson(1, size=(100, 2000)), dtype=np.float32
)
# 100 células (linhas) e 2.000 genes (colunas). [exemplo criado]
adata = ad.AnnData(counts)

#print(adata.X)

adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]

adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]

#print(adata.var_names[:10])


#Adding aligned metadata
ct = np.random.default_rng().choice(["B", "T", "Monocyte"], size=(adata.n_obs,))
adata.obs["cell_type"] = pd.Categorical(ct) 
# Categoricals are preferred for efficiency
#print(adata.obs)
#print(adata)

bdata = adata[adata.obs.cell_type == "B"]
#print(bdata)

#Observation/variable-level matrices
#Preenchendo obsm e varm

adata.obsm["X_umap"] = np.random.default_rng().normal(0, 1, size=(adata.n_obs, 2))

adata.varm["gene_stuff"] = np.random.default_rng().normal(0, 1, size=(adata.n_vars, 5))


adata.uns["random"] = [1,2,3]
#print(adata.uns)

adata.layers["log_transformed"] = np.log1p(adata.X)

adata.to_df(layer="log_transformed")
#print(adata.to_df(layer="log_transformed"))

#Salvando no SSD/HD e não apenas na RAM
adata.write("my_results.h5ad", compression="gzip")

#lendo de volta para a RAM e guardando na variável
adata_new = ad.read_h5ad("my_results.h5ad")

#print(adata_new)


#Efficient data access
obs_meta = pd.DataFrame(
    {
        "time_yr": np.random.default_rng().choice([0, 2, 4, 8], adata.n_obs),
        
        "subject_id": np.random.default_rng().choice(["subject 1", "subject 2", "subject 4", "subject 8"], adata.n_obs),
        
        "instrument_type": np.random.default_rng().choice(
            ["type a", "type b"], adata.n_obs),
        
        "site": np.random.default_rng().choice(["site x", "site y"], adata.n_obs), 
    },
    index=adata.obs.index
)

#adata = ad.AnnData(adata.X, obs=obs_meta, var=adata.var) #O recomendado em casos reais é fazer um adata.obs = adata.obs.join(obs_meta)!

adata.obs = adata.obs.join(obs_meta)

#print (adata)

adata_view = adata[:5, ["Gene_1", "Gene_3"]]
#print(adata_view)


#Fazendo uma cópia e não um ponteiro
adata_subset = adata[:5, ["Gene_1", "Gene_3"]].copy()

#print(adata[:3, "Gene_1"].X.toarray().tolist())

#adata[:3, "Gene_1"].X = [0,0,0] 

#Alterando os valores de adata sem usar cópia

#print(adata[:3, "Gene_1"].X.toarray().tolist) #valores alterados


adata_subset = adata[:3, ["Gene_1", "Gene_2"]]

print(adata_subset)

adata_subset.obs["foo"] = range(3)

print(adata_subset)

adata[adata.obs.time_yr.isin([2, 4])].obs.head()