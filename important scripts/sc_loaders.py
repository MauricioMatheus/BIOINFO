import os
import scanpy as sc
import pandas as pd
import scipy.io

def carregar_10x_padrao(pasta, arquivo_metadados=None, prefixo=""):
    """
    Usa o motor nativo do Scanpy. 
    Ideal para datasets perfeitos (features.tsv com 2+ colunas).
    """
    diretorio_original = os.getcwd()
    
    try:
        os.chdir(pasta)
        print(f"Carregando matriz padrão do diretório: '{pasta}'")
        
        # O Scanpy nativo já lida muito bem com prefixos se avisarmos ele
        adata = sc.read_10x_mtx(".", prefix=prefixo)
        adata.var_names_make_unique()
        
        if arquivo_metadados and os.path.exists(arquivo_metadados):
            print(f"Acoplando metadados de: {arquivo_metadados}")
            metadata = pd.read_csv(arquivo_metadados, sep="\t", index_col=0)
            adata.obs = metadata.reindex(adata.obs_names)
            
        print("Objeto AnnData montado com sucesso!\n")
        return adata
        
    finally:
        # Blinda o código: sempre volta para a pasta original, mesmo se der erro
        os.chdir(diretorio_original)


def carregar_10x_reparo(pasta, arquivo_metadados=None, prefixo=""):
    """
    Constrói o AnnData do zero.
    Ideal para contornar o bug de datasets onde o features.tsv tem apenas 1 coluna.
    """
    diretorio_original = os.getcwd()
    
    try:
        os.chdir(pasta)
        print(f"Montando matriz manualmente do diretório: '{pasta}'")
        
        nome_matriz = f"{prefixo}matrix.mtx"
        nome_celulas = f"{prefixo}barcodes.tsv"
        nome_genes = f"{prefixo}features.tsv"
        
        # 1. Lê a matriz
        with open(nome_matriz, "rb") as f:
            X = scipy.io.mmread(f).T.tocsr()
            
        # 2. Lê as células
        obs = pd.read_csv(nome_celulas, header=None, sep="\t", names=["barcode"])
        obs.set_index("barcode", inplace=True)
        
        # 3. Lê os genes (Forçando 1 coluna para evitar o erro do Pandas)
        var = pd.read_csv(nome_genes, header=None, sep="\t", names=["gene_symbol"])
        var.set_index("gene_symbol", inplace=True)
        
        # 4. Monta o objeto
        adata = sc.AnnData(X=X, obs=obs, var=var)
        adata.var_names_make_unique()
        
        # 5. Acopla Metadados
        if arquivo_metadados and os.path.exists(arquivo_metadados):
            print(f"Acoplando metadados de: {arquivo_metadados}")
            metadata = pd.read_csv(arquivo_metadados, sep="\t", index_col=0)
            adata.obs = metadata.reindex(adata.obs_names)
            
        print("Objeto AnnData montado com sucesso (Modo Reparo)!\n")
        return adata
        
    finally:
        os.chdir(diretorio_original)
        
        
#Exemplo de uso no arquivo principal:

import os
# Importa o seu arquivo de utilidades (supondo que salvou como sc_loaders.py)
import sc_loaders 

pasta_dados = os.path.join(os.path.dirname(__file__), "AD database")

# Se o dataset estiver quebrado (1 coluna):
adata = sc_loaders.carregar_10x_reparo(pasta_dados, arquivo_metadados="adsn_metadata.txt")

# Se o dataset for perfeito:
# adata = sc_loaders.carregar_10x_padrao(pasta_dados)

# A partir daqui, você já tem o seu 'adata' limpo e pronto para o QC!
print(adata)