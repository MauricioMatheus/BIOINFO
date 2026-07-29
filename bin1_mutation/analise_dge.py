import pandas as pd
import numpy as np
import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# =========================================================
# ETAPA 1: Carregamento dos Dados Processados
# =========================================================
print("Carregando o banco de dados anotado...")
adata = sc.read_h5ad("bin1_mutation/adata_limpo_anotado.h5ad")

# Se quiser rodar global (todas as micróglias juntas), basta comentar a linha abaixo.
#estado_alvo = 'Micróglia Associada à Doença - DAM (SPP1+ alto, FTH1+)'
#adata = adata[adata.obs['Estado_Microglial'] == estado_alvo].copy()

# =========================================================
# ETAPA 2: Extração das Contagens Brutas e Metadados
# =========================================================
print("Extraindo contagens brutas (raw counts)...")

# Resgatando as contagens ANTES da normalização/log1p
# O PyDESeq2 exige números inteiros.
if adata.raw is not None:
    X_raw = adata.raw.X
    var_names = adata.raw.var_names
else:
    X_raw = adata.X
    var_names = adata.var_names

# Convertendo para DataFrame do Pandas para facilitar o agrupamento matemático
if hasattr(X_raw, 'toarray'):
    df_counts = pd.DataFrame(X_raw.toarray(), index=adata.obs.index, columns=var_names)
else:
    df_counts = pd.DataFrame(X_raw, index=adata.obs.index, columns=var_names)

# Adicionando a coluna do paciente (donor_id) para agrupar
df_counts['donor_id'] = adata.obs['donor_id'].values

# =========================================================
# ETAPA 3: Pseudobulk
# =========================================================
print("Construindo a matriz de pseudobulk...")

# Agrupando por paciente e somando as contagens de todas as células daquele paciente
pseudobulk_counts = df_counts.groupby('donor_id').sum()

# Arredondando para garantir que sejam inteiros (exigência matemática do DESeq2)
pseudobulk_counts = pseudobulk_counts.round().astype(int)

# Removendo pacientes que por acaso ficaram com zero células nesse cluster
pseudobulk_counts = pseudobulk_counts.loc[(pseudobulk_counts.sum(axis=1) > 0)]

# =========================================================
# ETAPA 4: Limpeza e Construção da Tabela de Metadados
# =========================================================

# O PyDESeq2 precisa saber quem é quem (Qual doador é 0/0, qual é 1/1)
metadados = adata.obs[['donor_id', 'Mutacao_BIN1']].drop_duplicates().set_index('donor_id')
metadados = metadados.loc[pseudobulk_counts.index]

# Filtrando os genótipos de interesse
doadores_validos = metadados[metadados['Mutacao_BIN1'].isin(['0/0', '1/1'])].index
pseudobulk_counts = pseudobulk_counts.loc[doadores_validos]
metadados = metadados.loc[doadores_validos].copy()

# Removendo qualquer resquício de categorias antigas (como 0/1) da memória [estamos analisando controle vs homozigoto mutado]
metadados['Mutacao_BIN1'] = metadados['Mutacao_BIN1'].astype(str)

# Checagem de segurança para ver quantos doadores sobraram em cada grupo
print("\nDistribuição de doadores por genótipo neste cluster:")
print(metadados['Mutacao_BIN1'].value_counts())

# =========================================================
# ETAPA 4.1: Filtro de Genes com Baixa Expressão (Evita Singular Matrix)
# =========================================================
print("\nFiltrando genes sem expressão...")
qtd_genes_antes = pseudobulk_counts.shape[1]

# Mantém apenas os genes que tenham pelo menos 10 leituras somando todos os pacientes
# Isso limpa o "ruído de fundo" e salva a matemática do PyDESeq2
genes_para_manter = pseudobulk_counts.columns[pseudobulk_counts.sum(axis=0) >= 10]
pseudobulk_counts = pseudobulk_counts[genes_para_manter]

qtd_genes_depois = pseudobulk_counts.shape[1]
print(f"Genes mantidos: {qtd_genes_depois} (Eram {qtd_genes_antes} - Foram removidos {qtd_genes_antes - qtd_genes_depois})")

# =========================================================
# ETAPA 5: Inicialização e Execução do PyDESeq2
# =========================================================
print(f"\nIniciando modelagem DESeq2 com {len(metadados)} amostras pseudobulk...")

# Criando o objeto Dataset
dds = DeseqDataSet(
    counts=pseudobulk_counts,
    metadata=metadados,
    design_factors="Mutacao_BIN1",
    ref_level=["Mutacao_BIN1", "0/0"] # Definindo 0/0 como o Controle 
)

# Rodando o modelo estatístico
dds.deseq2()

# Extraindo os resultados estatísticos comparando 1/1 vs 0/0
stat_res = DeseqStats(dds, contrast=["Mutacao_BIN1", "1/1", "0/0"])
stat_res.summary()

# Transformando os resultados em um DataFrame do Pandas para visualizar e salvar
df_resultados = stat_res.results_df
df_resultados = df_resultados.dropna(subset=['padj']) # Remove genes com p-valor nulo

print("\n=== TOP 5 GENES MAIS SIGNIFICATIVOS ===")
print(df_resultados.sort_values('padj').head())

# Salvando a tabela completa
df_resultados.to_csv("bin1_mutation/DGE_Resultados_PyDESeq2.csv")
print("\nMatriz de expressão diferencial salva com sucesso!")