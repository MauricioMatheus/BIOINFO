import pandas as pd
import scanpy as sc
import anndata as ad

df_mutacao = pd.read_csv("mini_mutacao.tsv", sep="\t")

# 2. Faz uma limpeza rápida (O VCF tem os pacientes nas colunas, o h5ad tem nas linhas. é preciso inverter)
# Definindo o ID da mutação como o "nome" da linha
df_mutacao.set_index('ID', inplace=True)

## Joga fora as colunas técnicas do VCF e deixa só as colunas dos pacientes

df_pacientes = df_mutacao.drop(columns=['#CHROM', 'POS', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'])

# invertendo a tabela (Transposição)
df_pacientes = df_pacientes.T
df_pacientes.columns = ['Status_rs6733839'] # Renomeia a coluna 

#Remove o prefixo "1_" dos nomes como por exemplo em 1_H20.33.045 do .h5ad donor_id

#df_pacientes.index = df_pacientes.index.str.split('_').str[-1] -- o split é propenso a falhas em caso de erros ou mudanças de digitação

#usando expressões regulares (regex) para evitar esses erros
df_pacientes.index = df_pacientes.index.str.extract(r'(H\d{2}\.\d{2}\.\d{3})')[0]

# Remove linhas onde o índice virou NaN (garante que só fiquem IDs válidos)
df_pacientes = df_pacientes[df_pacientes.index.notna()]

# Remove IDs duplicados caso existam colunas repetidas no VCF original
df_pacientes = df_pacientes[~df_pacientes.index.duplicated(keep='first')]

print("=== PACIENTES E SUAS MUTAÇÕES ===")
print(df_pacientes.head())

adata = sc.read_h5ad("Microglial.h5ad")

# Mapeamento (A Injeção do DNA no RNA)
# cria uma nova coluna nos metadados (.obs) chamada 'Mutacao_BIN1'
# O map() cruza o 'donor_id' do h5ad com o índice da tabela df_pacientes
adata.obs['Mutacao_BIN1'] = adata.obs['donor_id'].map(df_pacientes['Status_rs6733839'])


# Tratamento de Dados Faltantes (Fallback)
# alguns pacientes têm RNA, mas não têm sequenciamento de DNA.
# O map() vai preencher eles com 'NaN' (Not a Number). trocando isso por uma informação clara:
adata.obs['Mutacao_BIN1'] = adata.obs['Mutacao_BIN1'].fillna('Sem_VCF')

adata.obs['Mutacao_BIN1'] = adata.obs['Mutacao_BIN1'].astype('category')

print("=== DISTRIBUIÇÃO DA MUTAÇÃO NAS MICRÓGLIAS ===")
print(adata.obs['Mutacao_BIN1'].value_counts())

adata.write_h5ad("BIN1_Mutation.h5ad", compression='gzip')