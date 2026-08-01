import os
import shutil

# nome das pastas das resoluções
resolucoes = [
    "bin1_mutation/figures/resolucao_0.275", 
    "bin1_mutation/figures/resolucao_0.3", 
    "bin1_mutation/figures/resolucao_0.35"
]

# Dicionário com o MAPEAMENTO EXATO dos nomes dos arquivos para suas respectivas pastas
mapeamento_arquivos = {
    # 01_Coorte_e_Demografia
    "heatmap_ADNC_bin1.png": "01_Coorte_e_Demografia",
    "heatmap_Braak_stage_bin1.png": "01_Coorte_e_Demografia",
    "heatmap_CERAD_score_bin1.png": "01_Coorte_e_Demografia",
    "heatmap_Thal_phase_bin1.png": "01_Coorte_e_Demografia",

    # 02_Validacao_do_Clustering
    "Fine Tuning da Estabilidade dos Clusters - Micróglias.png": "02_Validacao_do_Clustering",
    "UMAP.png": "02_Validacao_do_Clustering",
    "DOTPLOT_top5_marcadores_limpo.png": "02_Validacao_do_Clustering",
    "Heatmap_Contigencia_Best_Resolution.png": "02_Validacao_do_Clustering", 

    # 03_Abundancia_Celular_e_Vies
    "Painel_Integrado_Abundancia_Genotipo.png": "03_Abundancia_Celular_e_Vies",
    "Painel_Integrado_Recessivo.png": "03_Abundancia_Celular_e_Vies",
    "Painel_Stripplots_ADNC.png": "03_Abundancia_Celular_e_Vies",
    "Painel_Stripplots_Braak.png": "03_Abundancia_Celular_e_Vies",
    "Painel_Stripplots_CERAD.png": "03_Abundancia_Celular_e_Vies",
    "Painel_Stripplots_Thal.png": "03_Abundancia_Celular_e_Vies",
    "Proporcao_clusters_ADNC_limpo.png": "03_Abundancia_Celular_e_Vies",
    "Proporção_clusters_BIN1_limpo.png": "03_Abundancia_Celular_e_Vies", 
    "Proporcao_clusters_Braak_limpo.png": "03_Abundancia_Celular_e_Vies",
    "Proporcao_clusters_CERAD_limpo.png": "03_Abundancia_Celular_e_Vies",
    "Proporcao_clusters_Thal_limpo.png": "03_Abundancia_Celular_e_Vies",

    # 04_Expressao_Genica
    "Boxplot_BIN1_Clusters_e_Recessivo.png": "04_Expressao_Genica",
    "Expressão_BIN1_Ancestralidade_limpo.png": "04_Expressao_Genica",
    "Expressão_SPP1_Ancestralidade_limpo.png": "04_Expressao_Genica"
}

# Lista das 4 subpastas que precisam ser criadas
subpastas = [
    "01_Coorte_e_Demografia",
    "02_Validacao_do_Clustering",
    "03_Abundancia_Celular_e_Vies",
    "04_Expressao_Genica"
]

for res in resolucoes:
    # Pula se a pasta da resolução não existir
    if not os.path.exists(res):
        print(f"Pasta {res} não encontrada. Pulando...")
        continue
        
    print(f"\nOrganizando {res}...")
    
    # Criar as subpastas vazias
    for subpasta in subpastas:
        os.makedirs(os.path.join(res, subpasta), exist_ok=True)
        
    # Mover os arquivos para as subpastas corretas
    for arquivo in os.listdir(res):
        caminho_arquivo = os.path.join(res, arquivo)
        
        # Ignora se for uma pasta
        if os.path.isdir(caminho_arquivo):
            continue
            
        # Move o arquivo se ele estiver no dicionário
        if arquivo in mapeamento_arquivos:
            pasta_destino = mapeamento_arquivos[arquivo]
            shutil.move(caminho_arquivo, os.path.join(res, pasta_destino, arquivo))
            print(f"  ✓ Movido: {arquivo} -> {pasta_destino}")
        else:
            print(f"  ? Arquivo não mapeado ignorado: {arquivo}")

print("\nOrganização concluída!")