"""
plot_worst_classes.py — Script para gerar gráfico das piores classes baseado no F1-score.

Uso:
    python -m src.plot_worst_classes
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from backend.src import config

def main():
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.csv")
    out_file = os.path.join(config.RESULTS_DIR, "worst_f1_classes.png")
    
    if not os.path.exists(report_path):
        print(f"[ERRO] Arquivo de relatório não encontrado: {report_path}")
        print("Execute o script de avaliação primeiro: python -m src.evaluate")
        return
        
    print(f"[INFO] Lendo relatório: {report_path}")
    
    try:
        df = pd.read_csv(report_path)
        # Renomear a primeira coluna (que vem sem nome do scikit-learn) para 'class'
        df.rename(columns={df.columns[0]: 'class'}, inplace=True)
    except Exception as e:
        print(f"[ERRO] Falha ao ler o arquivo CSV: {e}")
        return
        
    # Filtrar apenas as linhas de classes únicas (ignorar accuracy, macro avg, weighted avg)
    # As classes normais geralmente têm nome com 1 caractere no nosso caso
    df_classes = df[df['class'].apply(lambda x: len(str(x)) == 1)].copy()
    
    if df_classes.empty:
        print("[ERRO] Nenhuma classe válida encontrada no relatório.")
        return
        
    # Ordenar pelo F1-score (crescente) para pegar as piores
    df_worst = df_classes.sort_values(by='f1-score', ascending=True).head(10)
    
    # Configurar estilo do gráfico
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Criar gráfico de barras horizontais
    bars = ax.barh(df_worst['class'], df_worst['f1-score'], color='salmon')
    
    ax.set_title("Top 10 Piores Classes (Menor F1-Score)", fontsize=16)
    ax.set_xlabel("F1-Score", fontsize=12)
    ax.set_ylabel("Classe (Letra)", fontsize=12)
    ax.set_xlim(0, 1.05)
    
    # Inverter o eixo Y para que a pior (menor score) fique no topo do gráfico
    ax.invert_yaxis()
    
    # Adicionar o valor exato no final de cada barra
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
                
    plt.tight_layout()
    
    try:
        plt.savefig(out_file, dpi=150)
        print(f"[SUCESSO] Gráfico salvo em: {out_file}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar gráfico: {e}")

if __name__ == "__main__":
    main()
