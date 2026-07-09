"""
plot_logs.py — Script para gerar gráficos visuais a partir dos logs de sessão (.csv).

Uso:
    python -m src.plot_logs                     (Processa todos os logs da pasta e salva)
    python -m src.plot_logs --log caminho.csv   (Processa apenas um log específico)
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from backend.src import config
from backend.src.utils import garantir_diretorio

def get_all_logs(log_dir):
    """Encontra todos os arquivos CSV no diretório."""
    search_pattern = os.path.join(log_dir, "*.csv")
    files = glob.glob(search_pattern)
    return files

def process_single_log(csv_path):
    """Processa um único log e salva a imagem."""
    print(f"[INFO] Processando log: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERRO] Falha ao ler o arquivo CSV: {e}")
        return

    if df.empty:
        print("[AVISO] O log está vazio.")
        return

    # Substituir valores vazios nas probabilidades por 0 para os gráficos
    cols_to_fill = ["prob_top1", "prob_top2", "prob_top3", "margin_top1_top2"]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Pegar as classes únicas que apareceram no top1
    top1_classes = df['top1'].dropna().unique()
    
    # Definir eixo X dinâmico (preferir elapsed_seconds)
    x_col = 'elapsed_seconds' if 'elapsed_seconds' in df.columns else 'frame'
    x_label = 'Tempo (s)' if x_col == 'elapsed_seconds' else 'Número do Frame'
    
    # Configurar estilo do gráfico (estilo minimalista)
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    fig.suptitle(f"Análise Temporal de Probabilidades", fontsize=16)

    # 1. Gráfico de Probabilidades do Top 1, Top 2 e Top 3
    ax1.plot(df[x_col], df['prob_top1'], label='Probabilidade 1º Lugar', color='green', linewidth=2)
    ax1.plot(df[x_col], df['prob_top2'], label='Probabilidade 2º Lugar', color='orange', linewidth=1.5, linestyle='--')
    ax1.plot(df[x_col], df['prob_top3'], label='Probabilidade 3º Lugar', color='red', linewidth=1.5, linestyle=':')
    
    ax1.set_ylabel("Probabilidade")
    ax1.set_title("Flutuação de Confiança (Top-3)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()

    # 2. Gráfico da Margem de Incerteza
    ax2.fill_between(df[x_col], df['margin_top1_top2'], color='skyblue', alpha=0.4)
    ax2.plot(df[x_col], df['margin_top1_top2'], color='dodgerblue', label='Margem Top1 - Top2', linewidth=2)
    
    ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Incerteza Crítica (< 10%)')
    ax2.set_ylabel("Margem")
    ax2.set_title("Margem de Segurança (Diferença entre o 1º e 2º)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()

    # 3. Gráfico de Trocas de Classe ao longo do tempo (Top 1)
    # Convertendo letras em índices numéricos para plotagem em Y
    if len(top1_classes) > 0:
        mapping = {c: i for i, c in enumerate(sorted(top1_classes))}
        df_valid = df.dropna(subset=['top1'])
        y_vals = df_valid['top1'].map(mapping)
        
        ax3.scatter(df_valid[x_col], y_vals, c='purple', alpha=0.7, label='Letra Detectada (Top 1)')
        
        # Opcional: Adicionar linha da predição suavizada se estiver ativa
        if 'smoothed_prediction' in df.columns:
            df_smoothed = df.dropna(subset=['smoothed_prediction'])
            y_smoothed = df_smoothed['smoothed_prediction'].map(mapping)
            ax3.plot(df_smoothed[x_col], y_smoothed, c='gold', linewidth=2, label='Predição Suavizada (Votação)')
            
        ax3.set_yticks(list(mapping.values()))
        ax3.set_yticklabels(list(mapping.keys()))
        ax3.set_ylabel("Letra Predita")
        ax3.set_title("Estabilidade de Classe e Suavização")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Nenhuma mão detectada no log.', horizontalalignment='center', verticalalignment='center')

    ax3.set_xlabel(x_label)

    plt.tight_layout()
    
    # Salvar sempre por padrão
    out_dir = os.path.join(config.RESULTS_DIR, "analysis", "summaries")
    garantir_diretorio(out_dir)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_file = os.path.join(out_dir, f"{base_name}_plot.png")
    plt.savefig(out_file, dpi=150)
    print(f"[INFO] Gráfico salvo em: {out_file}")
    plt.close(fig)

    # 4. Gráfico de Frequência de Classes (Gráfico de Barras)
    if len(top1_classes) > 0:
        fig2, ax_bar = plt.subplots(figsize=(10, 6))
        counts = df['top1'].value_counts()
        bars = ax_bar.bar(counts.index, counts.values, color='mediumpurple')
        
        ax_bar.set_title(f"Frequência das Classes (Top 1)")
        ax_bar.set_ylabel("Número de Frames")
        ax_bar.set_xlabel("Classe Detectada")
        
        # Adicionar o valor acima de cada barra
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')
            
        plt.tight_layout()
        freq_out_file = os.path.join(out_dir, f"{base_name}_class_frequency.png")
        plt.savefig(freq_out_file, dpi=150)
        print(f"[INFO] Gráfico de frequência salvo em: {freq_out_file}")
        plt.close(fig2)

def main():
    parser = argparse.ArgumentParser(description="Gera gráficos a partir dos logs do Modo Análise.")
    parser.add_argument("--log", type=str, default=None, help="Caminho para o arquivo CSV de log.")
    args = parser.parse_args()

    log_dir = os.path.join(config.RESULTS_DIR, "analysis", "logs")
    
    if args.log:
        process_single_log(args.log)
    else:
        logs = get_all_logs(log_dir)
        if not logs:
            print(f"[AVISO] Nenhum arquivo de log (.csv) encontrado em {log_dir}")
            return
        print(f"[INFO] Foram encontrados {len(logs)} logs na pasta. Processando todos...")
        for log_file in logs:
            process_single_log(log_file)
        print("[INFO] Processamento em lote concluído!")

if __name__ == "__main__":
    main()
